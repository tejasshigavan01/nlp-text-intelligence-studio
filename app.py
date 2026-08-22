"""
NLP Text Intelligence Studio
=============================
A production-ready Streamlit application built on top of an original NLP
notebook pipeline: text cleaning -> stopword removal -> lemmatization ->
TF-IDF vectorization.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analytics import basic_dataset_stats, detect_text_column
from src.preprocessing import get_cleaner, get_pipeline_preview, PIPELINE_STEPS
from src.tfidf import compute_tfidf, word_frequency
from src.utils import asset_path, df_to_csv_bytes, load_css, read_uploaded_file

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NLP Text Intelligence Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css(asset_path("assets", "style.css"))

DEFAULT_CORPUS = [
    "I love NLP! It is amazing.",
    "NLP is used in chatbots and search engines.",
    "I do not like boring lectures.",
    "This NLP session is very interesting!",
]

TOP_N_CHOICES = [5, 10, 20, 30]


# ---------------------------------------------------------------------------
# Small reusable UI helpers
# ---------------------------------------------------------------------------
def render_hero(title: str, subtitle: str, badge: str = "NLP · Machine Learning · Streamlit") -> None:
    st.markdown(
        f"""
        <div class="nlp-hero">
            <div class="nlp-badge">{badge}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (value, label) in zip(cols, items):
        col.markdown(
            f"""
            <div class="nlp-metric">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feature_card(col, icon: str, title: str, description: str) -> None:
    col.markdown(
        f"""
        <div class="nlp-card">
            <div class="nlp-icon">{icon}</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="nlp-footer">
            NLP Text Intelligence Studio • Built with Python, NLP & Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_clean_batch(texts: tuple[str, ...]) -> list[str]:
    cleaner = get_cleaner()
    return cleaner.clean_batch(texts)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home() -> None:
    render_hero(
        "NLP Text Intelligence Studio",
        "Transform raw text into meaningful NLP insights using advanced "
        "preprocessing and TF-IDF feature extraction.",
    )

    cleaner = get_cleaner()
    cleaned_demo = cleaner.clean_batch(DEFAULT_CORPUS)
    try:
        demo_result = compute_tfidf(cleaned_demo)
        vocab_size = demo_result.vocabulary_size
    except ValueError:
        vocab_size = 0

    resource_status = "Ready ✅" if cleaner.resources_ready else "Fallback mode ⚠️"

    st.markdown('<div class="nlp-section-title">Project Snapshot</div>', unsafe_allow_html=True)
    render_metric_row(
        [
            (str(len(DEFAULT_CORPUS)), "Sample Documents"),
            (str(vocab_size), "Vocabulary Size"),
            (resource_status, "NLTK Resources"),
            ("TF-IDF", "Vectorization Method"),
        ]
    )

    st.markdown('<div class="nlp-section-title">What This Studio Does</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="nlp-section-sub">A complete, end-to-end NLP preprocessing '
        "and feature-extraction pipeline you can explore interactively.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    render_feature_card(c1, "📝", "Text Analyzer", "Clean and lemmatize any custom sentence and inspect every pipeline step.")
    render_feature_card(c2, "📊", "Dataset Analysis", "Upload a CSV or TXT file and process an entire corpus in one click.")
    render_feature_card(c3, "🔤", "TF-IDF Explorer", "Visualize vocabulary, document vectors and top-ranked terms interactively.")

    st.markdown('<div class="nlp-section-title">NLP Pipeline Overview</div>', unsafe_allow_html=True)
    pipeline_labels = " → ".join(step[0] for step in PIPELINE_STEPS)
    st.info(pipeline_labels)

    render_footer()


def page_text_analyzer() -> None:
    render_hero(
        "Text Analyzer",
        "Enter any sentence and watch it move through the full cleaning "
        "and lemmatization pipeline, step by step.",
        badge="Live Preprocessing",
    )

    user_text = st.text_area(
        "Enter text to analyze",
        value="I love learning Natural Language Processing!",
        height=110,
        placeholder="Type or paste a sentence here...",
    )

    analyze_clicked = st.button("🔍 Analyze Text", type="primary")

    if analyze_clicked or user_text.strip():
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
            return

        cleaner = get_cleaner()
        cleaned = cleaner.clean(user_text)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="nlp-section-title">Original Text</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="nlp-card"><p>{user_text}</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="nlp-section-title">Cleaned Text</div>', unsafe_allow_html=True)
            display_value = cleaned if cleaned else "(nothing left after cleaning)"
            st.markdown(f'<div class="nlp-card"><p>{display_value}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="nlp-section-title">Preprocessing Pipeline Visualization</div>', unsafe_allow_html=True)
        steps = get_pipeline_preview(user_text)
        for i, step in enumerate(steps):
            st.markdown(
                f"""
                <div class="nlp-pipeline-step">
                    <div class="step-title">{i + 1}. {step['title']}</div>
                    <div class="step-value">{step['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if i < len(steps) - 1:
                st.markdown('<div class="nlp-arrow">↓</div>', unsafe_allow_html=True)

        if cleaned:
            st.markdown('<div class="nlp-section-title">TF-IDF for This Sentence</div>', unsafe_allow_html=True)
            st.caption(
                "TF-IDF needs more than one document to be meaningful, so this "
                "sentence is scored against the built-in sample corpus below."
            )
            try:
                corpus = [cleaner.clean(t) for t in DEFAULT_CORPUS] + [cleaned]
                result = compute_tfidf(corpus)
                top_terms = result.top_terms_per_document(len(corpus) - 1, top_n=10)
                if not top_terms.empty:
                    fig = px.bar(
                        top_terms,
                        x="tfidf_score",
                        y="term",
                        orientation="h",
                        color="tfidf_score",
                        color_continuous_scale="Purples",
                    )
                    fig.update_layout(
                        yaxis={"categoryorder": "total ascending"},
                        height=350,
                        margin=dict(l=10, r=10, t=20, b=10),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Every word in this sentence is a stopword relative to the sample corpus.")
            except ValueError as exc:
                st.info(str(exc))

    render_footer()


def page_dataset_analysis() -> None:
    render_hero(
        "Dataset Analysis",
        "Upload a CSV or TXT file to inspect, clean and process an entire "
        "collection of documents at once.",
        badge="Batch Processing",
    )

    uploaded_file = st.file_uploader("Upload a dataset", type=["csv", "txt"])

    use_sample = st.checkbox("Use built-in sample dataset instead", value=uploaded_file is None)

    df = None
    if uploaded_file is not None and not use_sample:
        df = read_uploaded_file(uploaded_file)
    elif use_sample:
        try:
            df = pd.read_csv(asset_path("data", "sample_data.csv"))
        except FileNotFoundError:
            st.error("Sample dataset not found.")

    if df is None:
        st.info("Upload a .csv or .txt file, or check the sample dataset box, to get started.")
        render_footer()
        return

    text_col = detect_text_column(df)
    columns = list(df.columns)
    if text_col is None:
        text_col = columns[0]

    st.markdown('<div class="nlp-section-title">Dataset Preview</div>', unsafe_allow_html=True)
    selected_col = st.selectbox(
        "Text column",
        options=columns,
        index=columns.index(text_col) if text_col in columns else 0,
        help="Column that contains the free text to analyze.",
    )
    st.dataframe(df.head(10), use_container_width=True)

    stats = basic_dataset_stats(df, selected_col)
    render_metric_row(
        [
            (str(stats["num_rows"]), "Rows"),
            (str(stats["num_columns"]), "Columns"),
            (str(stats["missing_values"]), "Missing Values"),
            (f"{stats['avg_word_length']:.1f}", "Avg. Words / Doc"),
        ]
    )

    if stats["num_non_empty_texts"] == 0:
        st.error("The selected text column has no usable text after removing empty rows.")
        render_footer()
        return

    st.markdown('<div class="nlp-section-title">Batch NLP Processing</div>', unsafe_allow_html=True)
    if st.button("⚙️ Process Entire Dataset", type="primary"):
        texts = df[selected_col].fillna("").astype(str).tolist()
        with st.spinner("Cleaning and lemmatizing all documents..."):
            cleaned_texts = cached_clean_batch(tuple(texts))
        processed_df = df.copy()
        processed_df["clean_text"] = cleaned_texts
        st.session_state["processed_df"] = processed_df
        st.session_state["processed_text_col"] = selected_col
        st.success(f"Processed {len(processed_df)} documents successfully.")

    processed_df = st.session_state.get("processed_df")
    if processed_df is not None and st.session_state.get("processed_text_col") == selected_col:
        st.markdown('<div class="nlp-section-title">Processed Dataset</div>', unsafe_allow_html=True)
        st.dataframe(processed_df.head(15), use_container_width=True)

        st.download_button(
            "⬇️ Download processed_nlp_dataset.csv",
            data=df_to_csv_bytes(processed_df),
            file_name="processed_nlp_dataset.csv",
            mime="text/csv",
        )

        non_empty_clean = [t for t in processed_df["clean_text"].tolist() if t.strip()]
        if non_empty_clean:
            st.markdown('<div class="nlp-section-title">TF-IDF on Processed Dataset</div>', unsafe_allow_html=True)
            try:
                result = compute_tfidf(non_empty_clean)
                st.caption(
                    f"Vocabulary size: **{result.vocabulary_size}** across "
                    f"**{result.num_documents}** documents."
                )
                st.dataframe(result.dataframe.round(4), use_container_width=True)
                st.download_button(
                    "⬇️ Download TF-IDF matrix (CSV)",
                    data=df_to_csv_bytes(result.dataframe),
                    file_name="tfidf_matrix.csv",
                    mime="text/csv",
                )
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.warning("All documents became empty after cleaning, so TF-IDF could not run.")

    render_footer()


def page_tfidf_explorer() -> None:
    render_hero(
        "TF-IDF Explorer",
        "Dive into vocabulary, document vectors and the highest-ranked "
        "terms produced by TfidfVectorizer.",
        badge="Feature Extraction",
    )

    source = st.radio(
        "Corpus source",
        options=["Built-in sample corpus", "Currently processed dataset"],
        horizontal=True,
    )

    cleaner = get_cleaner()
    if source == "Built-in sample corpus":
        documents = cleaner.clean_batch(DEFAULT_CORPUS)
        raw_docs = DEFAULT_CORPUS
    else:
        processed_df = st.session_state.get("processed_df")
        if processed_df is None:
            st.info("No processed dataset yet — go to **Dataset Analysis** and process a file first.")
            render_footer()
            return
        raw_docs = processed_df[st.session_state.get("processed_text_col")].fillna("").astype(str).tolist()
        documents = processed_df["clean_text"].fillna("").astype(str).tolist()

    non_empty_docs = [d for d in documents if d.strip()]
    if not non_empty_docs:
        st.warning("This corpus has no usable text after cleaning.")
        render_footer()
        return

    try:
        result = compute_tfidf(documents)
    except ValueError as exc:
        st.warning(str(exc))
        render_footer()
        return

    render_metric_row(
        [
            (str(result.vocabulary_size), "Vocabulary Size"),
            (str(result.num_documents), "Documents"),
            (str(len(raw_docs)), "Total Rows"),
        ]
    )

    tabs = st.tabs(["📋 Vocabulary & Matrix", "📈 Top Terms", "🔍 Per-Document View"])

    with tabs[0]:
        st.markdown("**Vocabulary**")
        st.write(", ".join(result.feature_names.tolist()))
        st.markdown("**TF-IDF DataFrame**")
        st.dataframe(result.dataframe.round(4), use_container_width=True)

    with tabs[1]:
        top_n = st.select_slider("Top N words", options=TOP_N_CHOICES, value=10)
        top_terms = result.top_terms(top_n=top_n)
        fig = px.bar(
            top_terms,
            x="tfidf_score",
            y="term",
            orientation="h",
            color="tfidf_score",
            color_continuous_scale="Teal",
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(300, top_n * 22),
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        doc_idx = st.number_input(
            "Document index",
            min_value=0,
            max_value=result.num_documents - 1,
            value=0,
            step=1,
        )
        st.caption(f"Original text: {raw_docs[doc_idx] if doc_idx < len(raw_docs) else '(n/a)'}")
        per_doc = result.top_terms_per_document(int(doc_idx), top_n=10)
        if per_doc.empty:
            st.info("This document has no non-zero TF-IDF terms.")
        else:
            st.dataframe(per_doc, use_container_width=True)

    render_footer()


def page_dashboard() -> None:
    render_hero(
        "NLP Dashboard",
        "A consolidated analytics view combining document statistics, word "
        "frequency and TF-IDF importance.",
        badge="Analytics Overview",
    )

    source = st.radio(
        "Data source",
        options=["Built-in sample corpus", "Currently processed dataset"],
        horizontal=True,
        key="dashboard_source",
    )

    cleaner = get_cleaner()
    if source == "Built-in sample corpus":
        documents = cleaner.clean_batch(DEFAULT_CORPUS)
    else:
        processed_df = st.session_state.get("processed_df")
        if processed_df is None:
            st.info("No processed dataset yet — go to **Dataset Analysis** and process a file first.")
            render_footer()
            return
        documents = processed_df["clean_text"].fillna("").astype(str).tolist()

    non_empty_docs = [d for d in documents if d.strip()]
    if not non_empty_docs:
        st.warning("No usable text to analyze.")
        render_footer()
        return

    avg_len = sum(len(d.split()) for d in non_empty_docs) / len(non_empty_docs)

    render_metric_row(
        [
            (str(len(non_empty_docs)), "Total Documents"),
            (f"{avg_len:.1f}", "Avg. Words / Doc"),
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="nlp-section-title">Most Frequent Words</div>', unsafe_allow_html=True)
        freq_df = word_frequency(non_empty_docs, top_n=15)
        if freq_df.empty:
            st.info("No words to display.")
        else:
            fig = px.bar(
                freq_df,
                x="frequency",
                y="term",
                orientation="h",
                color="frequency",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="nlp-section-title">Top TF-IDF Terms</div>', unsafe_allow_html=True)
        try:
            result = compute_tfidf(documents)
            top_terms = result.top_terms(top_n=15)
            fig2 = px.bar(
                top_terms,
                x="tfidf_score",
                y="term",
                orientation="h",
                color="tfidf_score",
                color_continuous_scale="Purples",
            )
            fig2.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ValueError as exc:
            st.info(str(exc))

    st.markdown('<div class="nlp-section-title">Document Length Distribution</div>', unsafe_allow_html=True)
    length_df = pd.DataFrame({"word_count": [len(d.split()) for d in non_empty_docs]})
    fig3 = px.histogram(length_df, x="word_count", nbins=15, color_discrete_sequence=["#6C5CE7"])
    fig3.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig3, use_container_width=True)

    render_footer()


def page_about() -> None:
    render_hero(
        "How It Works",
        "A quick explanation of every stage in the NLP pipeline, written "
        "for a student portfolio audience.",
        badge="About This Project",
    )

    with st.expander("🧹 Text Cleaning", expanded=True):
        st.write(
            "Raw text is messy: punctuation, numbers, capitalization and "
            "extra whitespace add noise that doesn't carry meaning for most "
            "NLP tasks. Cleaning standardizes the text — lowercasing it and "
            "removing non-alphabetic characters — so that words like "
            "\"NLP\", \"nlp!\" and \"NLP.\" are all treated as the same token."
        )

    with st.expander("🚫 Stopword Removal"):
        st.write(
            "Stopwords are extremely common words (\"the\", \"is\", \"and\", "
            "\"a\"...) that appear in almost every sentence and carry very "
            "little distinguishing information. Removing them lets the "
            "model focus on the words that actually differentiate one "
            "document from another."
        )

    with st.expander("🔤 Lemmatization"):
        st.write(
            "Lemmatization reduces a word to its dictionary base form using "
            "vocabulary and grammar rules — for example, \"running\", "
            "\"ran\" and \"runs\" all become \"run\". This groups together "
            "different grammatical forms of the same underlying word, which "
            "shrinks the vocabulary and improves downstream modeling."
        )

    with st.expander("📐 TF-IDF"):
        st.markdown(
            """
**TF** — *Term Frequency*: how often a word appears in a specific document.

**IDF** — *Inverse Document Frequency*: a measure of how rare a word is
across the whole corpus — common words score low, rare/specific words
score high.

**TF-IDF = TF × IDF**

Multiplying the two gives a score that is high for words that appear
often in one document but rarely across the rest of the corpus — exactly
the words that best characterize that document.
            """
        )

    st.markdown('<div class="nlp-section-title">Technology Stack</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    render_feature_card(c1, "🐍", "Python", "Core application language.")
    render_feature_card(c2, "🖥️", "Streamlit", "Interactive web UI framework.")
    render_feature_card(c3, "🧠", "NLTK", "Stopwords & WordNet lemmatization.")
    render_feature_card(c4, "📊", "scikit-learn", "TF-IDF vectorization.")

    render_footer()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Home": page_home,
    "📝 Text Analyzer": page_text_analyzer,
    "📊 Dataset Analysis": page_dataset_analysis,
    "🔤 TF-IDF Explorer": page_tfidf_explorer,
    "📈 NLP Dashboard": page_dashboard,
    "ℹ️ About Project": page_about,
}


def main() -> None:
    st.sidebar.markdown("## 🧠 NLP Text\nIntelligence Studio")
    st.sidebar.caption("Preprocessing · TF-IDF · Analytics")
    st.sidebar.divider()
    selection = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.sidebar.divider()

    cleaner = get_cleaner()
    if cleaner.resources_ready:
        st.sidebar.success("NLTK resources ready")
    else:
        st.sidebar.warning("Using fallback stopword list (NLTK download unavailable)")

    PAGES[selection]()


if __name__ == "__main__":
    main()

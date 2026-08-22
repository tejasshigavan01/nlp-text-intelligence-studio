# NLP Text Intelligence Studio

A production-ready Streamlit web application that turns a classic NLP
preprocessing + TF-IDF notebook pipeline into an interactive, portfolio-ready
tool for exploring text cleaning, lemmatization and feature extraction.

## Features

- **Home Dashboard** — project overview, live vocabulary/document metrics, and a pipeline summary.
- **Text Analyzer** — type any sentence and watch it move through every cleaning step, plus a live TF-IDF chart.
- **Dataset Analysis** — upload a CSV/TXT file (or use the built-in sample dataset), auto-detect the text column, and batch-clean the whole file.
- **TF-IDF Explorer** — inspect vocabulary, the full TF-IDF matrix, top-N ranked terms, and per-document term breakdowns.
- **NLP Dashboard** — word-frequency chart, top TF-IDF terms, and document-length distribution in one view.
- **About / How It Works** — plain-language explanations of text cleaning, stopwords, lemmatization and TF-IDF.
- Graceful error handling for missing NLTK data, empty files, and CSVs without a usable text column.
- CSV downloads for both the processed dataset (`clean_text` column added) and the TF-IDF matrix.

## Technologies Used

- [Streamlit](https://streamlit.io/) — web application framework
- [Pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data handling
- [NLTK](https://www.nltk.org/) — stopwords & WordNet lemmatization
- [scikit-learn](https://scikit-learn.org/) — `TfidfVectorizer`
- [Plotly](https://plotly.com/python/) — interactive charts

## NLP Pipeline

```
Raw Text
   ↓
Lowercasing
   ↓
Remove Special Characters
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
Clean Text
   ↓
TF-IDF Vectorization
```

This mirrors the original notebook's `clean_text()` function and
`TfidfVectorizer` usage, wrapped in reusable, testable modules.

## Project Structure

```
NLP-Text-Intelligence-Studio/
│
├── app.py                  # Streamlit entry point & page routing
├── requirements.txt
├── README.md
├── .gitignore
├── packages.txt
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Text cleaning, NLTK resource handling
│   ├── tfidf.py             # TF-IDF vectorization helpers
│   ├── analytics.py        # Dataset statistics
│   └── utils.py             # CSS loading, file parsing, CSV export
│
├── assets/
│   └── style.css            # Custom theme
│
└── data/
    └── sample_data.csv      # Built-in demo dataset
```

## How to Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app will automatically download the required NLTK corpora
(`stopwords`, `wordnet`, `omw-1.4`) the first time it runs. If a
network-restricted environment blocks that download, the app falls back to
a small built-in stopword list so it never crashes — a banner in the
sidebar tells you which mode is active.

## Streamlit Cloud Deployment

1. Create a new GitHub repository and push this project to it (the `.gitignore` already excludes virtual environments, caches and NLTK data).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and connect your GitHub repository.
4. Select the branch and set the main file path to `app.py`.
5. Click **Deploy**. Streamlit Cloud will install `requirements.txt` automatically.
6. On first load, the app downloads the NLTK corpora it needs — subsequent reruns reuse them for the life of the container.

## Limitations & Assumptions

- The original notebook used a tiny 4-sentence demo corpus with no ML classifier — this app focuses on preprocessing, TF-IDF and analytics rather than adding a classification model that wasn't in the source notebook.
- TF-IDF is computed fresh per interaction for the corpus currently selected in each page (built-in sample vs. processed upload); very large uploaded datasets (tens of thousands of rows) may be slow in a free-tier Streamlit Cloud container.
- Automatic text-column detection uses a simple heuristic (longest average string length among text columns) — you can always override it with the dropdown.
- Sentiment/category columns (like the `category` column in `sample_data.csv`) are shown in the dataset preview but are not used for classification, since the source notebook did not include a labeled classification task.

---

NLP Text Intelligence Studio • Built with Python, NLP & Streamlit

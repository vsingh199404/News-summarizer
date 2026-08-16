# 📰 News Article Text Summarization

An end-to-end News Article Summarization Application using an **Encoder-Decoder Transformer (T5)** model, built for the **Natural Language Processing** course (**Assignment 2: News Article Text Summarization**).

## Team

| Name | BITS ID | Contribution |
|------|---------|-------------|
| Vishal Singh Tomar | 2025AA05331 | 60% |
| Vivek Sharma | 2025AA05588 | 30% |
| Yash | 2024AD05399 | 10% |

## Features

- **T5-Small Transformer** — Encoder-Decoder architecture (60M parameters) for abstractive summarization
- **Streamlit Web App** — Premium dark-themed UI with glassmorphism design
- **ROUGE Evaluation** — Automatic ROUGE-1, ROUGE-2, ROUGE-L scoring against reference summaries
- **Multiple Input Modes** — Paste text, upload files (.txt/.pdf/.docx), or use sample articles
- **Batch Processing** — Summarize all sample articles with aggregate metrics
- **Training Pipeline** — Fine-tune T5 on CNN/DailyMail dataset

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web app
streamlit run app.py

# Open http://localhost:8501
```

## Project Structure

```
news_summarizer/
├── app.py                 # Streamlit web application
├── model.py               # T5 model wrapper (load, summarize, model info)
├── preprocess.py           # Data loading, cleaning, tokenization
├── evaluate.py             # ROUGE metric computation
├── train.py               # Fine-tuning script (CNN/DailyMail)
├── test_pipeline.py       # End-to-end integration test
├── requirements.txt       # Python dependencies
├── sample_articles/       # 5 curated news articles
├── reference_summaries/   # Human-written reference summaries
└── report/                # Project report
```

## Results (Zero-Shot T5-Small)

| Metric | Average Score |
|--------|-------------|
| ROUGE-1 F1 | 47.21% |
| ROUGE-2 F1 | 23.64% |
| ROUGE-L F1 | 37.03% |

## Training (Optional)

```bash
python train.py --num_train_samples 5000 --epochs 3 --batch_size 4
```

## Tech Stack

Python 3.10 · PyTorch · HuggingFace Transformers · Streamlit · rouge-score

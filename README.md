
# 🩺 MedScan AI

<p align="center">
  <img src="https://img.shields.io/badge/Built%20with-LangGraph-6366f1?style=for-the-badge" />
  <img src="https://img.shields.io/badge/LLM-Claude%20Sonnet%204.5-a78bfa?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Search-Tavily-38bdf8?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-yellow?style=for-the-badge" />
</p>

> An AI-powered medical report analyser built with **LangGraph** + **Claude** + **Streamlit**.  
> Upload a USG report or prescription — the agent extracts patient details, searches the web for medical context, and returns a plain-English summary with built-in safety guardrails.

---

## ✨ Features

- 📄 **Multi-format upload** — PDF, PNG, JPG, TXT (OCR for scanned images)
- 🧠 **LangGraph agent pipeline** — structured 5-node graph with typed state
- 🔍 **Web-augmented analysis** — Tavily search enriches findings with medical context
- 🛡️ **5-layer guardrail system** — input validation + output safety checks
- ⚡ **Claude Sonnet 4.5** — fast, accurate medical document understanding
- 💾 **Downloadable summary** — export patient summary as `.txt`

---

## 🛡️ Guardrail System

MedScan AI includes a dedicated `guardrails.py` module with soft-warning guardrails — violations show alerts but never block the pipeline.

| # | Guardrail | Type | Method |
|---|-----------|------|--------|
| 1 | Non-medical document rejection | Input | Claude Haiku classifier (YES/NO) |
| 2 | PII detection & redaction | Input | Regex — phones, emails, Aadhaar, PAN |
| 3 | Harmful advice detection | Output | Keyword scan → LLM confirmation |
| 4 | Disclaimer enforcement | Output | Auto-appends if missing |
| 5 | Summary length cap | Output | Trims to 400 words to reduce hallucination |

---

## 🔄 Agent Pipeline

```
Upload
  │
  ▼
┌──────────────┐
│ input_guard  │  ← Classifies doc type, redacts PII
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ extract_info │  ← Claude parses → structured JSON
│              │    (name, age, findings, impression…)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  web_search  │  ← Tavily fetches medical context
│              │    (up to 3 queries, 2 results each)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  synthesise  │  ← Claude writes plain-English summary
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ output_guard │  ← Checks safety, disclaimer, length
└──────┬───────┘
       │
      END → Summary + Warnings displayed in UI
```

---

## 📁 Project Structure

```
MedScan-AI/
├── app.py            ← Streamlit frontend + warning banners
├── agent.py          ← LangGraph 5-node compiled graph
├── guardrails.py     ← All guardrail logic (input + output)
├── utils.py          ← File text extraction (PDF / image / txt)
├── requirements.txt
├── .env.example      ← Template for environment variables
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/anweshaban21/MedScan-AI.git
cd MedScan-AI
```

### 2. Install system dependency (for image OCR)
```bash
# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Create a virtual environment & install packages
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
```

### 5. Run the app
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 API Keys Required

| Key | Where to get |
|-----|-------------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com/api-keys) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |

---

## 📂 Supported File Types

| Format | Extraction Method |
|--------|------------------|
| `.txt` | Direct UTF-8 decode |
| `.pdf` | pdfplumber → PyMuPDF fallback |
| `.png` `.jpg` `.jpeg` | Tesseract OCR via pytesseract |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | [Anthropic Claude Sonnet 4.5](https://www.anthropic.com) |
| Guardrail LLM | Anthropic Claude Haiku 4.5 (fast + cheap) |
| Web search | [Tavily](https://tavily.com) |
| Frontend | [Streamlit](https://streamlit.io) |
| PDF parsing | pdfplumber + PyMuPDF |
| OCR | Tesseract + pytesseract |

---

## ⚠️ Disclaimer

This tool is for **informational purposes only** and is **not a substitute for professional medical advice**. Always consult a qualified healthcare provider for medical decisions.

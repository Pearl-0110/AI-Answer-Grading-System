# 🧠 AI-Based Answer Grading System with N-Gram Models & Copy Detection
This is a Streamlit-based application that performs *automated answer grading* using different N-gram models (Unigram, Bigram, Trigram, Quadgram) and also checks for potential *copying between student responses* using *cosine similarity*.

---

## 📦 Requirements
- Python 3
- Streamlit – for web UI
- Pandas – for data handling
- Scikit-learn – for TF-IDF and cosine similarity
- NLTK – for NLP preprocessing


---

## 🚀 Features
- Upload an Excel file with model answers and multiple student answers.
- Select Unigram, Bigram, Trigram, or Quadgram-based similarity model.
- Calculate grades using cosine similarity with a fine-grained grading scale.
- Detect potential or confirmed cases of copying between students.
- Compute average similarity score across all N-gram models.
- Export results to CSV.

---

## 📁 Folder Structure
Ai-Answer-Grading/
│
├── grading_app.py                # Streamlit UI app (main application)
├── average_ngram_grading.py     # Final processor for averaging all n-gram results
├── text_preprocessing.py        # General text preprocessing (base version)
├── preprocessing_unigram.py     # Unigram-based preprocessing
├── preprocessing_bigram.py      # Bigram-based preprocessing
├── preprocessing_trigram.py     # Trigram-based preprocessing
├── preprocessing_quadgram.py    # Quadgram-based preprocessing
├── requirements.txt             # Required Python libraries
├── README.md                    # This documentation file
└── ans.xlsx                     # Sample input Excel file (optional)

---

## 📝 Excel File Format

Make sure your Excel file contains the following columns:

| Model Answer | Student Answer 1 | Student Answer 2 | ... |
|--------------|------------------|------------------|-----|

---

## ▶ How to Run

1. *Install Dependencies*
   bash
   pip install -r requirements.txt

2. **How to Run on Streamlit**
   bash
   streamlit run grading_app.py
---

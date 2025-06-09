import streamlit as st
import pandas as pd
import importlib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Title
st.title("AI-Based Answer Grading System with Averaged N-gram Evaluation")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("### Uploaded Data")
    st.write(df)

    # Preprocessing modules
    ngram_modules = {
        'Unigram': 'preprocessing_unigram',
        'Bigram': 'preprocessing_bigram',
        'Trigram': 'preprocessing_trigram',
        'Quadgram': 'preprocessing_quadgram'
    }

    model_answers = df['Model Answer']
    student_columns = [col for col in df.columns if col.startswith('Student Answer')]
    student_answers_all = df[student_columns]

    results = []

    # For each question
    for i in range(len(model_answers)):
        question_data = {
            'Question': f"Question {i+1}"
        }

        # Store vectors from each model
        student_scores_ngram = {f'S{j+1}': [] for j in range(len(student_columns))}
        student_vectors_ngram = {f'S{j+1}': [] for j in range(len(student_columns))}

        # Loop through all N-gram types
        for ngram_name, module_name in ngram_modules.items():
            preprocess_module = importlib.import_module(module_name)
            preprocess_text = preprocess_module.preprocess_text

            model_processed = ' '.join(preprocess_text(model_answers[i]))
            student_processed = [' '.join(preprocess_text(ans)) for ans in student_answers_all.iloc[i]]

            all_texts = [model_processed] + student_processed
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)

            model_vec = tfidf_matrix[0]
            student_vecs = tfidf_matrix[1:]

            for idx, vec in enumerate(student_vecs):
                sim = cosine_similarity(model_vec, vec)[0][0]
                student_scores_ngram[f'S{idx+1}'].append(sim)
                student_vectors_ngram[f'S{idx+1}'].append(vec)

        # Compute average and assign grade
        for s_key, sims in student_scores_ngram.items():
            avg_sim = sum(sims) / len(sims)
            if avg_sim >= 0.90:
                grade = "Outstanding"
            elif avg_sim >= 0.80:
                grade = "Excellent"
            elif avg_sim >= 0.70:
                grade = "Very Good"
            elif avg_sim >= 0.60:
                grade = "Good"
            elif avg_sim >= 0.50:
                grade = "Satisfactory"
            elif avg_sim >= 0.45:
                grade = "Needs Improvement"
            elif avg_sim >= 0.40:
                grade = "Weak"
            else:
                grade = "Poor"

            question_data[f'Avg Cosine Sim {s_key}'] = round(avg_sim, 4)
            question_data[f'Final Grade {s_key}'] = grade

        # Copy Detection using average of vectors
        copy_cases = []
        student_keys = list(student_vectors_ngram.keys())
        for i1, i2 in combinations(range(len(student_keys)), 2):
            key1, key2 = student_keys[i1], student_keys[i2]
            avg_sim = 0
            for j in range(len(ngram_modules)):
                vec1 = student_vectors_ngram[key1][j]
                vec2 = student_vectors_ngram[key2][j]
                avg_sim += cosine_similarity(vec1, vec2)[0][0]
            avg_sim /= len(ngram_modules)

            if avg_sim == 1.0:
                copy_cases.append(f"{key1}-{key2}: Confirmed Copy")
            elif avg_sim >= 0.95:
                copy_cases.append(f"{key1}-{key2}: Potential Copy")

        question_data['Copy Case Summary'] = ', '.join(copy_cases) if copy_cases else "No Copy Detected"
        results.append(question_data)

    # Display and allow download
    result_df = pd.DataFrame(results)
    st.write("### Final Grading & Copy Detection Results (Averaged across N-grams)")
    st.dataframe(result_df)

    st.download_button(
        label="Download Grading Results",
        data=result_df.to_csv(index=False).encode(),
        file_name="grading_results_averaged.csv",
        mime="text/csv"
    )

import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Excel file
file_path = 'ans.xlsx'
df = pd.read_excel(file_path)

# Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [w for w in words if w not in stop_words]

# Columns
model_answers = df['Model Answer']
student_answers_1 = df['Student Answer 1']
student_answers_2 = df['Student Answer 2']

# Process each row
for i in range(len(df)):
    model = ' '.join(preprocess_text(model_answers[i]))
    stu1 = ' '.join(preprocess_text(student_answers_1[i]))
    stu2 = ' '.join(preprocess_text(student_answers_2[i]))

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([model, stu1, stu2])

    sim_m1 = cosine_similarity(vectors[0], vectors[1])[0][0]
    sim_m2 = cosine_similarity(vectors[0], vectors[2])[0][0]
    sim_12 = cosine_similarity(vectors[1], vectors[2])[0][0]

    # Grades
    def grade(sim): return "Excellent" if sim >= 0.8 else "Needs Review" if sim >= 0.5 else "Poor"
    g1 = grade(sim_m1)
    g2 = grade(sim_m2)

    # Copy Detection
    if sim_12 == 1.0:
        copy_status = "Confirmed Copy"
    elif sim_12 >= 0.95:
        copy_status = "Potential Copy"
    else:
        copy_status = "No Copy Detected"

    # Output
    print(f"\n=== Question {i+1} ===")
    print(f"Model ↔ S1 Similarity: {sim_m1:.4f} | Grade: {g1}")
    print(f"Model ↔ S2 Similarity: {sim_m2:.4f} | Grade: {g2}")
    print(f"S1 ↔ S2 Similarity: {sim_12:.4f} | Copy Status: {copy_status}")

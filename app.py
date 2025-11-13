import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np
import pandas as pd
import string, re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import os

nltk.download("stopwords")
nltk.download("wordnet")


# Feedback Saver (CSV)
def save_feedback_to_csv(data, file_path="feedback_log.csv"):
    df_new = pd.DataFrame([data])

    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(file_path, index=False)
    else:
        df_new.to_csv(file_path, index=False)


# Preprocessing

_stopwords = set(stopwords.words("english"))
_lem = WordNetLemmatizer()

def text_prep(raw_text):
    text = str(raw_text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-z0-9._\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [tok for tok in text.split() if tok not in _stopwords and len(tok) > 2]
    lemmas = [_lem.lemmatize(tok) for tok in tokens]
    return " ".join(lemmas)



# Load Title Model
title_model_dir = "models/title"

device = "cuda" if torch.cuda.is_available() else "cpu"

title_model = AutoModelForSequenceClassification.from_pretrained(
    title_model_dir,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=False,
    _fast_init=False
)

title_model = title_model.to(device)
title_model.eval()

tokenizer = AutoTokenizer.from_pretrained(title_model_dir)
label_encoder = joblib.load(f"{title_model_dir}/label_encoder.pkl")


# Load Grade Models
@st.cache_resource
def load_grade_models():
    model_dir = "models/Grade"
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f"{model_dir}/salary_xgboost.json")
    text_pipeline = joblib.load(f"{model_dir}/text_pipeline.pkl")
    return xgb_model, text_pipeline
def identity_preprocessor(x):
    return x

def whitespace_tokenizer(x):
    return x.split()

grade_model, grade_pipeline = load_grade_models()



# Load Salary Models
@st.cache_resource
def load_salary_models():
    model_dir = "models/salary"
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(f"{model_dir}/salary_xgboost.json")
    embedder = SentenceTransformer(f"{model_dir}/embedding_model")
    return xgb_model, embedder

salary_model, salary_embedder = load_salary_models()


# UI
st.title("Job Title & Compensation Predictor")
st.write("Paste a job description and get Title, Grade, and Salary predictions.")

job_text = st.text_area("Job Description")


# PREDICTION BUTTON
if st.button("Predict Job Title, Grade, and Salary"):
    if not job_text.strip():
        st.warning("Please enter text first.")
    else:
        cleaned = text_prep(job_text)

        # TITLE
        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            logits = title_model(**inputs).logits
            predicted_id = torch.argmax(logits, dim=1).cpu().numpy()[0]

        predicted_title = label_encoder.inverse_transform([predicted_id])[0]
        st.success(f" Predicted Title: **{predicted_title}**")

        # GRADE
        try:
            grade_vec = grade_pipeline.transform([cleaned])
            grade_df = pd.DataFrame(grade_vec)
            grade_pred = int(round(grade_model.predict(grade_df)[0]))
            st.success(f" Predicted Grade: **{grade_pred}**")
        except Exception as e:
            st.error(f"Grade prediction failed: {e}")
            grade_pred = None

        # SALARY
        try:
            emb = salary_embedder.encode([cleaned])
            emb_df = pd.DataFrame(emb)
            salary_pred = salary_model.predict(emb_df)[0]
            salary_fmt = f"{round(salary_pred):,}€"
            st.success(f" Predicted Annual Salary: **{salary_fmt}**")
        except Exception as e:
            st.error(f"Salary prediction failed: {e}")
            salary_fmt = None

        # STORE RESULTS
        st.session_state.predicted_title = predicted_title
        st.session_state.predicted_grade = grade_pred
        st.session_state.predicted_salary = salary_fmt
        st.session_state.has_prediction = True


if st.session_state.get("has_prediction", False):

    st.markdown("---")
    st.subheader("HR Feedback")

    feedback = st.radio(
        "Do you approve the job evaluation results?",
        ("Yes", "No"),
        key="approval_radio"
    )

    if feedback == "No":

        issue = st.selectbox(
            "Which part is incorrect?",
            ("Title", "Grade", "Salary", "Other / Multiple parts"),
            key="which_issue"
        )

        if issue == "Salary":
            problem_options = ["It is high", "It is low"]

        elif issue == "Title":
            problem_options = ["The job title is incorrect"]

        elif issue == "Grade":
            problem_options = ["It is high", "It is low"]

        else:
            problem_options = [
                "It is high", "It is low",
                "The job title is incorrect",
                "The grade is incorrect",
                "Other"
            ]

        selected_problems = st.multiselect(
            "Select the issues you see:",
            problem_options,
            key="problem_choices"
        )

        optional_text = st.text_area(
            "Additional comments (optional):",
            key="optional_feedback",
            height=100
        )

        if st.button("Submit Feedback", key="submit_feedback"):

            feedback_record = {
                "job_description": job_text,
                "predicted_title": st.session_state.predicted_title,
                "predicted_grade": st.session_state.predicted_grade,
                "predicted_salary": st.session_state.predicted_salary,
                "approved": feedback,
                "issue_type": issue,
                "selected_problems": ", ".join(selected_problems),
                "additional_comments": optional_text,
                "timestamp": pd.Timestamp.now()
            }

            save_feedback_to_csv(feedback_record)

            st.success("Thank you. Your feedback has been recorded!")

    else:
        st.write("Thank you for confirming that the prediction is correct!")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import streamlit as st

osteoporosis = pd.read_csv(
    r"C:\Users\KAsab\Desktop\MACHINE LEARNING PROJECTS\osteoporosis_ML_classification\cleaned_survey.csv"
)

num_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessing = ColumnTransformer(
    [
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ]
)

from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(
    [
        ("preprocessor", preprocessing),
        ("random_forest", RandomForestClassifier(random_state=42)),
    ]
)

X = osteoporosis.drop(columns=["Osteoporosis"])
y = osteoporosis["Osteoporosis"]

rf_model.fit(X, y)

joblib.dump(rf_model, "osteoporosis_model.joblib")

st.title("Osteoporosis Model Inference")
with st.sidebar:
    st.header("Data requirements")
    st.caption("A model for predicting osteoposis based on risk factors")
    with st.expander("Data format"):
        st.markdown(" - utf-8")
        st.markdown(" - seperated by coma")
        st.markdown(' - seperated by "." ')
        st.markdown(" - first row - header")
    st.divider()
    st.caption("Developed by Sumanguru")


if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}


def clicked(button):
    st.session_state.clicked[button] = True


st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        osteoporosis = pd.read_csv(uploaded_file, low_memory=True)
        st.header("Uploaded data sample")
        st.write(osteoporosis.head())
        model = joblib.load("osteoporosis_model.joblib")
        pred = model.predict_proba(osteoporosis)
        pred = pd.DataFrame(pred, columns=["No", "Yes"])
        st.header("Predicted values")
        st.write(pred.head())
        pred = pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction",
            pred,
            "prediction.csv",
            "text/csv",
            key="download-csv",
        )

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load

# Load TF-IDF vectorizer and Logistic Regression model
tfidf = load("tfidf_vectorizer.joblib")
log_reg = load("logreg_model.joblib")

app = FastAPI(
    title="Sentiment Classification Microservice",
    description="TF-IDF + Logistic Regression model from Module 3",
    version="1.0"
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    probability: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Transform text into TF-IDF features
    X = tfidf.transform([req.text])

    # Predict probability of positive class (1)
    proba = log_reg.predict_proba(X)[0][1]

    label = "positive" if proba >= 0.5 else "negative"

    return PredictResponse(
        label=label,
        probability=float(proba)
    )
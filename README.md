# Text_Classification_Microservice_Data Mining techniques

FastAPI microservice for sentiment classification built.  
The service uses TF-IDF + Logistic Regression to classify text as positive or negative.

## Files
- `main.py` — FastAPI microservice
- `requirements.txt` — Python dependencies
- `logreg_model.joblib` — Trained Logistic Regression model
- `tfidf_vectorizer.joblib` — TF-IDF vectorizer

## How to Run Locally

```bash
python3 -m venv venv
source venv/bin/activate   # Mac
pip install -r requirements.txt
uvicorn main:app --reload
Check - http://127.0.0.1:8000/docs

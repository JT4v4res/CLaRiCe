from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pred.essay_correction import *
# import tensorflow as tf
import nltk
from app.model_loader import load_model
import spacy
from transformers import AutoTokenizer

app = FastAPI(title="Essay correction API")

model = None
nlp = None

tokenizer = None

class Essay(BaseModel):
    text: str

@app.on_event("startup")
async def startup_libraries():
    global model, nlp, tokenizer
    model = load_model()

    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    nlp = spacy.load("pt_core_news_lg")

    try:
        nltk.download('stopwords')
    except:
        print('stopwords already downloaded')

@app.post("/predict/tf/", status_code=200)
async def predict_essay(request: Essay):
    prediction = tf_essay_correction(request.text)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="The essay was not found.",
        )

    return prediction


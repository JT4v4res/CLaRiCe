from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pred.essay_correction import *

app = FastAPI(title="Essay correction API")

class Essay(BaseModel):
    text: str

@app.post("/predict/tf/", status_code=200)
async def predict_essay(request: Essay):
    prediction = tf_essay_correction(request.text)

    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="The essay was not found.",
        )

    return prediction


from pred.model.tf_pred import *
from typing import Any

def tf_essay_correction(text: str) -> str:
    text = text.lower()

    if text is None:
        return None

    pred_results = tf_predict(text)
    pred_results["status_code"] = 200

    return pred_results

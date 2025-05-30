from fastapi import FastAPI
from pydantic import BaseModel
from second_phase.predictor import Predictor

app = FastAPI()
predictor = Predictor()

class Content(BaseModel):
    content: str

@app.post("/api/v1/predict")
async def predict(input: Content):
    '''
    Predict label based on content
    input: {"content": "text"}
    output: {"label": "world|sports|business|sci/tech", "pred_time": 0.2343}
    '''
    labels = ["World", "Sports", "Business", "Cri/Tech"]
    _, pred_label, pred_time = predictor.predict_full_text(sentence=input.content)

    return {"label": labels[pred_label], "pred_time": pred_time}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

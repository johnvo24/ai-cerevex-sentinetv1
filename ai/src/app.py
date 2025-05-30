from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Content(BaseModel):
    content: str

class UserFeedback(BaseModel):
    ???

@app.post("/api/v1/predict")
async def predict(input: Content):
    '''
    Predict label based on content
    input: {"content": "text"}
    output: {"label": "world|sports|business|sci/tech"}
    '''
    # TODO

    return {"label": sentiment}

@app.post("/api/v1/update")
async def update(input: UserFeedback):
    '''
    Update model based on user feedback
    input: ????
    output: ????
    '''
    # TODO

    return ????

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)

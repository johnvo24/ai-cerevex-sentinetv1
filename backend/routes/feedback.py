from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/feedback", tags=["AI Feedback"])

class Feedback(BaseModel):
    article_id: int
    user_id: int
    feedback_txt: str
    is_correct: bool

@router.post("/feedback")
async def feedback(feedback: Feedback):
    return {"message": "Feedback submitted successfully", "feedback": feedback}

@router.get("/feedback")
async def get_feedback():
    return {"message": "Feedback retrieved successfully", "feedback": []}
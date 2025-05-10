from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class FeedbackCreate(BaseModel):
    article_id: int
    user_id: int
    feedback_txt: str
    is_correct: bool

class FeedbackUpdate(BaseModel):
    feedback_txt: Optional[str] = None
    is_correct: Optional[bool] = None

class FeedbackResponse(BaseModel):
    id: int
    article_id: int
    user_id: int
    feedback_txt: str
    is_correct: bool
    created_at: datetime

    class Config:
        from_attributes = True

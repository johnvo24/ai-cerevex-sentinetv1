from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CommentCreate(BaseModel):
    article_id: int
    user_id: int
    content: str 
    sentiment: Optional[str] = None

class CommentUpdate(BaseModel):
    content: Optional[str] = None 

class CommentResponse(BaseModel):
    id: int
    article_id: int
    user_id: int
    content: str
    sentiment: Optional[str] = None 
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

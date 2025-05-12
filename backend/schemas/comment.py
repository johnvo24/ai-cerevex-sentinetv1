from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class CommentCreate(BaseModel):
    article_id: int
    user_id: int
    content: str 

class CommentUpdate(BaseModel):
    content: Optional[str] = None 

class CommentResponse(BaseModel):
    id: int
    article_id: int
    user_id: int
    content: str 
    created_at: datetime

    class Config:
        from_attributes = True

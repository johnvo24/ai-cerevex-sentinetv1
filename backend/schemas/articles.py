from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ArticleCreate(BaseModel):
    user_id: int
    model_id: Optional[int] = None
    title: Optional[str]
    content: Optional[str]
    sentiment_label: Optional[str]
    class Config:
        from_attributes = True
        protected_namespaces = ()

class ArticleUpdate(BaseModel):
    model_id: Optional[int] = None
    title: Optional[str] = None
    content: Optional[str] = None
    sentiment_label: Optional[str] = None
    class Config:
        from_attributes = True
        protected_namespaces = ()

class ArticleResponse(BaseModel):
    id: int
    user_id: int
    model_id: Optional[int]
    title: Optional[str]
    content: Optional[str]
    sentiment_label: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
        protected_namespaces = ()

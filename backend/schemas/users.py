from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    username: Optional[str]
    password: Optional[str]
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[bool] = False

class UserUpdate(BaseModel):
    username: Optional[str]
    password: Optional[str] = None
    name: Optional[str]
    email: Optional[str]
    role: Optional[bool]

class UserResponse(BaseModel):
    id: int
    username: str
    name: Optional[str]
    email: Optional[str]
    role: Optional[bool]
    created_at: datetime

    class Config:
        from_attributes = True

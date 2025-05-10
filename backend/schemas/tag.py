from pydantic import BaseModel
from typing import Optional

class TagCreate(BaseModel):
    name: str

class TagUpdate(BaseModel):
    name: Optional[str] = None

class TagResponse(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

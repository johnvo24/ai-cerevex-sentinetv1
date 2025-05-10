from pydantic import BaseModel

class ArticleTagCreate(BaseModel):
    article_id: int
    tag_id: int

class ArticleTagResponse(BaseModel):
    article_id: int
    tag_id: int

    class Config:
        from_attributes = True
        protected_namespaces = ()

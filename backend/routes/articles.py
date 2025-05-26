from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from models.database import get_db_conn
from schemas.articles import ArticleCreate, ArticleUpdate, ArticleResponse
from models import articles as article_model

router = APIRouter(prefix="/articles", tags=["Articles"])

@router.post("/create", response_model=ArticleResponse)
async def create_article(article: ArticleCreate, conn: AsyncConnection = Depends(get_db_conn)):
    result = await article_model.create_article(conn, article)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create article")
    return result

@router.get("/get/{article_id}", response_model=ArticleResponse)
async def get_article(article_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await article_model.get_article(conn, article_id)
    if not result:
        raise HTTPException(status_code=404, detail="Article not found")
    return result

@router.get("/all", response_model=list[ArticleResponse])
async def get_all_articles(conn: AsyncConnection = Depends(get_db_conn)):
    return await article_model.get_all_articles(conn)

@router.put("/update/{article_id}", response_model=ArticleResponse)
async def update_article(article_id: int, article: ArticleUpdate, conn: AsyncConnection = Depends(get_db_conn)):
    result = await article_model.update_article(conn, article_id, article)
    if not result:
        raise HTTPException(status_code=404, detail="Article not found")
    return result

@router.delete("/delete/{article_id}")
async def delete_article(article_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await article_model.delete_article(conn, article_id)
    if not result:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"detail": "Article deleted successfully"}

from psycopg import AsyncConnection
from schemas.articles import ArticleCreate, ArticleResponse, ArticleUpdate
from models.database import get_db_conn

async def create_article(conn: AsyncConnection, article: ArticleCreate) -> ArticleResponse:
    query = """
        INSERT INTO Article (user_id, model_id, title, content, sentiment_label)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, user_id, model_id, title, content, sentiment_label, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (
            article.user_id, 
            article.model_id, 
            article.title, 
            article.content, 
            article.sentiment_label
        ))
        row = await cur.fetchone()
        if row:
            return ArticleResponse(
                id=row[0],
                user_id=row[1],
                model_id=row[2],
                title=row[3],
                content=row[4],
                sentiment_label=row[5],
                created_at=row[6]
            )
        return None

async def get_article(conn: AsyncConnection, article_id: int) -> ArticleResponse:
    query = """
        SELECT id, user_id, model_id, title, content, sentiment_label, created_at
        FROM Article
        WHERE id = %s
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (article_id,))
        row = await cur.fetchone()
        if row:
            return ArticleResponse(
                id=row[0],
                user_id=row[1],
                model_id=row[2],
                title=row[3],
                content=row[4],
                sentiment_label=row[5],
                created_at=row[6]
            )
        return None

async def get_all_articles(conn: AsyncConnection) -> list[ArticleResponse]:
    query = """
        SELECT id, user_id, model_id, title, content, sentiment_label, created_at
        FROM Article
    """
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        articles = [ArticleResponse(
            id=row[0],
            user_id=row[1],
            model_id=row[2],
            title=row[3],
            content=row[4],
            sentiment_label=row[5],
            created_at=row[6]
        ) for row in rows]
        return articles

async def update_article(conn: AsyncConnection, article_id: int, article: ArticleUpdate) -> ArticleResponse:
    query = """
        UPDATE Article
        SET model_id = %s, title = %s, content = %s, sentiment_label = %s
        WHERE id = %s
        RETURNING id, user_id, model_id, title, content, sentiment_label, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (
            article.model_id, 
            article.title, 
            article.content, 
            article.sentiment_label, 
            article_id
        ))
        row = await cur.fetchone()
        if row:
            return ArticleResponse(
                id=row[0],
                user_id=row[1],
                model_id=row[2],
                title=row[3],
                content=row[4],
                sentiment_label=row[5],
                created_at=row[6]
            )
        return None

async def delete_article(conn: AsyncConnection, article_id: int) -> bool:
    query = """
        DELETE FROM Article
        WHERE id = %s
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (article_id,))
        if cur.rowcount == 0:
            return False
        return True

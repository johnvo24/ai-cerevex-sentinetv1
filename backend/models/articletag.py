from psycopg import AsyncConnection
from schemas.articletag import ArticleTagCreate, ArticleTagResponse

async def create_article_tag(conn: AsyncConnection, link: ArticleTagCreate):
    query = """
        INSERT INTO ArticleTag (article_id, tag_id)
        VALUES (%s, %s)
        RETURNING article_id, tag_id
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (link.article_id, link.tag_id))
        row = await cur.fetchone()
        if row:
            return ArticleTagResponse(article_id=row[0], tag_id=row[1])

async def get_tags_by_article(conn: AsyncConnection, article_id: int):
    query = """
        SELECT t.id, t.name
        FROM Tag t
        JOIN ArticleTag at ON t.id = at.tag_id
        WHERE at.article_id = %s
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (article_id,))
        rows = await cur.fetchall()
        return [{"id": row[0], "name": row[1]} for row in rows]

async def get_articles_by_tag(conn: AsyncConnection, tag_id: int):
    query = """
        SELECT a.id, a.title
        FROM Article a
        JOIN ArticleTag at ON a.id = at.article_id
        WHERE at.tag_id = %s
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (tag_id,))
        rows = await cur.fetchall()
        return [{"id": row[0], "title": row[1]} for row in rows]

async def delete_article_tag(conn: AsyncConnection, article_id: int, tag_id: int):
    query = """
        DELETE FROM ArticleTag
        WHERE article_id = %s AND tag_id = %s
        RETURNING article_id, tag_id
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (article_id, tag_id))
        row = await cur.fetchone()
        return bool(row)

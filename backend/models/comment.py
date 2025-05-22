from psycopg import AsyncConnection
from schemas.comment import CommentCreate, CommentUpdate, CommentResponse

async def create_comment(conn: AsyncConnection, comment: CommentCreate):
    query = """
        INSERT INTO Comment (article_id, user_id, content, sentiment)
        VALUES (%s, %s, %s, %s)
        RETURNING id, article_id, user_id, content, sentiment, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (comment.article_id, comment.user_id, comment.content, comment.sentiment))
        row = await cur.fetchone()
        if row:
            return CommentResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                content=row[3],
                sentiment=row[4],
                created_at=row[5]
            )

async def get_all_comments(conn: AsyncConnection):
    query = "SELECT * FROM Comment"
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        return [CommentResponse(
            id=row[0],
            article_id=row[1],
            user_id=row[2],
            content=row[3],
            sentiment=row[4],
            created_at=row[5]
        ) for row in rows]

async def get_comment(conn: AsyncConnection, comment_id: int):
    query = "SELECT * FROM Comment WHERE id = %s"
    async with conn.cursor() as cur:
        await cur.execute(query, (comment_id,))
        row = await cur.fetchone()
        if row:
            return CommentResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                content=row[3],
                sentiment=row[4],
                created_at=row[5]
            )

async def update_comment(conn: AsyncConnection, comment_id: int, comment: CommentUpdate):
    query = """
        UPDATE Comment
        SET content = %s
        WHERE id = %s
        RETURNING id, article_id, user_id, content, sentiment, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (comment.content, comment_id))
        row = await cur.fetchone()
        if row:
            return CommentResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                content=row[3],
                sentiment=row[4],
                created_at=row[5]
            )
        return None

async def delete_comment(conn: AsyncConnection, comment_id: int):
    query = "DELETE FROM Comment WHERE id = %s RETURNING id"
    async with conn.cursor() as cur:
        await cur.execute(query, (comment_id,))
        row = await cur.fetchone()
        return bool(row)

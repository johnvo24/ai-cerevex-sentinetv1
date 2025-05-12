from psycopg import AsyncConnection
from schemas.comment import CommentCreate, CommentUpdate, CommentResponse

async def create_comment(conn: AsyncConnection, comment: CommentCreate):
    query = """
        INSERT INTO Comment (article_id, user_id, content)
        VALUES (%s, %s, %s)
        RETURNING id, article_id, user_id, content, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (comment.article_id, comment.user_id, comment.content))
        row = await cur.fetchone()
        if row:
            return CommentResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                content=row[3],
                created_at=row[4]
            )

async def get_all_comments(conn: AsyncConnection):
    query = "SELECT id, article_id, user_id, content, created_at FROM Comment"
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        return [CommentResponse(
            id=row[0],
            article_id=row[1],
            user_id=row[2],
            content=row[3],
            created_at=row[4]
        ) for row in rows]

async def get_comment(conn: AsyncConnection, comment_id: int):
    query = "SELECT id, article_id, user_id, content, created_at FROM Comment WHERE id = %s"
    async with conn.cursor() as cur:
        await cur.execute(query, (comment_id,))
        row = await cur.fetchone()
        if row:
            return CommentResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                content=row[3],
                created_at=row[4]
            )

async def update_comment(conn: AsyncConnection, comment_id: int, comment: CommentUpdate):
    query = """
        UPDATE Comment
        SET content = %s
        WHERE id = %s
        RETURNING id, article_id, user_id, content, created_at
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
                created_at=row[4]
            )
        return None

async def delete_comment(conn: AsyncConnection, comment_id: int):
    query = "DELETE FROM Comment WHERE id = %s RETURNING id"
    async with conn.cursor() as cur:
        await cur.execute(query, (comment_id,))
        row = await cur.fetchone()
        return bool(row)

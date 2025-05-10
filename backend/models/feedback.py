from psycopg import AsyncConnection
from schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackResponse

async def create_feedback(conn: AsyncConnection, feedback: FeedbackCreate):
    query = """
        INSERT INTO Feedback (article_id, user_id, feedback_txt, is_correct)
        VALUES (%s, %s, %s, %s)
        RETURNING id, article_id, user_id, feedback_txt, is_correct, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (feedback.article_id, feedback.user_id, feedback.feedback_txt, feedback.is_correct))
        row = await cur.fetchone()
        if row:
            return FeedbackResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                feedback_txt=row[3],
                is_correct=row[4],
                created_at=row[5]
            )

async def get_all_feedback(conn: AsyncConnection):
    query = "SELECT * FROM Feedback"
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        return [FeedbackResponse(
            id=row[0],
            article_id=row[1],
            user_id=row[2],
            feedback_txt=row[3],
            is_correct=row[4],
            created_at=row[5]
        ) for row in rows]

async def get_feedback(conn: AsyncConnection, feedback_id: int):
    query = "SELECT * FROM Feedback WHERE id = %s"
    async with conn.cursor() as cur:
        await cur.execute(query, (feedback_id,))
        row = await cur.fetchone()
        if row:
            return FeedbackResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                feedback_txt=row[3],
                is_correct=row[4],
                created_at=row[5]
            )

async def update_feedback(conn: AsyncConnection, feedback_id: int, feedback: FeedbackUpdate):
    query = """
        UPDATE Feedback
        SET feedback_txt = %s, is_correct = %s
        WHERE id = %s
        RETURNING id, article_id, user_id, feedback_txt, is_correct, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (feedback.feedback_txt, feedback.is_correct, feedback_id))
        row = await cur.fetchone()
        if row:
            return FeedbackResponse(
                id=row[0],
                article_id=row[1],
                user_id=row[2],
                feedback_txt=row[3],
                is_correct=row[4],
                created_at=row[5]
            )
        return None

async def delete_feedback(conn: AsyncConnection, feedback_id: int):
    query = "DELETE FROM Feedback WHERE id = %s RETURNING id"
    async with conn.cursor() as cur:
        await cur.execute(query, (feedback_id,))
        row = await cur.fetchone()
        if row:
            return True
        return False

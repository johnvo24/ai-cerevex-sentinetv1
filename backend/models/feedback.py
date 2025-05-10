from psycopg import AsyncConnection
from schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackResponse

async def create_feedback(conn: AsyncConnection, feedback: FeedbackCreate) -> FeedbackResponse:
    query = """
        INSERT INTO Feedback (article_id, user_id, feedback_txt, is_correct, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        RETURNING id, article_id, user_id, feedback_txt, is_correct, created_at
    """
    async with conn.cursor() as cur:
        await cur.execute(query, (feedback.article_id, feedback.user_id, feedback.feedback_txt, feedback.is_correct))
        row = await cur.fetchone()
        return FeedbackResponse(**dict(zip([desc[0] for desc in cur.description], row)))

async def get_feedback(conn: AsyncConnection, feedback_id: int) -> FeedbackResponse | None:
    query = "SELECT * FROM Feedback WHERE id = %s"
    async with conn.cursor() as cur:
        await cur.execute(query, (feedback_id,))
        row = await cur.fetchone()
        if row:
            return FeedbackResponse(**dict(zip([desc[0] for desc in cur.description], row)))
        return None

async def get_all_feedback(conn: AsyncConnection) -> list[FeedbackResponse]:
    query = "SELECT * FROM Feedback ORDER BY created_at DESC"
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        return [
            FeedbackResponse(**dict(zip([desc[0] for desc in cur.description], row)))
            for row in rows
        ]

async def update_feedback(conn: AsyncConnection, feedback_id: int, update: FeedbackUpdate) -> bool:
    fields = []
    values = []

    if update.feedback_txt is not None:
        fields.append("feedback_txt = %s")
        values.append(update.feedback_txt)
    if update.is_correct is not None:
        fields.append("is_correct = %s")
        values.append(update.is_correct)

    if not fields:
        return False

    values.append(feedback_id)

    query = f"""
        UPDATE Feedback
        SET {', '.join(fields)}
        WHERE id = %s
        RETURNING id
    """
    async with conn.cursor() as cur:
        await cur.execute(query, values)
        row = await cur.fetchone()
        if row:
            await conn.commit()
            return True
        return False

async def delete_feedback(conn: AsyncConnection, feedback_id: int) -> bool:
    query = "DELETE FROM Feedback WHERE id = %s RETURNING id"
    async with conn.cursor() as cur:
        await cur.execute(query, (feedback_id,))
        row = await cur.fetchone()
        if row:
            await conn.commit()
            return True
        return False

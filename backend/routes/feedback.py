from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from models.database import get_db_conn
from schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackResponse
from models import feedback as feedback_model

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/create", response_model=FeedbackResponse)
async def create_feedback(data: FeedbackCreate, conn: AsyncConnection = Depends(get_db_conn)):
    return await feedback_model.create_feedback(conn, data)

@router.get("/{feedback_id}", response_model=FeedbackResponse)
async def get_feedback(feedback_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await feedback_model.get_feedback(conn, feedback_id)
    if not result:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return result

@router.get("/", response_model=list[FeedbackResponse])
async def get_all_feedback(conn: AsyncConnection = Depends(get_db_conn)):
    return await feedback_model.get_all_feedback(conn)

@router.put("/{feedback_id}")
async def update_feedback(feedback_id: int, update: FeedbackUpdate, conn: AsyncConnection = Depends(get_db_conn)):
    success = await feedback_model.update_feedback(conn, feedback_id, update)
    if not success:
        raise HTTPException(status_code=404, detail="Feedback not found or no fields updated")
    return {"message": "Feedback updated successfully"}

@router.delete("/{feedback_id}")
async def delete_feedback(feedback_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    success = await feedback_model.delete_feedback(conn, feedback_id)
    if not success:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return {"message": "Feedback deleted successfully"}

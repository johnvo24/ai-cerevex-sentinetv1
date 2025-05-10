from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from models.database import get_db_conn
from schemas.feedback import FeedbackCreate, FeedbackUpdate, FeedbackResponse
from models import feedback as feedback_model

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/create", response_model=FeedbackResponse)
async def create_feedback(feedback: FeedbackCreate, conn: AsyncConnection = Depends(get_db_conn)):
    return await feedback_model.create_feedback(conn, feedback)

@router.get("/get/{feedback_id}", response_model=FeedbackResponse)
async def get_feedback(feedback_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await feedback_model.get_feedback(conn, feedback_id)
    if not result:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return result

@router.get("/all", response_model=list[FeedbackResponse])
async def get_all_feedback(conn: AsyncConnection = Depends(get_db_conn)):
    return await feedback_model.get_all_feedback(conn)

@router.put("/update/{feedback_id}", response_model=FeedbackResponse)
async def update_feedback(feedback_id: int, feedback: FeedbackUpdate, conn: AsyncConnection = Depends(get_db_conn)):
    result = await feedback_model.update_feedback(conn, feedback_id, feedback)
    if not result:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return result

@router.delete("/delete/{feedback_id}")
async def delete_feedback(feedback_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await feedback_model.delete_feedback(conn, feedback_id)
    if not result:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return {"message": "Feedback deleted successfully"}

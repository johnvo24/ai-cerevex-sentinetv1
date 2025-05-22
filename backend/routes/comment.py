from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from models.database import get_db_conn
from schemas.comment import CommentCreate, CommentUpdate, CommentResponse
from models import comment as comment_model 

router = APIRouter(prefix="/comments", tags=["Comments"])

@router.post("/create", response_model=CommentResponse)
async def create_comment(comment: CommentCreate, conn: AsyncConnection = Depends(get_db_conn)):
    return await comment_model.create_comment(conn, comment)

@router.get("/get/{comment_id}", response_model=CommentResponse)
async def get_comment(comment_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await comment_model.get_comment(conn, comment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Comment not found")
    return result

@router.get("/all", response_model=list[CommentResponse])
async def get_all_comments(conn: AsyncConnection = Depends(get_db_conn)):
    return await comment_model.get_all_comments(conn)

@router.put("/update/{comment_id}", response_model=CommentResponse)
async def update_comment(comment_id: int, comment: CommentUpdate, conn: AsyncConnection = Depends(get_db_conn)):
    result = await comment_model.update_comment(conn, comment_id, comment)
    if not result:
        raise HTTPException(status_code=404, detail="Comment not found")
    return result

@router.delete("/delete/{comment_id}")
async def delete_comment(comment_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await comment_model.delete_comment(conn, comment_id)
    if not result:
        raise HTTPException(status_code=404, detail="Comment not found")
    return {"message": "Comment deleted successfully"}

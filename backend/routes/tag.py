from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from models.database import get_db_conn
from schemas.tag import TagCreate, TagUpdate, TagResponse
from models import tag as tag_model

router = APIRouter(prefix="/tags", tags=["Tags"])

@router.post("/create", response_model=TagResponse)
async def create_tag(tag: TagCreate, conn: AsyncConnection = Depends(get_db_conn)):
    result = await tag_model.create_tag(conn, tag)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to create tag")
    return result

@router.get("/get/{tag_id}", response_model=TagResponse)
async def get_tag(tag_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await tag_model.get_tag(conn, tag_id)
    if not result:
        raise HTTPException(status_code=404, detail="Tag not found")
    return result

@router.get("/all", response_model=list[TagResponse])
async def get_all_tags(conn: AsyncConnection = Depends(get_db_conn)):
    return await tag_model.get_all_tags(conn)

@router.put("/update/{tag_id}", response_model=TagResponse)
async def update_tag(tag_id: int, tag: TagUpdate, conn: AsyncConnection = Depends(get_db_conn)):
    result = await tag_model.update_tag(conn, tag_id, tag)
    if not result:
        raise HTTPException(status_code=404, detail="Tag not found")
    return result

@router.delete("/delete/{tag_id}")
async def delete_tag(tag_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await tag_model.delete_tag(conn, tag_id)
    if not result:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"detail": "Tag deleted successfully"}

from fastapi import APIRouter, Depends, HTTPException
from psycopg import AsyncConnection
from models.database import get_db_conn
from schemas.users import UserCreate, UserUpdate, UserResponse, UserLogin
from models import users as user_model

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/create", response_model=UserResponse)
async def create_user(user: UserCreate, conn: AsyncConnection = Depends(get_db_conn)):
    return await user_model.create_user(conn, user)

@router.post("/login", response_model=UserResponse)
async def login(data: UserLogin, conn: AsyncConnection = Depends(get_db_conn)):
    result = await user_model.login(conn, data)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return result

@router.get("/get/{user_id}", response_model=UserResponse)
async def get_user_by_id(user_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await user_model.get_user_by_id(conn, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result

@router.get("/all", response_model=list[UserResponse])
async def get_all_users(conn: AsyncConnection = Depends(get_db_conn)):
    return await user_model.get_all_users(conn)

@router.put("/update/{user_id}")
async def update_user(user_id: int, user: UserUpdate, conn: AsyncConnection = Depends(get_db_conn)):
    result = await user_model.update_user(conn, user_id, user)
    if not result:
        raise HTTPException(status_code=404, detail="User not found or no changes")
    return result

@router.delete("/delete/{user_id}")
async def delete_user(user_id: int, conn: AsyncConnection = Depends(get_db_conn)):
    result = await user_model.delete_user(conn, user_id)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    return result

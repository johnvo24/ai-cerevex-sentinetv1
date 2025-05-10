from psycopg import AsyncConnection
from schemas.users import UserCreate, UserUpdate, UserResponse
from models.helper import hash_pw

async def create_user(conn: AsyncConnection, user: UserCreate):
    query = """
        INSERT INTO Users (username, password, name, email, role)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id, username, name, email, role, created_at
    """
    async with conn.cursor() as cur:
        hashed_pw = hash_pw(user.password)
        await cur.execute(query, (user.username, hashed_pw, user.name, user.email, user.role))
        row = await cur.fetchone()
        return UserResponse(
            id=row[0], username=row[1], name=row[2], email=row[3], role=row[4], created_at=row[5]
        )

async def get_user(conn: AsyncConnection, user_id: int):
    query = "SELECT id, username, name, email, role, created_at FROM Users WHERE id = %s"
    async with conn.cursor() as cur:
        await cur.execute(query, (user_id,))
        row = await cur.fetchone()
        if row:
            return UserResponse(
                id=row[0], username=row[1], name=row[2],
                email=row[3], role=row[4], created_at=row[5]
            )

async def get_all_users(conn: AsyncConnection):
    query = "SELECT id, username, name, email, role, created_at FROM Users"
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        return [UserResponse(
            id=row[0], username=row[1], name=row[2],
            email=row[3], role=row[4], created_at=row[5]
        ) for row in rows]

async def update_user(conn: AsyncConnection, user_id: int, user: UserUpdate):
    fields = []
    values = []

    if user.name:
        fields.append("name = %s")
        values.append(user.name)
    if user.email:
        fields.append("email = %s")
        values.append(user.email)
    if user.password:
        fields.append("password = %s")
        values.append(hash_pw(user.password))
    if user.role is not None:
        fields.append("role = %s")
        values.append(user.role)

    if not fields:
        return None

    query = f"""
        UPDATE Users SET {', '.join(fields)}
        WHERE id = %s RETURNING id
    """
    values.append(user_id)

    async with conn.cursor() as cur:
        await cur.execute(query, tuple(values))
        row = await cur.fetchone()
        if row:
            await conn.commit()
            return {"message": "User updated"}

async def delete_user(conn: AsyncConnection, user_id: int):
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM Users WHERE id = %s RETURNING id", (user_id,))
        row = await cur.fetchone()
        if row:
            await conn.commit()
            return {"message": "User deleted"}

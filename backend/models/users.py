from psycopg import AsyncConnection
from schemas.users import UserCreate, UserLogin, UserUpdate, UserResponse
from models.helper import hash_pw, verify_password

async def create_user(conn: AsyncConnection, user: UserCreate):
    query = """
        INSERT INTO users (username, password, name, email)
        VALUES (%s, %s, %s, %s)
        RETURNING id, username, name, email, created_at
    """
    async with conn.cursor() as cur:
        hashed_pw = hash_pw(user.password)
        await cur.execute(query, (user.username, hashed_pw, user.name, user.email))
        row = await cur.fetchone()
        return UserResponse(
            id=row[0], username=row[1], name=row[2], email=row[3], created_at=row[4]
        )

async def get_user_by_id(conn: AsyncConnection, user_id: int):
    query = "SELECT id, username, name, email, created_at FROM Users WHERE id = %s"
    async with conn.cursor() as cur:
        await cur.execute(query, (user_id,))
        row = await cur.fetchone()
        if row:
            return UserResponse(
                id=row[0], username=row[1], name=row[2],
                email=row[3], created_at=row[4]
            )

async def login(conn: AsyncConnection, data: UserLogin):
    query = "SELECT id, username, name, email, password, created_at FROM Users WHERE username = %s"
    async with conn.cursor() as cur:
        await cur.execute(query, (data.username,))
        row = await cur.fetchone()
        if row:
            stored_hashed_pw = row[4]
            if verify_password(data.password, stored_hashed_pw):
                return UserResponse(
                    id=row[0], username=row[1], name=row[2],
                    email=row[3], created_at=row[5]
                )
    return None


async def get_all_users(conn: AsyncConnection):
    query = "SELECT id, username, name, email, created_at FROM Users"
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        return [UserResponse(
            id=row[0], username=row[1], name=row[2],
            email=row[3], created_at=row[4]
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

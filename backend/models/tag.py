from psycopg import AsyncConnection
from schemas.tag import TagCreate, TagResponse, TagUpdate
from models.database import get_db_conn

async def create_tag(conn: AsyncConnection, tag: TagCreate) -> TagResponse:
    query = """INSERT INTO Tag (name) VALUES (%s) RETURNING id, name"""
    async with conn.cursor() as cur:
        await cur.execute(query, (tag.name,))
        row = await cur.fetchone()
        if row:
            return TagResponse(id=row[0], name=row[1])
        return None

async def get_tag(conn: AsyncConnection, tag_id: int) -> TagResponse:
    query = """SELECT id, name FROM Tag WHERE id = %s"""
    async with conn.cursor() as cur:
        await cur.execute(query, (tag_id,))
        row = await cur.fetchone()
        if row:
            return TagResponse(id=row[0], name=row[1])
        return None

async def get_all_tags(conn: AsyncConnection) -> list[TagResponse]:
    query = """SELECT id, name FROM Tag"""
    async with conn.cursor() as cur:
        await cur.execute(query)
        rows = await cur.fetchall()
        tags = [TagResponse(id=row[0], name=row[1]) for row in rows]
        return tags

async def update_tag(conn: AsyncConnection, tag_id: int, tag: TagUpdate) -> TagResponse:
    query = """UPDATE Tag SET name = %s WHERE id = %s RETURNING id, name"""
    async with conn.cursor() as cur:
        await cur.execute(query, (tag.name, tag_id))
        row = await cur.fetchone()
        if row:
            return TagResponse(id=row[0], name=row[1])
        return None
async def delete_tag(conn: AsyncConnection, tag_id: int) -> bool:
    query = """DELETE FROM Tag WHERE id = %s"""
    async with conn.cursor() as cur:
        await cur.execute(query, (tag_id,))
        if cur.rowcount > 0:
            return True
        return False
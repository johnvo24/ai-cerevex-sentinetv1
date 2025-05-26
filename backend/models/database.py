from psycopg_pool import AsyncConnectionPool
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

pool = AsyncConnectionPool(DATABASE_URL, min_size=1, max_size=10)

async def connect_to_db():
    await pool.open()

async def close_db_connection():
    await pool.close()

async def get_db_conn():
    async with pool.connection() as conn:
        yield conn

import psycopg2
from psycopg2 import sql
from config import DATABASE

def db_connect():
    try:
        conn = psycopg2.connect(
            host=DATABASE['host'],
            database=DATABASE['database'],
            user=DATABASE['user'],
            password=DATABASE['password']
        )
        return conn
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        return None

def create_songs_table():
    conn = db_connect()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS songs (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255),
                    artist VARCHAR(255),
                    album VARCHAR(255),
                    fingerprint BYTEA
                );
            ''')
            conn.commit()
        except Exception as error:
            print(f"Error creating table: {error}")
        finally:
            cursor.close()
            conn.close()

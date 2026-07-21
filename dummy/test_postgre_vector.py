import psycopg
from psycopg.rows import dict_row

conn_string = "postgresql://postgres:mysecretpassword@localhost:5431/postgres"


def initialize_database():
    print("Trying to connect to PostgreSQL DB")

    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            print("Enabling the pgvector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute("""
            CREATE TABLE IF NOT EXISTS test_embeddings (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding VECTOR(3)
            );""")

            cur.execute(
                "INSERT INTO test_embeddings (content, embedding) VALUES (%s, %s)",
                ("Hello Enterprise RAG", "[0.1, 0.2, 0.3]"),
            )

            print("✅ Database successfully configured and data inserted!")


def show_data():
    conn = psycopg.connect(conn_string, row_factory=dict_row)
    cursor = conn.cursor()
    query = "SELECT * FROM test_embeddings;"
    result = cursor.execute(query)
    data = result.fetchall()
    print(data)
    cursor.close()
    conn.close()


def vector_search():
    conn = psycopg.connect(conn_string, row_factory=dict_row)
    cursor = conn.cursor()
    query_vector = "[0.5, 0.4, 0.1]"
    query = "SELECT *, embedding <=> %s AS distance FROM test_embeddings ORDER BY distance ASC LIMIT 10;"
    result = cursor.execute(query, (query_vector,))
    data = result.fetchall()
    for row in data:
        print(row)
    cursor.close()
    conn.close()


if __name__ == "__main__":
    vector_search()

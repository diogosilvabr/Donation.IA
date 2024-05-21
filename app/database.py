from neo4j import GraphDatabase
from app.config import Config

class Neo4jConnection:
    def __init__(self):
        self.driver = GraphDatabase.driver(Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None, **kwargs):
        with self.driver.session() as session:
            return session.run(query, parameters, **kwargs)

def buscarDados():
    conn = Neo4jConnection()
    query = """
    MATCH (n:Texto)
    RETURN n.text AS text, n.hate_speech AS Hate_speech, n.sexism AS Sexism
    """
    result = conn.query(query)
    data = []
    for record in result:
        data.append(record.data())
    conn.close()
    return data

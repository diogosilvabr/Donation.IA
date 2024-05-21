import os

class Config:
    NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j+s://66522652.databases.neo4j.io')
    NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'wDqKpcOPAvNvq5cbsp6bt5rL70spbB0lwjvmI70i6P4')
    CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', 'data/base.csv')

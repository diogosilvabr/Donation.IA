from flask import Flask
from flask_cors import CORS
from app import create_app

app = create_app()

# Adicionar suporte a CORS
CORS(app)

if __name__ == "__main__":
    app.run(debug=True, port=8888)

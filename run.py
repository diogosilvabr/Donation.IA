from flask import Flask
from flask_cors import CORS
from app import create_app

# Cria a aplicação Flask usando a função create_app
app = create_app()

# Adiciona suporte a CORS à aplicação
CORS(app)

# Se o script for executado diretamente, inicia o servidor Flask
if __name__ == "__main__":
    # Executa a aplicação no modo debug na porta 8888
    app.run(debug=True, port=8888)

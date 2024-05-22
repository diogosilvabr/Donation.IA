from app import create_app
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

app = create_app()

# Adicionar suporte a CORS
CORS(app)

if __name__ == "__main__":
    app.run(debug=True, port=8888)

from app import create_app
from flask_swagger_ui import get_swaggerui_blueprint
from flask_cors import CORS

app = create_app()

# Configuração do Swagger
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL, config={'app_name': "Donation.IA"})
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Adicionar suporte a CORS
CORS(app)

if __name__ == "__main__":
    app.run(debug=True, port=8888)

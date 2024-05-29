from flask import Flask

def create_app():
    app = Flask(__name__)
    
    # Registrar os blueprints
    from .routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app

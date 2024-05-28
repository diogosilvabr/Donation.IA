import csv
from flask import Blueprint, request, jsonify
from app.preprocessing import preprocessarTexto
from ml.modelo import carregarModelo

api_bp = Blueprint('api', __name__)

# Carregar o modelo e o vetorizador
modelo, vetorizador = carregarModelo('ml/modelo_inappropriate.pkl', 'ml/vetorizador_inappropriate.pkl')

@api_bp.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    texto = data.get('text', '')
    texto_limpo = preprocessarTexto(texto)
    vetor = vetorizador.transform([texto_limpo])

    # Previsão usando o modelo carregado
    predicao = modelo.predict(vetor)[0]

    # Converter para bool padrão do Python
    is_inappropriate = bool(predicao == 1)

    return jsonify({'inapropriado': is_inappropriate})

@api_bp.route('/add-feedback', methods=['POST'])
def add_feedback():
    data = request.json
    texto = data.get('text', '')
    inappropriate = data.get('inappropriate', 0)

    # Preprocessar o texto
    texto_limpo = preprocessarTexto(texto)

    # Salvar o feedback no arquivo CSV
    with open('data/feedback.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([texto, texto_limpo, inappropriate])

    return jsonify({'message': 'Feedback added successfully'})

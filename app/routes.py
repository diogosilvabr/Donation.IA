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
    if 'text' not in data:
        return jsonify({'error': 'Missing text parameter'}), 400

    texto = data.get('text', '')
    texto_limpo = preprocessarTexto(texto)
    vetor = vetorizador.transform([texto_limpo])

    # Previsão usando o modelo carregado
    predicao = modelo.predict(vetor)[0]

    # Converter para bool padrão do Python
    is_inappropriate = bool(predicao == 1)

    return jsonify({'inapropriado': is_inappropriate})

# Endpoint para adicionar feedback manualmente
@api_bp.route('/add-feedback', methods=['POST'])
def add_feedback():
    data = request.json
    if 'text' not in data or 'inappropriate' not in data:
        return jsonify({'error': 'Missing text or inappropriate parameter'}), 400

    text = data['text']
    inappropriate = data['inappropriate']

    # Adicionar o feedback ao arquivo CSV
    with open('data/feedback_atualizado.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([text, inappropriate])

    return jsonify({'message': 'Feedback added successfully'}), 200

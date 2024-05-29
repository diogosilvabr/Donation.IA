import csv
import pandas as pd
from flask import Blueprint, request, jsonify
from app.preprocessing import preprocessarTexto
from ml.modelo import carregarModelo
from sklearn.metrics import f1_score

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

@api_bp.route('/model-performance', methods=['GET'])
def model_performance():
    # Carregar dados de teste
    df_feedback = pd.read_csv('data/feedback.csv')
    df_feedback['textoLimpo'] = df_feedback['text'].apply(preprocessarTexto)
    X_test = vetorizador.transform(df_feedback['textoLimpo'])
    y_test = df_feedback['inappropriate'].astype(int)
    
    # Prever e calcular o F1-Score
    y_pred = modelo.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    return jsonify({'F1-Score': f1})

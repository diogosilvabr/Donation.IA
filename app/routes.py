import csv 
import pandas as pd 
from flask import Blueprint, request, jsonify
from app.preprocessing import preprocessarTexto
from ml.modelo import carregarModelo 
from sklearn.metrics import f1_score

# Cria um blueprint para organizar as rotas da API
api_bp = Blueprint('api', __name__)

# Carrega o modelo de machine learning e o vetorizador
modelo, vetorizador = carregarModelo('ml/modelo_inappropriate.pkl', 'ml/vetorizador_inappropriate.pkl')

# Cria a rota para o endpoint de análise de texto
@api_bp.route('/analyze-text', methods=['POST'])
def analyze_text():
    # Pega os dados da solicitação POST no formato JSON
    data = request.json
    # Pega o texto enviado
    texto = data.get('text', '')
    # Pré-processa o texto
    texto_limpo = preprocessarTexto(texto)
    # Transforma o texto em vetor
    vetor = vetorizador.transform([texto_limpo])

    # Faz a previsão usando o modelo carregado
    predicao = modelo.predict(vetor)[0]

    # Converte a previsão para um booleano (True ou False) para ser retornado no request pelo endpoint
    is_inappropriate = bool(predicao == 1)

    # Retorna o resultado no formato JSON
    return jsonify({'inapropriado': is_inappropriate})

# Cria a rota para o endpoint de inclusão de feedback
@api_bp.route('/add-feedback', methods=['POST'])
def add_feedback():
    # Pega os dados da solicitação POST no formato JSON
    data = request.json
    # Pega o texto enviado
    texto = data.get('text', '')
    # Pega a marcação de inapropriado (padrão é 0)
    inappropriate = data.get('inappropriate', 0)

    # Pré-processa o texto
    texto_limpo = preprocessarTexto(texto)

    # Abre o arquivo CSV do feedback e adiciona uma nova linha
    with open('data/feedback.csv', 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([texto, texto_limpo, inappropriate])

    # Retorna uma mensagem de sucesso
    return jsonify({'message': 'Feedback adicionado com sucesso!'})

# Cria a rota para um endpoint de verificação do desempenho do modelo
@api_bp.route('/model-performance', methods=['GET'])
def model_performance():
    # Carrega os dados de feedback do CSV
    df_feedback = pd.read_csv('data/feedback.csv')
    # Pré-processa os textos
    df_feedback['textoLimpo'] = df_feedback['text'].apply(preprocessarTexto)
    # Transforma os textos em vetores
    X_test = vetorizador.transform(df_feedback['textoLimpo'])
    # Converte as marcas de inapropriado para inteiros
    y_test = df_feedback['inappropriate'].astype(int)
    
    # Faz previsões com o modelo e calcula o F1-Score
    y_pred = modelo.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    # Retorna o F1-Score no formato JSON
    return jsonify({'F1-Score': f1})

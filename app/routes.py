# routes.py
import csv
from flask import Blueprint, request, jsonify
from app.preprocessing import preprocessarTexto
from ml.modelo import carregarModelo

api_bp = Blueprint('api', __name__)

modelo_homophobia, vetorizador = carregarModelo('ml/modelo_tree_homophobia.pkl', 'ml/vetorizador_tree.pkl')
modelo_obscene, _ = carregarModelo('ml/modelo_tree_obscene.pkl', 'ml/vetorizador_tree.pkl')
modelo_insult, _ = carregarModelo('ml/modelo_tree_insult.pkl', 'ml/vetorizador_tree.pkl')
modelo_racism, _ = carregarModelo('ml/modelo_tree_racism.pkl', 'ml/vetorizador_tree.pkl')
modelo_misogyny, _ = carregarModelo('ml/modelo_tree_misogyny.pkl', 'ml/vetorizador_tree.pkl')
modelo_xenophobia, _ = carregarModelo('ml/modelo_tree_xenophobia.pkl', 'ml/vetorizador_tree.pkl')

@api_bp.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.json
    texto = data.get('text', '')
    texto_limpo = preprocessarTexto(texto)
    vetor = vetorizador.transform([texto_limpo])

    predicao_homophobia = modelo_homophobia.predict(vetor)[0]
    predicao_obscene = modelo_obscene.predict(vetor)[0]
    predicao_insult = modelo_insult.predict(vetor)[0]
    predicao_racism = modelo_racism.predict(vetor)[0]
    predicao_misogyny = modelo_misogyny.predict(vetor)[0]
    predicao_xenophobia = modelo_xenophobia.predict(vetor)[0]

    is_inappropriate = any([
        predicao_homophobia,
        predicao_obscene,
        predicao_insult,
        predicao_racism,
        predicao_misogyny,
        predicao_xenophobia
    ])

    return jsonify({'inapropriado?': is_inappropriate})

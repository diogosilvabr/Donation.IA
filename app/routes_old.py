import csv
from flask import Blueprint, request, jsonify
from app.preprocessing import preprocessarTexto
from ml.modelo import carregarModelo

api_bp = Blueprint('api', __name__)

# Carregar modelos e vetorizador para Árvores de Decisão
modelo_homophobia, vetorizador = carregarModelo('ml/modelo_tree_homophobia.pkl', 'ml/vetorizador_tree.pkl')
modelo_obscene, _ = carregarModelo('ml/modelo_tree_obscene.pkl', 'ml/vetorizador_tree.pkl')
modelo_insult, _ = carregarModelo('ml/modelo_tree_insult.pkl', 'ml/vetorizador_tree.pkl')
modelo_racism, _ = carregarModelo('ml/modelo_tree_racism.pkl', 'ml/vetorizador_tree.pkl')
modelo_misogyny, _ = carregarModelo('ml/modelo_tree_misogyny.pkl', 'ml/vetorizador_tree.pkl')
modelo_xenophobia, _ = carregarModelo('ml/modelo_tree_xenophobia.pkl', 'ml/vetorizador_tree.pkl')

# Variável global para armazenar as predições do último teste
predicoes_atuais = {}

def salvar_feedback(texto_original, texto_limpo, predicoes, correcoes):
    with open('data/feedback.csv', 'a', newline='', encoding='latin1') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            texto_original,
            texto_limpo,
            predicoes.get('homophobia', 0),
            predicoes.get('obscene', 0),
            predicoes.get('insult', 0),
            predicoes.get('racism', 0),
            predicoes.get('misogyny', 0),
            predicoes.get('xenophobia', 0),
            correcoes.get('homophobiaCorrection', 0),
            correcoes.get('obsceneCorrection', 0),
            correcoes.get('insultCorrection', 0),
            correcoes.get('racismCorrection', 0),
            correcoes.get('misogynyCorrection', 0),
            correcoes.get('xenophobiaCorrection', 0)
        ])

@api_bp.route('/test', methods=['POST'])
def test_model():
    data = request.json
    texto = data['text']
    texto_limpo = preprocessarTexto(texto)
    vetor = vetorizador.transform([texto_limpo])
    
    predicao_homophobia = modelo_homophobia.predict(vetor)[0]
    predicao_obscene = modelo_obscene.predict(vetor)[0]
    predicao_insult = modelo_insult.predict(vetor)[0]
    predicao_racism = modelo_racism.predict(vetor)[0]
    predicao_misogyny = modelo_misogyny.predict(vetor)[0]
    predicao_xenophobia = modelo_xenophobia.predict(vetor)[0]

    predicoes = {
        'homophobia': int(predicao_homophobia),
        'obscene': int(predicao_obscene),
        'insult': int(predicao_insult),
        'racism': int(predicao_racism),
        'misogyny': int(predicao_misogyny),
        'xenophobia': int(predicao_xenophobia)
    }

    # Armazenar as predições atuais para uso no feedback
    global predicoes_atuais
    predicoes_atuais = predicoes

    return jsonify({
        'textoOriginal': texto,
        'textoLimpo': texto_limpo,
        'predicoes': predicoes
    })

@api_bp.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    texto = data['text']
    predicoes = data['predicoes']
    correcoes = {
        'homophobiaCorrection': data['correcoes'].get('homophobiaCorrection', predicoes['homophobia']),
        'obsceneCorrection': data['correcoes'].get('obsceneCorrection', predicoes['obscene']),
        'insultCorrection': data['correcoes'].get('insultCorrection', predicoes['insult']),
        'racismCorrection': data['correcoes'].get('racismCorrection', predicoes['racism']),
        'misogynyCorrection': data['correcoes'].get('misogynyCorrection', predicoes['misogyny']),
        'xenophobiaCorrection': data['correcoes'].get('xenophobiaCorrection', predicoes['xenophobia'])
    }

    # Salvar feedback no arquivo feedback.csv
    salvar_feedback(texto, preprocessarTexto(texto), predicoes, correcoes)

    return jsonify({"message": "Sucesso! Feedback salvo!"}), 200

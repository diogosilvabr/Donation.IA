import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib

# Adicionar o diretório raiz ao caminho do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.preprocessing import preprocessarTexto

# Definir uma semente para reprodutibilidade
SEED = 42

def carregar_e_preprocessar_dados(caminho_dados, feedback_dados=None):
    df = pd.read_csv(caminho_dados)
    if feedback_dados:
        df_feedback = pd.read_csv(feedback_dados, names=['text', 'textoLimpo', 'homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia', 'homophobiaCorrection', 'obsceneCorrection', 'insultCorrection', 'racismCorrection', 'misogynyCorrection', 'xenophobiaCorrection'], encoding='latin1', on_bad_lines='skip')
        
        # Remover a linha do cabeçalho se estiver presente nos dados
        df_feedback = df_feedback[df_feedback['homophobiaCorrection'] != 'homophobiaCorrection']
        
        # Garantir que as colunas de correção sejam numéricas e preencher NaNs
        for col in ['homophobiaCorrection', 'obsceneCorrection', 'insultCorrection', 'racismCorrection', 'misogynyCorrection', 'xenophobiaCorrection']:
            df_feedback[col] = pd.to_numeric(df_feedback[col], errors='coerce').fillna(0).astype(int)
        
        df = pd.concat([df, df_feedback], ignore_index=True)
    df['textoLimpo'] = df['text'].apply(preprocessarTexto)
    df = df.dropna(subset=['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia'])  # Remover linhas com valores nulos nas colunas de destino
    return df

def treinarModelo(dados):
    vetorizador = TfidfVectorizer(max_features=5000)
    X = vetorizador.fit_transform(dados['textoLimpo'])
    modelos = {}
    metricas_iniciais = {}

    for coluna in ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']:
        y = dados[coluna]
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=SEED)
        
        modelo = DecisionTreeClassifier(random_state=SEED)
        modelo.fit(X_treino, y_treino)
        
        y_teste_pred = modelo.predict(X_teste)
        metricas_iniciais[coluna] = classification_report(y_teste, y_teste_pred, output_dict=True, zero_division=1)
        
        modelos[coluna] = modelo

    return modelos, vetorizador, metricas_iniciais

def avaliarModelo(modelos, vetorizador, dados):
    X = vetorizador.transform(dados['textoLimpo'])
    metricas_finais = {}

    for coluna in ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']:
        y = dados[coluna + 'Correction'].fillna(0).astype(int)  # Garantir que os valores de y sejam inteiros e preencher NaNs com 0
        y_pred = modelos[coluna].predict(X)
        metricas_finais[coluna] = classification_report(y, y_pred, output_dict=True, zero_division=1)
        
    return metricas_finais

def salvarModelo(modelos, vetorizador, caminhoModelo, caminhoVetorizador):
    joblib.dump(vetorizador, caminhoVetorizador)
    for nome_modelo, modelo in modelos.items():
        joblib.dump(modelo, f'{caminhoModelo}_{nome_modelo}.pkl')

def imprimir_metricas(metricas_iniciais, metricas_finais):
    for coluna in metricas_iniciais.keys():
        print(f"\nMétricas para {coluna}:\n")
        print("Antes do re-treinamento:")
        imprimir_classification_report(metricas_iniciais[coluna])
        print("\nDepois do re-treinamento:")
        imprimir_classification_report(metricas_finais[coluna])

def imprimir_classification_report(report_dict):
    for label, metrics in report_dict.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"Classe: {label}")
            print(f"  Precisão: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-score: {metrics['f1-score']:.2f}")
            print(f"  Suporte: {metrics['support']}")
    if 'accuracy' in report_dict:
        print(f"  Acurácia: {report_dict['accuracy']:.2f}")
    for avg in ['macro avg', 'weighted avg']:
        if avg in report_dict:
            print(f"  {avg.capitalize()}:")
            print(f"    Precisão: {report_dict[avg]['precision']:.2f}")
            print(f"    Recall: {report_dict[avg]['recall']:.2f}")
            print(f"    F1-score: {report_dict[avg]['f1-score']:.2f}")
            print(f"    Suporte: {report_dict[avg]['support']}")

if __name__ == "__main__":
    caminho_dados = 'data/ToLD-BR.csv'
    feedback_dados = 'data/feedback.csv'
    caminho_modelo = 'ml/modelo_tree'
    caminho_vetorizador = 'ml/vetorizador_tree.pkl'

    # Carregar e preprocessar os dados
    dados_iniciais = carregar_e_preprocessar_dados(caminho_dados)
    dados_retreinados = carregar_e_preprocessar_dados(caminho_dados, feedback_dados)

    # Treinar o modelo inicial
    modelos_iniciais, vetorizador_inicial, metricas_iniciais = treinarModelo(dados_iniciais)

    # Salvar o modelo inicial
    salvarModelo(modelos_iniciais, vetorizador_inicial, caminho_modelo, caminho_vetorizador)

    # Avaliar o modelo inicial
    metricas_finais = avaliarModelo(modelos_iniciais, vetorizador_inicial, dados_retreinados)

    # Imprimir as métricas de avaliação
    imprimir_metricas(metricas_iniciais, metricas_finais)

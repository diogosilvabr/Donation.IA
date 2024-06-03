import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
import joblib
import re
import string
import os
from datetime import datetime
import lightgbm as lgb
from scipy.stats import uniform, randint
from sklearn.metrics import make_scorer, f1_score

# Função para preprocessar o texto
def preprocessarTexto(texto):
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = texto.strip()
    return texto

# Função para carregar e preprocessar os dados
def carregar_e_preprocessar_dados(caminho_dados, caminho_feedback=None):
    # Carrega os dados do CSV
    df = pd.read_csv(caminho_dados)
    # Aplica a função de preprocessamento
    df['textoLimpo'] = df['text'].apply(preprocessarTexto)

    if caminho_feedback:
        # Carrega feedback adicional se disponível
        df_feedback = pd.read_csv(caminho_feedback)
        # Aplica preprocessamento ao feedback
        df_feedback['textoLimpo'] = df_feedback['text'].apply(preprocessarTexto)
        # Combina os dados principais com o feedback
        df = pd.concat([df, df_feedback], ignore_index=True)

    # Remove linhas com valores nulos na coluna `inappropriate`
    df = df.dropna(subset=['inappropriate'])
    return df

# Função para treinar o modelo com SMOTE e validação cruzada
def treinarModelo(dados):
    # Configura o vetorizador TF-IDF
    vetorizador = TfidfVectorizer(max_features=5000, stop_words='english')
    # Transforma os textos limpos em vetores
    X = vetorizador.fit_transform(dados['textoLimpo'])
    # Converte a coluna 'inappropriate' para inteiros
    y = dados['inappropriate'].astype(int)
    # Configura SMOTE para balanceamento
    smote = SMOTE(random_state=42)
    # Aplica SMOTE para balancear as classes
    X_smote, y_smote = smote.fit_resample(X, y)
    # Configura o modelo LightGBM
    modelo = lgb.LGBMClassifier(random_state=42)  # Configura o modelo LightGBM
    # Define os hiperparâmetros para a busca aleatória
    param_dist = {
        'num_leaves': randint(20, 50),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(50, 200)
    }
    # Define o F1-Score como métrica de avaliação
    f1_scorer = make_scorer(f1_score, average='weighted') 

    # Configura a busca aleatória para otimizar o modelo
    random_search = RandomizedSearchCV(modelo, param_distributions=param_dist, n_iter=50, cv=3, scoring=f1_scorer, random_state=42, n_jobs=-1)
    # Treina o modelo com busca aleatória
    random_search.fit(X_smote, y_smote)
    # Obtém o melhor modelo
    melhor_modelo = random_search.best_estimator_

    # Avalia o modelo com validação cruzada
    scores = cross_val_score(melhor_modelo, X_smote, y_smote, cv=3, scoring=f1_scorer)
    # Calcula a média do F1-Score
    f1_media = scores.mean()

    # Treina o melhor modelo com todos os dados
    melhor_modelo.fit(X_smote, y_smote)

    # Retorna o modelo, vetorizador e a média do F1-Score
    return melhor_modelo, vetorizador, f1_media

# Função para salvar o modelo, vetorizador e métricas
def salvarModelo(modelo, vetorizador, caminhoModelo, caminhoVetorizador, caminhoMetricas, f1):
    # Salva o modelo em um arquivo
    joblib.dump(modelo, caminhoModelo)
    # Salva o vetorizador em um arquivo
    joblib.dump(vetorizador, caminhoVetorizador)

    # Cria uma nova linha com a data e o F1-Score
    nova_linha = pd.DataFrame({
        'data': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'f1_score': [f1]
    })

    # Salva a linha no arquivo de métricas, criando ou adicionando ao arquivo
    if os.path.exists(caminhoMetricas):
        nova_linha.to_csv(caminhoMetricas, mode='a', header=False, index=False)
    else:
        nova_linha.to_csv(caminhoMetricas, index=False)
    # Imprime o F1-Score salvo
    print(f"F1-Score do modelo salvo: {f1:.2f}")  

# Função principal
if __name__ == "__main__":
    caminho_dados = 'data/base.csv'  # Caminho para o arquivo de dados
    caminho_feedback = 'data/feedback.csv'  # Caminho para o arquivo de feedback
    caminho_modelo = 'ml/modelo_inappropriate.pkl'  # Caminho para salvar o modelo
    caminho_vetorizador = 'ml/vetorizador_inappropriate.pkl'  # Caminho para salvar o vetorizador
    caminho_metricas = 'ml/historico_metricas.csv'  # Caminho para salvar as métricas

    # Carrega e preprocessa os dados
    dados_completos = carregar_e_preprocessar_dados(caminho_dados, caminho_feedback)
    # Treina o modelo
    modelo, vetorizador, f1 = treinarModelo(dados_completos)
    # Salva o modelo e vetorizador
    salvarModelo(modelo, vetorizador, caminho_modelo, caminho_vetorizador, caminho_metricas, f1)
    # Imprime o F1-Score médio
    print(f"F1-Score Médio do Modelo (Validação Cruzada): {f1:.2f}")

    # Monitoramento de melhoria
    if os.path.exists(caminho_metricas):
        # Carrega o histórico de métricas
        metricas = pd.read_csv(caminho_metricas)
        if 'f1_score' in metricas.columns:
            # Pega o último F1-Score
            ultima_f1 = metricas['f1_score'].iloc[-1] 
            if f1 > ultima_f1:
                # Informa se o F1-Score melhorou
                print("F1-Score melhorou.")
            else:
                # Informa se o F1-Score não melhorou
                print("F1-Score não melhorou.")
        else:
            # Informa se a coluna 'f1_score' não foi encontrada
            print("Não foi possível encontrar a coluna 'f1_score' no histórico de métricas.")

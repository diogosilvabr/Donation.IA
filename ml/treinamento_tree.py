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
    df = pd.read_csv(caminho_dados)
    df['textoLimpo'] = df['text'].apply(preprocessarTexto)
    
    if caminho_feedback:
        df_feedback = pd.read_csv(caminho_feedback)
        df_feedback['textoLimpo'] = df_feedback['text'].apply(preprocessarTexto)
        df = pd.concat([df, df_feedback], ignore_index=True)
    
    # Remover linhas com valores nulos na coluna `inappropriate`
    df = df.dropna(subset=['inappropriate'])  
    return df

# Função para treinar o modelo com SMOTE e validação cruzada
def treinarModelo(dados):
    vetorizador = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vetorizador.fit_transform(dados['textoLimpo'])
    y = dados['inappropriate'].astype(int)

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    modelo = lgb.LGBMClassifier(random_state=42)

    param_dist = {
        'num_leaves': randint(20, 50),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(50, 200)
    }

    f1_scorer = make_scorer(f1_score, average='weighted')

    random_search = RandomizedSearchCV(modelo, param_distributions=param_dist, n_iter=50, cv=3, scoring=f1_scorer, random_state=42, n_jobs=-1)
    random_search.fit(X_smote, y_smote)
    melhor_modelo = random_search.best_estimator_

    scores = cross_val_score(melhor_modelo, X_smote, y_smote, cv=3, scoring=f1_scorer)
    f1_media = scores.mean()

    melhor_modelo.fit(X_smote, y_smote)

    return melhor_modelo, vetorizador, f1_media

# Função para salvar o modelo, vetorizador e métricas
def salvarModelo(modelo, vetorizador, caminhoModelo, caminhoVetorizador, caminhoMetricas, f1):
    joblib.dump(modelo, caminhoModelo)
    joblib.dump(vetorizador, caminhoVetorizador)

    nova_linha = pd.DataFrame({
        'data': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'f1_score': [f1]
    })
    
    if os.path.exists(caminhoMetricas):
        nova_linha.to_csv(caminhoMetricas, mode='a', header=False, index=False)
    else:
        nova_linha.to_csv(caminhoMetricas, index=False)

    print(f"F1-Score do modelo salvo: {f1:.2f}")

# Função principal
if __name__ == "__main__":
    caminho_dados = 'data/base.csv'
    caminho_feedback = 'data/feedback.csv'
    caminho_modelo = 'ml/modelo_inappropriate.pkl'
    caminho_vetorizador = 'ml/vetorizador_inappropriate.pkl'
    caminho_metricas = 'ml/historico_metricas.csv'

    dados_completos = carregar_e_preprocessar_dados(caminho_dados, caminho_feedback)

    modelo, vetorizador, f1 = treinarModelo(dados_completos)

    salvarModelo(modelo, vetorizador, caminho_modelo, caminho_vetorizador, caminho_metricas, f1)

    print(f"F1-Score Médio do Modelo (Validação Cruzada): {f1:.2f}")

    # Monitoramento de melhoria
    if os.path.exists(caminho_metricas):
        metricas = pd.read_csv(caminho_metricas)
        if 'f1_score' in metricas.columns:
            ultima_f1 = metricas['f1_score'].iloc[-1]
            if f1 > ultima_f1:
                print("F1-Score melhorou.")
            else:
                print("F1-Score não melhorou.")
        else:
            print("Não foi possível encontrar a coluna 'f1_score' no histórico de métricas.")

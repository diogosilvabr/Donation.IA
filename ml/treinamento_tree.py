import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Função de pré-processamento
def preprocessarTexto(texto):
    # Implementar sua lógica de pré-processamento aqui
    return texto.lower()

# Carregar os dados
df = pd.read_csv('data/ToLD-BR.csv')

# Pré-processar os textos
df['textoLimpo'] = df['text'].apply(preprocessarTexto)

# Vetorização
vetorizador = TfidfVectorizer(max_features=5000)
X = vetorizador.fit_transform(df['textoLimpo'])

# Treinar modelos separados para cada classe
modelos = {}
for coluna in ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']:
    y = df[coluna]
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = DecisionTreeClassifier(random_state=42)
    modelo.fit(X_treino, y_treino)
    
    y_teste_pred = modelo.predict(X_teste)
    print(f"Relatório de classificação para {coluna}:\n")
    print(classification_report(y_teste, y_teste_pred))
    
    modelos[coluna] = modelo
    joblib.dump(modelo, f'ml/modelo_tree_{coluna}.pkl')

# Salvar o vetorizador
joblib.dump(vetorizador, 'ml/vetorizador_tree.pkl')

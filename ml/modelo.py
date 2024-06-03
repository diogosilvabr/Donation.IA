import joblib

# Define a função para carregar o modelo de machine learning e o vetorizador
def carregarModelo(caminho_modelo, caminho_vetorizador):
    # Carrega o modelo a partir do caminho especificado
    modelo = joblib.load(caminho_modelo)
    # Carrega o vetorizador a partir do caminho especificado
    vetorizador = joblib.load(caminho_vetorizador)
    # Retorna o modelo e o vetorizador carregados
    return modelo, vetorizador

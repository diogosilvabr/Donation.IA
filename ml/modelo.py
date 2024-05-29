import joblib

def carregarModelo(caminho_modelo, caminho_vetorizador):
    modelo = joblib.load(caminho_modelo)
    vetorizador = joblib.load(caminho_vetorizador)
    return modelo, vetorizador

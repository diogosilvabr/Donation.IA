import joblib

def carregarModelo(caminhoModelo, caminhoVetorizador):
    modelo = joblib.load(caminhoModelo)
    vetorizador = joblib.load(caminhoVetorizador)
    return modelo, vetorizador

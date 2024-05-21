import re
import string

def preprocessarTexto(texto):
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    texto = texto.lower()
    texto = re.sub(r'\d+', '', texto)
    texto = texto.strip()
    return texto

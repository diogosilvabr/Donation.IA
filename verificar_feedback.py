import pandas as pd

def carregar_feedback(caminho_feedback):
    colunas = [
        'text', 'textoLimpo', 'homophobia_pred', 'obscene_pred', 'insult_pred',
        'racism_pred', 'misogyny_pred', 'xenophobia_pred', 'homophobia_corr',
        'obscene_corr', 'insult_corr', 'racism_corr', 'misogyny_corr', 'xenophobia_corr'
    ]
    feedback_df = pd.read_csv(caminho_feedback, names=colunas, encoding='latin1')
    return feedback_df

def verificar_feedback(feedback_df):
    corretos = 0
    incorretos = 0
    inconsistencias = []

    for index, row in feedback_df.iterrows():
        for coluna in ['homophobia', 'obscene', 'insult', 'racism', 'misogyny', 'xenophobia']:
            pred_coluna = f"{coluna}_pred"
            corr_coluna = f"{coluna}_corr"
            if row[pred_coluna] == row[corr_coluna]:
                corretos += 1
            else:
                incorretos += 1
                inconsistencias.append({
                    "Texto": row['text'],
                    "Classe": coluna,
                    "Predição": row[pred_coluna],
                    "Correção": row[corr_coluna]
                })

    total = corretos + incorretos
    precisao = (corretos / total * 100) if total > 0 else 0
    return {
        "Total": total,
        "Corretos": corretos,
        "Incorretos": incorretos,
        "Precisão": precisao,
        "Inconsistências": inconsistencias
    }

def main():
    caminho_feedback = 'data/feedback.csv'
    feedback_df = carregar_feedback(caminho_feedback)
    resultados = verificar_feedback(feedback_df)

    print(f"Total de feedbacks: {resultados['Total']}")
    print(f"Corretos: {resultados['Corretos']}")
    print(f"Incorretos: {resultados['Incorretos']}")
    print(f"Precisão do feedback: {resultados['Precisão']:.2f}%")
    if resultados['Inconsistências']:
        print("\nInconsistências encontradas:")
        for inconsistencia in resultados['Inconsistências']:
            print(f"Texto: {inconsistencia['Texto']}, Classe: {inconsistencia['Classe']}, "
                  f"Predição: {inconsistencia['Predição']}, Correção: {inconsistencia['Correção']}")

if __name__ == "__main__":
    main()

import pandas as pd

# Metodo que recebe o .xlsx e o entrega ao pandas
def carregar_planilha(caminho):
    try:
        df = pd.read_excel(caminho)

        print("Planilha carregada com sucesso!")
        print(f"Linhas: {df.shape[0]}")
        print(f"Colunas: {df.shape[1]}")

        return df

    except Exception as erro:
        print(f"Erro ao carregar a planilha: {erro}")
        return None
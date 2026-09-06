import pandas as pd

# Método unitário para printar dados e permitir analise manual
def analisar_dados(df):

    print("\n===== INFORMAÇÕES DOS DADOS =====")

    print(df.info())

    print("\n===== PRIMEIRAS LINHAS =====")

    print(df.head())

    print("\n===== VALORES AUSENTES =====")

    print(df.isnull().sum())

    print("\n===== ESTATÍSTICAS =====")

    print(df.describe())

# Método para tratar os dados automaticamente e permitir analise de uma unica coluna
def preparar_dados(df, coluna_alvo):

    df = df.copy()

    # Remove linhas duplicadas
    df = df.drop_duplicates()

    # Remove linhas com valores ausentes
    df = df.dropna()

    # Separa as características da variável alvo
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]

    return X, y
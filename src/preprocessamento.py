import pandas as pd

# Características do questionário (q1 a q10) utilizadas como entrada do SVM
COLUNAS_QUESTIONARIO = [f"q{i}" for i in range(1, 11)]

# Colunas de identificação/controle: não são utilizadas como características do modelo
COLUNAS_IDENTIFICACAO = ["id_usuario", "data_hora"]

# Rótulo (target) presente apenas em dados_treinamento
COLUNA_ALVO = "area_ti_atual"


def _validar_colunas_obrigatorias(df, colunas_obrigatorias):
    faltantes = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]

    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes na planilha: {faltantes}")


def _filtrar_respostas_validas(df):
    df = df.copy()

    for coluna in COLUNAS_QUESTIONARIO:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    linhas_antes = len(df)

    respostas_validas = df[COLUNAS_QUESTIONARIO].apply(lambda serie: serie.between(1, 5))
    mascara_validas = respostas_validas.all(axis=1)

    df = df[mascara_validas]

    linhas_removidas = linhas_antes - len(df)
    if linhas_removidas > 0:
        print(f"Aviso: {linhas_removidas} registro(s) com respostas inválidas ou incompletas foram removidos.")

    return df


def analisar_dados(df, titulo="DADOS"):

    print(f"\n===== INFORMAÇÕES DOS {titulo} =====")
    print(df.info())

    print("\n===== PRIMEIRAS LINHAS =====")
    print(df.head())

    print("\n===== VALORES AUSENTES =====")
    print(df.isnull().sum())


def validar_dados_treinamento(df):
    """Valida e limpa a planilha dados_treinamento (exige area_ti_atual)."""

    _validar_colunas_obrigatorias(
        df,
        COLUNAS_IDENTIFICACAO + COLUNAS_QUESTIONARIO + [COLUNA_ALVO]
    )

    df = df.dropna(subset=[COLUNA_ALVO])
    df = _filtrar_respostas_validas(df)
    df = df.drop_duplicates(subset=["id_usuario"])

    if df.empty:
        raise ValueError("Nenhum registro válido encontrado em dados_treinamento.")

    return df.reset_index(drop=True)


def validar_dados_teste(df):
    """Valida e limpa a planilha dados_teste (não exige area_ti_atual)."""

    _validar_colunas_obrigatorias(df, COLUNAS_IDENTIFICACAO + COLUNAS_QUESTIONARIO)

    df = _filtrar_respostas_validas(df)
    df = df.drop_duplicates(subset=["id_usuario"])

    if df.empty:
        raise ValueError("Nenhum registro válido encontrado em dados_teste.")

    return df.reset_index(drop=True)


def separar_features_alvo(df, coluna_alvo=COLUNA_ALVO):
    """Separa as características (q1..q10) do rótulo (area_ti_atual)."""

    X = df[COLUNAS_QUESTIONARIO]
    y = df[coluna_alvo]

    return X, y

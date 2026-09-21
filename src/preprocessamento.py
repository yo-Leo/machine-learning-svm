import pandas as pd

# Declaração para as colunas do questionário (q1 a q10) utilizadas como entrada do SVM
COLUNAS_QUESTIONARIO = [f"q{i}" for i in range(1, 11)]

# Declaração para as colunas de identificação do usuario participante do quiz
COLUNAS_IDENTIFICACAO = ["id_usuario", "data_hora"]

# Declaração da coluna target (alvo buscado na analise SVM)
COLUNA_ALVO = "area_ti_atual"

# Verifica se todas as colunas necessárias estão presentes na planilha de dados
def _validar_colunas_obrigatorias(df, colunas_obrigatorias):
    faltantes = [coluna for coluna in colunas_obrigatorias if coluna not in df.columns]

    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes na planilha: {faltantes}")


def _filtrar_respostas_validas(df):
    # Cria uma copia dos dados recebidos, garantindo que não ocorra nenhuma alteração nos valores
    df = df.copy()

    # Converte cada coluna QUESTIONARIO (q1 a q10) para número, o que não for, é transformado em NaN
    for coluna in COLUNAS_QUESTIONARIO:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    # Processo que reserva a quantidade de dados antes da filtragem
    linhas_antes = len(df)
    # Metodo que verifica se os valores passados estão dentro do intervalo proposto, retornando true ou false para cada linha
    respostas_validas = df[COLUNAS_QUESTIONARIO].apply(lambda serie: serie.between(1, 5))
    # Separa apenas as linhas que retornaram 1 (true)
    mascara_validas = respostas_validas.all(axis=1)

    df = df[mascara_validas]

    # Apresenta o registro de linhas modificadas
    linhas_removidas = linhas_antes - len(df)
    if linhas_removidas > 0:
        print(f"Aviso: {linhas_removidas} registro com respostas inválidas ou incompletas foram removidos.")

    return df

# Printa os resultados apos a validação
def analisar_dados(df, titulo="DADOS"):

    print(f"\n===== INFORMAÇÕES DOS {titulo} =====")
    print(df.info())

    print("\n===== PRIMEIRAS LINHAS =====")
    print(df.head())

    print("\n===== VALORES AUSENTES =====")
    print(df.isnull().sum())

# Processo de validação e limpeza da planilha de treinamento
def validar_dados_treinamento(df):

    # Chama a função que verifica a existencia de colunas, passando a planilha e as colunas obrigatorias (exige COLUNA_ALVO)
    _validar_colunas_obrigatorias(
        df,
        COLUNAS_IDENTIFICACAO + COLUNAS_QUESTIONARIO + [COLUNA_ALVO]
    )

    # Descarta dados em que COLUNA_ALVO está vazia
    df = df.dropna(subset=[COLUNA_ALVO])
    df = _filtrar_respostas_validas(df)

    # Descarta registros de usuarios duplicados
    df = df.drop_duplicates(subset=["id_usuario"])

    if df.empty:
        raise ValueError("Nenhum registro válido encontrado em dados_treinamento.")

    return df.reset_index(drop=True)

# Processo de validação e limpeza da planilha de teste
def validar_dados_teste(df):
    # Chama a função que verifica a existencia de colunas, passando a planilha e as colunas obrigatorias (não exige COLUNA_ALVO)
    _validar_colunas_obrigatorias(df, COLUNAS_IDENTIFICACAO + COLUNAS_QUESTIONARIO)

    df = _filtrar_respostas_validas(df)
    # Descarta registros de usuarios duplicados
    df = df.drop_duplicates(subset=["id_usuario"])

    if df.empty:
        raise ValueError("Nenhum registro válido encontrado em dados_teste.")

    return df.reset_index(drop=True)

# Realiza a separação dos dados caracteristicas (X) dos dados alvo (Y)
def separar_features_alvo(df, coluna_alvo=COLUNA_ALVO):

    X = df[COLUNAS_QUESTIONARIO]
    y = df[coluna_alvo]

    return X, y

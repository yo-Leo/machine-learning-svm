from src.preprocessamento import COLUNAS_QUESTIONARIO


def prever_area_ti(modelo, scaler, df_teste):
    """
    Aplica o mesmo pré-processamento (scaler) do treinamento aos dados reais
    do quiz (dados_teste) e utiliza o modelo SVM treinado para prever a área
    de TI com maior afinidade de cada usuário.
    """

    X_teste = df_teste[COLUNAS_QUESTIONARIO]

    X_teste_escalado = scaler.transform(X_teste)

    previsoes = modelo.predict(X_teste_escalado)

    resultado = df_teste.copy()
    resultado["area_ti_predita"] = previsoes

    return resultado

import pandas as pd


def realizar_previsao(modelo, scaler, novos_dados):

    dados = pd.DataFrame(
        [novos_dados]
    )

    dados_escalados = scaler.transform(dados)

    previsao = modelo.predict(dados_escalados)

    return previsao[0]
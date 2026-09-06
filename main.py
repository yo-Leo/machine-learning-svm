from src.carregar_dados import carregar_planilha
from src.preprocessamento import (
    analisar_dados,
    preparar_dados
)

from src.modelo_svm import (
    dividir_dados,
    treinar_modelo
)

from src.avaliacao import avaliar_modelo

from src.visualizacao import (
    grafico_distribuicao_classes,
    grafico_matriz_confusao,
    grafico_pca
)

from src.previsao import realizar_previsao


CAMINHO_PLANILHA = "data/dados_svm.xlsx"

COLUNA_ALVO = "compra"


def main():

    df = carregar_planilha(
        CAMINHO_PLANILHA
    )

    if df is None:
        return

    analisar_dados(df)

    X, y = preparar_dados(
        df,
        COLUNA_ALVO
    )

    grafico_distribuicao_classes(y)

    (
        X_treino,
        X_teste,
        y_treino,
        y_teste
    ) = dividir_dados(X, y)


    print("\n===== DIVISÃO DOS DADOS =====")

    print(
        f"Treinamento: {len(X_treino)}"
    )

    print(
        f"Teste: {len(X_teste)}"
    )

    modelo, scaler = treinar_modelo(
        X_treino,
        y_treino
    )

    previsoes, matriz = avaliar_modelo(
        modelo,
        scaler,
        X_teste,
        y_teste
    )

    grafico_matriz_confusao(
        matriz
    )

    grafico_pca(
        X,
        y
    )

    novos_dados = {
        "idade": 31,
        "renda": 3500,
        "score": 680
    }

    resultado = realizar_previsao(
        modelo,
        scaler,
        novos_dados
    )

    print("\n===== NOVA PREVISÃO =====")

    print(
        f"Resultado previsto: {resultado}"
    )


if __name__ == "__main__":
    main()
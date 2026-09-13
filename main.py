from src.carregar_dados import carregar_planilha

from src.preprocessamento import (
    COLUNAS_QUESTIONARIO,
    COLUNA_ALVO,
    analisar_dados,
    validar_dados_treinamento,
    validar_dados_teste,
    separar_features_alvo
)

from src.modelo_svm import (
    dividir_dados,
    treinar_modelo
)

from src.avaliacao import avaliar_modelo

from src.previsao import prever_area_ti

from src.visualizacao import (
    grafico_dispersao_2d,
    grafico_dispersao_3d
)


CAMINHO_DADOS_TREINAMENTO = "data/dados_treinamento.xlsx"
CAMINHO_DADOS_TESTE = "data/dados_teste.xlsx"


def main():

    # Carregar e validar dados de treinamento 

    df_treinamento = carregar_planilha(CAMINHO_DADOS_TREINAMENTO)

    if df_treinamento is None:
        return

    df_treinamento = validar_dados_treinamento(df_treinamento)

    analisar_dados(df_treinamento, titulo="DADOS DE TREINAMENTO")

    # Separar características (X) e alvo (y)

    X, y = separar_features_alvo(df_treinamento, COLUNA_ALVO)

    print(f"\nÁreas de TI identificadas nos dados: {sorted(y.unique())}")

    # Dividir dados para treinamento/avaliação do modelo 

    (
        X_treino,
        X_teste_validacao,
        y_treino,
        y_teste_validacao
    ) = dividir_dados(X, y)

    print("\n===== DIVISÃO DOS DADOS (TREINAMENTO/VALIDAÇÃO) =====")

    print(f"Treinamento: {len(X_treino)}")
    print(f"Validação: {len(X_teste_validacao)}")

    # Treinar o modelo SVM 

    modelo, scaler = treinar_modelo(X_treino, y_treino)

    # Avaliar o modelo treinado 

    avaliar_modelo(
        modelo,
        scaler,
        X_teste_validacao,
        y_teste_validacao
    )

    # Carregar dados reais do quiz (dados_teste) 

    df_teste = carregar_planilha(CAMINHO_DADOS_TESTE)

    if df_teste is None:
        return

    df_teste = validar_dados_teste(df_teste)

    # Realizar a previsão real com o modelo já treinado 

    resultado = prever_area_ti(modelo, scaler, df_teste)

    colunas_resultado = (
        ["id_usuario", "data_hora"] + COLUNAS_QUESTIONARIO + ["area_ti_predita"]
    )

    print("\n===== RESULTADO DA CLASSIFICAÇÃO (DADOS REAIS DO QUIZ) =====")

    print(resultado[colunas_resultado].to_string(index=False))

    # Visualização 2D e 3D (treino + previsões reais) 

    grafico_dispersao_2d(
        X,
        y,
        resultado[COLUNAS_QUESTIONARIO],
        resultado["area_ti_predita"]
    )

    grafico_dispersao_3d(
        X,
        y,
        resultado[COLUNAS_QUESTIONARIO],
        resultado["area_ti_predita"]
    )


if __name__ == "__main__":
    main()

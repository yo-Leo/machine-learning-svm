from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def avaliar_modelo(modelo, scaler, X_teste, y_teste):

    X_teste_escalado = scaler.transform(X_teste) # Utiliza a mesma scala definida no treino para padronizar os dados

    previsoes = modelo.predict(X_teste_escalado) # Tenta realizar a previsão dos dados de teste

    acuracia = accuracy_score(y_teste, previsoes) # Mede número de acertos

    labels = sorted(y_teste.unique())

    matriz = confusion_matrix(y_teste, previsoes, labels=labels) # Mede resultado da matriz de confusão

    relatorio = classification_report(
        y_teste,
        previsoes,
        labels=labels
    )

    print("\n===== AVALIAÇÃO DO MODELO (VALIDAÇÃO) =====")

    print(f"Acurácia: {acuracia:.4f}")

    print("\n===== MATRIZ DE CONFUSÃO =====")

    print(matriz)

    print("\n===== CLASSIFICATION REPORT =====")

    print(relatorio)

    return previsoes, matriz, labels

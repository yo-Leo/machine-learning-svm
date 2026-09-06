from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def avaliar_modelo(modelo, scaler, X_teste, y_teste):

    X_teste_escalado = scaler.transform(X_teste)

    previsoes = modelo.predict(X_teste_escalado)

    acuracia = accuracy_score(y_teste, previsoes)

    matriz = confusion_matrix(y_teste, previsoes)

    relatorio = classification_report(
        y_teste,
        previsoes
    )

    print("\n===== RESULTADOS =====")

    print(f"Acurácia: {acuracia:.4f}")

    print("\n===== MATRIZ DE CONFUSÃO =====")

    print(matriz)

    print("\n===== CLASSIFICATION REPORT =====")

    print(relatorio)

    return previsoes, matriz
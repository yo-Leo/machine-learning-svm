from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def dividir_dados(X, y, test_size=0.2, random_state=42):

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_treino, X_teste, y_treino, y_teste


def criar_modelo():

    modelo = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )

    return modelo


def treinar_modelo(X_treino, y_treino):

    # StandardScaler utilizado para garantir que todas as respostas do questionário
    # estejam em uma escala condizente de comparação antes de treinar o SVM.

    scaler = StandardScaler()

    X_treino_escalado = scaler.fit_transform(X_treino)

    modelo = criar_modelo()

    modelo.fit(X_treino_escalado, y_treino)

    return modelo, scaler

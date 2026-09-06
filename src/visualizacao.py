import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


def grafico_distribuicao_classes(y):

    plt.figure(figsize=(8, 5))

    sns.countplot(x=y)

    plt.title("Distribuição das Classes")
    plt.xlabel("Classe")
    plt.ylabel("Quantidade")

    plt.tight_layout()

    plt.savefig("graphs/distribuicao_classes.png")

    plt.show()


def grafico_matriz_confusao(matriz):

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Matriz de Confusão")
    plt.xlabel("Classe Prevista")
    plt.ylabel("Classe Real")

    plt.tight_layout()

    plt.savefig("graphs/matriz_confusao.png")

    plt.show()


def grafico_pca(X, y):

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y,
        cmap="viridis"
    )

    plt.title("Visualização dos Dados utilizando PCA")

    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")

    plt.colorbar(scatter, label="Classe")

    plt.tight_layout()

    plt.savefig("graphs/dados_pca.png")

    plt.show()
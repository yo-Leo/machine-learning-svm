import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - necessário para habilitar a projeção 3D


def grafico_distribuicao_classes(y):

    plt.figure(figsize=(9, 5))

    sns.countplot(x=y, order=sorted(y.unique()))

    plt.title("Distribuição das Áreas de TI (Dados de Treinamento)")
    plt.xlabel("Área de TI")
    plt.ylabel("Quantidade de Usuários")
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    plt.savefig("graphs/distribuicao_classes.png")

    plt.show()


def grafico_matriz_confusao(matriz, labels):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title("Matriz de Confusão")
    plt.xlabel("Área Prevista")
    plt.ylabel("Área Real")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig("graphs/matriz_confusao.png")

    plt.show()


def _reduzir_dimensionalidade(X_treino, X_teste, n_componentes):
    """
    Reduz as 10 características do questionário (q1..q10) para n_componentes
    dimensões via PCA, ajustando a técnica sobre treino + teste combinados
    para que ambos fiquem no mesmo espaço de visualização.
    """

    X_completo = pd.concat([X_treino, X_teste], ignore_index=True)

    pca = PCA(n_components=n_componentes)
    componentes = pca.fit_transform(X_completo)

    n_treino = len(X_treino)

    return componentes[:n_treino], componentes[n_treino:]


def _cores_por_classe(y_treino, y_teste_predita):
    classes = sorted(pd.concat([y_treino, y_teste_predita]).unique())
    paleta = sns.color_palette("tab10", n_colors=len(classes))

    return classes, dict(zip(classes, paleta))


def grafico_dispersao_2d(X_treino, y_treino, X_teste, y_teste_predita):

    componentes_treino, componentes_teste = _reduzir_dimensionalidade(X_treino, X_teste, 2)

    classes, cor_por_classe = _cores_por_classe(y_treino, y_teste_predita)

    plt.figure(figsize=(10, 7))

    for classe in classes:
        mascara = (y_treino == classe).to_numpy()
        if mascara.any():
            plt.scatter(
                componentes_treino[mascara, 0],
                componentes_treino[mascara, 1],
                color=cor_por_classe[classe],
                marker="o",
                label=f"{classe} (treino)",
                edgecolor="black",
                linewidth=0.3,
                alpha=0.8
            )

    for classe in classes:
        mascara = (y_teste_predita == classe).to_numpy()
        if mascara.any():
            plt.scatter(
                componentes_teste[mascara, 0],
                componentes_teste[mascara, 1],
                color=cor_por_classe[classe],
                marker="*",
                s=250,
                label=f"{classe} (teste - previsto)",
                edgecolor="black",
                linewidth=0.6
            )

    plt.title("Dispersão 2D das Áreas de TI (Redução via PCA)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    plt.savefig("graphs/dispersao_2d.png")

    plt.show()


def grafico_dispersao_3d(X_treino, y_treino, X_teste, y_teste_predita):

    componentes_treino, componentes_teste = _reduzir_dimensionalidade(X_treino, X_teste, 3)

    classes, cor_por_classe = _cores_por_classe(y_treino, y_teste_predita)

    fig = plt.figure(figsize=(11, 8))
    eixo = fig.add_subplot(111, projection="3d")

    for classe in classes:
        mascara = (y_treino == classe).to_numpy()
        if mascara.any():
            eixo.scatter(
                componentes_treino[mascara, 0],
                componentes_treino[mascara, 1],
                componentes_treino[mascara, 2],
                color=cor_por_classe[classe],
                marker="o",
                label=f"{classe} (treino)",
                edgecolor="black",
                linewidth=0.3,
                alpha=0.8
            )

    for classe in classes:
        mascara = (y_teste_predita == classe).to_numpy()
        if mascara.any():
            eixo.scatter(
                componentes_teste[mascara, 0],
                componentes_teste[mascara, 1],
                componentes_teste[mascara, 2],
                color=cor_por_classe[classe],
                marker="*",
                s=250,
                label=f"{classe} (teste - previsto)",
                edgecolor="black",
                linewidth=0.6
            )

    eixo.set_title("Dispersão 3D das Áreas de TI (Redução via PCA)")
    eixo.set_xlabel("Componente Principal 1")
    eixo.set_ylabel("Componente Principal 2")
    eixo.set_zlabel("Componente Principal 3")
    eixo.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    plt.savefig("graphs/dispersao_3d.png")

    plt.show()

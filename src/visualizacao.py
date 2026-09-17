import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - necessário para habilitar a projeção 3D

from src.modelo_svm import criar_modelo


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


def _treinar_modelo_visualizacao(componentes, rotulos):
    """
    O SVM real do pipeline é treinado com as 10 características originais
    (q1..q10), cujo hiperplano não pode ser desenhado diretamente. Para
    ilustrar visualmente a fronteira de decisão, um SVM auxiliar (mesmos
    hiperparâmetros de src.modelo_svm.criar_modelo) é treinado apenas sobre
    as componentes principais já reduzidas (2D ou 3D).
    """

    modelo_visualizacao = criar_modelo()
    modelo_visualizacao.fit(componentes, rotulos)

    return modelo_visualizacao


def _mapa_indices_classe(classes):
    return {classe: indice for indice, classe in enumerate(classes)}


def grafico_dispersao_2d(X_treino, y_treino, X_teste, y_teste_predita):

    componentes_treino, componentes_teste = _reduzir_dimensionalidade(X_treino, X_teste, 2)
    componentes_completo = np.vstack([componentes_treino, componentes_teste])
    rotulos_completo = np.concatenate([y_treino.to_numpy(), y_teste_predita.to_numpy()])

    classes, cor_por_classe = _cores_por_classe(y_treino, y_teste_predita)
    paleta = [cor_por_classe[classe] for classe in classes]
    indice_por_classe = _mapa_indices_classe(classes)

    modelo_visualizacao = _treinar_modelo_visualizacao(componentes_completo, rotulos_completo)

    margem = 1.0
    resolucao = 300

    x_min, x_max = componentes_completo[:, 0].min() - margem, componentes_completo[:, 0].max() + margem
    y_min, y_max = componentes_completo[:, 1].min() - margem, componentes_completo[:, 1].max() + margem

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolucao),
        np.linspace(y_min, y_max, resolucao)
    )

    grade = np.column_stack([xx.ravel(), yy.ravel()])
    previsoes_grade = modelo_visualizacao.predict(grade)
    zz = np.vectorize(indice_por_classe.get)(previsoes_grade).reshape(xx.shape)

    cmap = ListedColormap(paleta)
    norm = BoundaryNorm(np.arange(len(classes) + 1) - 0.5, cmap.N)

    plt.figure(figsize=(10, 7))

    # Regiões coloridas = decisão do SVM; linhas pretas = hiperplano de separação entre as classes
    plt.contourf(xx, yy, zz, cmap=cmap, norm=norm, alpha=0.25)
    plt.contour(xx, yy, zz, colors="black", linewidths=0.6, alpha=0.6)

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

    plt.title("Gráfico de Dispersão 2D")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    plt.savefig("graphs/dispersao_2d.png")

    plt.show()


def grafico_dispersao_3d(X_treino, y_treino, X_teste, y_teste_predita):

    componentes_treino, componentes_teste = _reduzir_dimensionalidade(X_treino, X_teste, 3)
    componentes_completo = np.vstack([componentes_treino, componentes_teste])
    rotulos_completo = np.concatenate([y_treino.to_numpy(), y_teste_predita.to_numpy()])

    classes, cor_por_classe = _cores_por_classe(y_treino, y_teste_predita)
    indice_por_classe = _mapa_indices_classe(classes)

    modelo_visualizacao = _treinar_modelo_visualizacao(componentes_completo, rotulos_completo)

    margem = 1.0
    resolucao = 35

    minimos = componentes_completo.min(axis=0) - margem
    maximos = componentes_completo.max(axis=0) + margem

    xx, yy, zz = np.meshgrid(
        np.linspace(minimos[0], maximos[0], resolucao),
        np.linspace(minimos[1], maximos[1], resolucao),
        np.linspace(minimos[2], maximos[2], resolucao)
    )

    grade = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    previsoes_grade = modelo_visualizacao.predict(grade)
    indices_grade = np.vectorize(indice_por_classe.get)(previsoes_grade).reshape(xx.shape)

    # Um ponto da grade pertence ao hiperplano de separação quando pelo menos
    # um vizinho imediato (em x, y ou z) é classificado em uma área diferente.
    fronteira = np.zeros_like(indices_grade, dtype=bool)
    for eixo_grade in range(3):
        diferentes = np.diff(indices_grade, axis=eixo_grade) != 0
        fatia_anterior = [slice(None)] * 3
        fatia_posterior = [slice(None)] * 3
        fatia_anterior[eixo_grade] = slice(None, -1)
        fatia_posterior[eixo_grade] = slice(1, None)
        fronteira[tuple(fatia_anterior)] |= diferentes
        fronteira[tuple(fatia_posterior)] |= diferentes

    pontos_fronteira = grade[fronteira.ravel()]
    classes_fronteira = indices_grade[fronteira]

    fig = plt.figure(figsize=(11, 8))
    eixo = fig.add_subplot(111, projection="3d")

    # Pontos da superfície de fronteira = hiperplano de separação entre as classes
    for classe in classes:
        mascara_grade = classes_fronteira == indice_por_classe[classe]
        if mascara_grade.any():
            eixo.scatter(
                pontos_fronteira[mascara_grade, 0],
                pontos_fronteira[mascara_grade, 1],
                pontos_fronteira[mascara_grade, 2],
                color=cor_por_classe[classe],
                marker=".",
                s=18,
                alpha=0.5,
                linewidth=0,
                depthshade=False
            )

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
                alpha=0.9
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

    eixo.set_title("Gráfico de Dispersão 3D")
    eixo.set_xlabel("Componente Principal 1")
    eixo.set_ylabel("Componente Principal 2")
    eixo.set_zlabel("Componente Principal 3")
    eixo.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    plt.savefig("graphs/dispersao_3d.png")

    plt.show()

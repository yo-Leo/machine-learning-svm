# Machine Learning SVM

Protótipo didático de classificação com **Support Vector Machine (SVM)**, construído em Python com `scikit-learn`. O projeto lê uma base de clientes a partir de uma planilha Excel, treina um classificador SVM para prever se um cliente realiza uma compra (`compra`: 0 ou 1) com base em `idade`, `renda` e `score`, avalia o modelo e gera gráficos de apoio à análise.

## Estrutura do projeto

```
machine-learning-svm/
├── data/
│   └── dados_svm.xlsx        # Base de dados de entrada
├── graphs/                   # Gráficos gerados pela execução
│   ├── distribuicao_classes.png
│   ├── matriz_confusao.png
│   └── dados_pca.png
├── src/
│   ├── carregar_dados.py     # Leitura da planilha
│   ├── preprocessamento.py   # Análise exploratória e limpeza dos dados
│   ├── modelo_svm.py         # Divisão treino/teste, escala e treinamento do SVM
│   ├── avaliacao.py          # Métricas de avaliação do modelo
│   ├── previsao.py           # Previsão para novos dados
│   └── visualizacao.py       # Geração dos gráficos
├── main.py                   # Orquestra o pipeline de ponta a ponta
└── requirements.txt
```

## O que o projeto faz

O `main.py` executa um pipeline completo de Machine Learning supervisionado:

1. **Carrega** os dados de `data/dados_svm.xlsx`.
2. **Analisa** os dados (tipos, nulos, estatísticas descritivas) para fins de inspeção manual.
3. **Prepara** os dados: remove duplicados/nulos e separa variáveis preditoras (`X`) do alvo (`y`, coluna `compra`).
4. **Divide** os dados em treino (80%) e teste (20%), de forma estratificada.
5. **Treina** um `SVC` (SVM com kernel RBF) sobre os dados padronizados com `StandardScaler`.
6. **Avalia** o modelo no conjunto de teste (acurácia, matriz de confusão, classification report).
7. **Gera gráficos** (distribuição das classes, matriz de confusão e projeção PCA dos dados).
8. **Realiza uma previsão de exemplo** para um novo cliente fictício.

## Principais métodos

### `src/carregar_dados.py` — `carregar_planilha(caminho)`
Lê o arquivo `.xlsx` com `pandas.read_excel` e retorna o `DataFrame`. Encapsula o carregamento em um `try/except` para não interromper o pipeline caso o arquivo esteja ausente ou corrompido, retornando `None` nesse caso (verificado logo em seguida em `main.py`).

### `src/preprocessamento.py`
- **`analisar_dados(df)`**: imprime informações do `DataFrame` (`info`, `head`, contagem de nulos, `describe`). É um método apenas de inspeção, usado para apoiar a análise exploratória — não altera os dados.
- **`preparar_dados(df, coluna_alvo)`**: faz a limpeza mínima (remove duplicatas e linhas com valores ausentes) e separa as features (`X`) da variável alvo (`y`), dado o nome da coluna alvo (`"compra"`).

### `src/modelo_svm.py`
- **`dividir_dados(X, y)`**: usa `train_test_split` com `test_size=0.2`, `random_state=42` e `stratify=y` para garantir reprodutibilidade e manter a proporção das classes entre treino e teste.
- **`criar_modelo()`**: instancia o `SVC` com kernel `"rbf"`, `C=1.0` e `gamma="scale"` — configuração padrão do scikit-learn, adequada para fronteiras de decisão não lineares.
- **`treinar_modelo(X_treino, y_treino)`**: ajusta um `StandardScaler` aos dados de treino (essencial para SVM, sensível à escala das variáveis) e treina o modelo sobre os dados já padronizados. Retorna o modelo treinado **e** o `scaler`, pois o mesmo scaler precisa ser reaplicado depois em teste/novas previsões.

### `src/avaliacao.py` — `avaliar_modelo(modelo, scaler, X_teste, y_teste)`
Aplica o `scaler` (já ajustado no treino) ao conjunto de teste, gera as previsões e calcula `accuracy_score`, `confusion_matrix` e `classification_report`, imprimindo tudo no console. Retorna as previsões e a matriz de confusão para uso posterior (geração do gráfico).

### `src/previsao.py` — `realizar_previsao(modelo, scaler, novos_dados)`
Recebe um dicionário com os dados de um novo cliente, converte para `DataFrame` (mantendo os nomes das colunas esperados pelo scaler/modelo), aplica a mesma padronização do treino e retorna a classe prevista (`0` ou `1`).

### `src/visualizacao.py`
- **`grafico_distribuicao_classes(y)`**: gráfico de barras com a contagem de cada classe do alvo, salvo em `graphs/distribuicao_classes.png`.
- **`grafico_matriz_confusao(matriz)`**: heatmap da matriz de confusão, salvo em `graphs/matriz_confusao.png`.
- **`grafico_pca(X, y)`**: reduz as features a 2 componentes principais via `PCA` e plota um scatter colorido pela classe, salvo em `graphs/dados_pca.png` — útil para visualizar a separabilidade das classes em 2D.

## Como executar

Pré-requisitos: Python 3.10+ instalado.

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute o pipeline a partir da raiz do projeto:

```bash
python main.py
```

A execução imprime no console a análise exploratória, o resultado da divisão treino/teste, as métricas de avaliação e o resultado de uma previsão de exemplo, além de salvar os três gráficos na pasta `graphs/` (as janelas de plot também abrem interativamente, caso o ambiente suporte).

## Resultados entregues

- **Console**: informações do dataset (linhas, colunas, nulos, estatísticas descritivas), tamanho dos conjuntos de treino/teste, acurácia, matriz de confusão e `classification_report` do modelo, e a previsão para um novo cliente de exemplo (`idade=31, renda=3500, score=680`).
- **Gráficos** (pasta `graphs/`):
  - [`distribuicao_classes.png`](graphs/distribuicao_classes.png) — balanceamento das classes (compra vs. não compra).
  - [`matriz_confusao.png`](graphs/matriz_confusao.png) — desempenho do classificador no conjunto de teste.
  - [`dados_pca.png`](graphs/dados_pca.png) — projeção 2D (PCA) dos dados, colorida pela classe.

> **Observação:** a base (`data/dados_svm.xlsx`) contém apenas 8 registros — é um dataset de exemplo com finalidade didática, usado para demonstrar o pipeline de ponta a ponta (carregamento, pré-processamento, treinamento, avaliação e previsão). Os resultados numéricos (ex.: acurácia de 100%) refletem esse volume reduzido de dados e não devem ser interpretados como desempenho em um cenário de produção.

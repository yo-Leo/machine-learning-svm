# Machine Learning SVM — Classificação de Afinidade em TI

Pipeline de Machine Learning supervisionado que utiliza **Support Vector Machine (SVM)**, construído em Python com `scikit-learn`, para classificar a área de Tecnologia da Informação (Front-end, Back-end, DevOps/Infraestrutura, Dados/Data Science, Inteligência Artificial ou Segurança da Informação) com a qual um usuário possui **maior afinidade**, a partir das respostas de um questionário Likert de 10 afirmações (`q1` a `q10`, cada uma de 1 — Discordo Totalmente a 5 — Concordo Totalmente).

O resultado é uma **classificação de afinidade de perfil**, não uma determinação definitiva de profissão, competência ou carreira.

## Estrutura do projeto

```
machine-learning-svm/
├── data/
│   ├── dados_treinamento.xlsx   # Base histórica: q1..q10 + area_ti_atual (rótulo conhecido)
│   └── dados_teste.xlsx         # Respostas reais do quiz: q1..q10 (sem rótulo conhecido)
├── graphs/                      # Gráficos gerados pela execução
│   ├── dispersao_2d.png
│   └── dispersao_3d.png
├── src/
│   ├── carregar_dados.py        # Leitura genérica de planilhas .xlsx
│   ├── preprocessamento.py      # Validação, limpeza e separação de X/y
│   ├── modelo_svm.py            # Divisão treino/validação, escala e treinamento do SVM
│   ├── avaliacao.py             # Métricas de avaliação do modelo
│   ├── previsao.py              # Previsão da área de TI para dados reais do quiz
│   └── visualizacao.py          # Geração dos gráficos de dispersão PCA 2D/3D
├── main.py                      # Orquestra o pipeline de ponta a ponta
└── requirements.txt
```

## Fontes de dados

O algoritmo trabalha com duas planilhas distintas:

- **`dados_treinamento.xlsx`**: registros históricos com as respostas `q1`..`q10` e a coluna `area_ti_atual` (rótulo conhecido), usados para **treinar e avaliar** o modelo.
- **`dados_teste.xlsx`**: respostas reais extraídas do quiz, contendo apenas `q1`..`q10` (sem `area_ti_atual`), usadas para **prever** a área de TI de novos usuários com o modelo já treinado.

Em ambas, `id_usuario` e `data_hora` servem apenas para identificação/controle e **não** são utilizadas como características do modelo.

## O que o pipeline faz (`main.py`)

1. **Carrega e valida** `dados_treinamento.xlsx` (colunas obrigatórias, respostas entre 1 e 5, `area_ti_atual` preenchida).
2. **Separa** as características `X = [q1..q10]` do alvo `y = area_ti_atual`.
3. **Divide** os dados em treino/validação (80/20, estratificado).
4. **Treina** um `SVC` (SVM com kernel RBF) sobre os dados padronizados com `StandardScaler`.
5. **Avalia** o modelo no conjunto de validação (acurácia, matriz de confusão, classification report).
6. **Carrega e valida** `dados_teste.xlsx` (respostas reais do quiz, sem rótulo).
7. **Aplica o mesmo `StandardScaler`** ajustado no treino e **prevê** a `area_ti_predita` de cada usuário real.
8. **Gera os gráficos** "Gráfico de Dispersão 2D" e "Gráfico de Dispersão 3D" (via PCA), com treino e teste identificáveis.

## Principais módulos

### `src/carregar_dados.py` — `carregar_planilha(caminho)`
Lê um arquivo `.xlsx` com `pandas.read_excel`. Encapsula o carregamento em `try/except` para não interromper o pipeline caso o arquivo esteja ausente ou corrompido, retornando `None` nesse caso.

### `src/preprocessamento.py`
- **`analisar_dados(df, titulo)`**: inspeção exploratória (`info`, `head`, nulos).
- **`validar_dados_treinamento(df)`**: garante que as colunas obrigatórias (`id_usuario`, `data_hora`, `q1`..`q10`, `area_ti_atual`) existem, remove registros sem `area_ti_atual`, filtra respostas fora do intervalo 1–5 e remove duplicados por `id_usuario`.
- **`validar_dados_teste(df)`**: mesma validação, porém sem exigir `area_ti_atual`.
- **`separar_features_alvo(df, coluna_alvo)`**: separa `X` (`q1`..`q10`) de `y` (`area_ti_atual`).

### `src/modelo_svm.py`
- **`dividir_dados(X, y)`**: `train_test_split` com `test_size=0.2`, `random_state=42` e `stratify=y`.
- **`criar_modelo()`**: `SVC` com kernel `"rbf"`, `C=1.0` e `gamma="scale"`.
- **`treinar_modelo(X_treino, y_treino)`**: ajusta um `StandardScaler` aos dados de treino e treina o SVM sobre os dados padronizados. Retorna modelo **e** `scaler`, pois o mesmo `scaler` é reaplicado depois na validação e nos dados reais do quiz.

### `src/avaliacao.py` — `avaliar_modelo(modelo, scaler, X_teste, y_teste)`
Aplica o `scaler` já ajustado no treino, gera previsões e calcula `accuracy_score`, `confusion_matrix` e `classification_report` (multiclasse, com as áreas de TI identificadas nos dados). Retorna previsões, matriz e a lista de rótulos.

### `src/previsao.py` — `prever_area_ti(modelo, scaler, df_teste)`
Aplica o mesmo pré-processamento do treinamento às respostas reais do quiz (`dados_teste`) e retorna uma cópia do `DataFrame` com a coluna `area_ti_predita` preenchida pelo modelo.

### `src/visualizacao.py`
- **`grafico_dispersao_2d` / `grafico_dispersao_3d`**: reduzem as 10 respostas (`q1`..`q10`) para 2 ou 3 dimensões via `PCA`, plotando treino (círculos) e previsões reais do quiz (estrelas) coloridos por área de TI, permitindo visualizar agrupamentos e a separação entre classes. Títulos: "Gráfico de Dispersão 2D" e "Gráfico de Dispersão 3D".

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

A execução imprime no console a validação/análise dos dados de treinamento, a divisão treino/validação, as métricas de avaliação do modelo e a `area_ti_predita` para cada usuário de `dados_teste.xlsx`, além de salvar os dois gráficos de dispersão na pasta `graphs/` (as janelas de plot também abrem interativamente, caso o ambiente suporte).

## Resultados entregues

- **Console**: validação e análise exploratória de `dados_treinamento.xlsx`, tamanho dos conjuntos de treino/validação, acurácia/matriz de confusão/`classification_report` do modelo, e a tabela final com `id_usuario`, `data_hora`, `q1`..`q10` e `area_ti_predita` para cada usuário de `dados_teste.xlsx`.
- **Gráficos** (pasta `graphs/`):
  - [`dispersao_2d.png`](graphs/dispersao_2d.png) — "Gráfico de Dispersão 2D": projeção 2D (PCA) dos dados, com treino e previsões reais identificáveis por classe.
  - [`dispersao_3d.png`](graphs/dispersao_3d.png) — "Gráfico de Dispersão 3D": projeção 3D (PCA) dos dados, com treino e previsões reais identificáveis por classe.

> **Observação:** `data/dados_treinamento.xlsx` e `data/dados_teste.xlsx` contêm dados sintéticos gerados para fins didáticos, com o objetivo de demonstrar o fluxo completo de Machine Learning supervisionado (carregamento, validação, pré-processamento, treinamento, avaliação, previsão e visualização). Os resultados numéricos não devem ser interpretados como desempenho em um cenário de produção com dados reais de usuários.

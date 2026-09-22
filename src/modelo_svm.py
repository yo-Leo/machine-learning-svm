from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Realiza a divisão dos dados fornecidos, "test_size" reserva 20% dos dados para validação, "random_state" garante um valor fixo de reproduções, permitindo comparações validas
def dividir_dados(X, y, test_size=0.2, random_state=42):

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y # Garante a divisão igualitaria entre as classes de dados
    )

    return X_treino, X_teste, y_treino, y_teste


def criar_modelo():
    # Chama a classe SVC, especializada em classificação
    modelo = SVC(
        kernel="rbf", # Valor especifico para o calculo de Kernel, avisando para a classe que está lidando com uma classificação não linear
        C=1.0, # Parametro C com valor alto, diminuindo a tolerância com objetos fora da margem 
        gamma="scale" # Determina automaticamente o quanto objetos vão influenciar na fronteira
    )

    return modelo


def treinar_modelo(X_treino, y_treino):

    scaler = StandardScaler() # Método para padronizar as caracteristicas (X), impedindo que valores muito distoantes corrompam o processo
    X_treino_escalado = scaler.fit_transform(X_treino) # Metodo duplo, que analisa grandes diferenças de valores e realiza o tratamento 
    modelo = criar_modelo() # Grava o padrão de modelo pre estabelecida na variavel 
    modelo.fit(X_treino_escalado, y_treino) # Analisa os dados de treino com o padrão SVM passado

    return modelo, scaler

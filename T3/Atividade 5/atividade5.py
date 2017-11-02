# Atividade 5 - Disciplina IA369Y 2S 2017
# Classificador de emoções para base de fotos rotulada.

# Biblioteca pandas para carregar arquivo csv em dataframes.
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Para reformatar coordenadas.
def changeCoord(x):
    x = x.strip('()')
    return x.split(";")

#Efetua leitura do arquivo csv.
def read_csv():
    "Leitura das colunas: nome do arquivo, rótulo, 56 colunas de coordenadas."
    all_cols = list(range(0,65))
    remove_cols = set(range(2,9))
    cols = [col for col in all_cols if col not in remove_cols]
    return pd.read_csv('Faces_Disciplina\imagedb_CH_disciplina.csv', header=None, usecols=cols)

df = read_csv()

# Renomeia colunas
new_cols = []
new_cols.append('file_name')
new_cols.append('label')
for i in range(1, len(df.columns)-1):
    new_cols.append('p' + str(i))
df.columns = new_cols

# Faz o split das coordenadas, substituindo o formato (x1;y1) por x1 y1
for col in df.columns[2:]:
    s = df[col].apply(changeCoord)
    df['x'+col[1:]] = s.apply(lambda x: x[0])
    df['y'+col[1:]] = s.apply(lambda x: x[1])
    del df[col]

class_score = []
for i in range(0, 20):
    # Divisão do dataset em treinamento (80%) e validação (20%)
    msk = np.random.rand(len(df)) < 0.80
    training = df[msk]
    testing = df[~msk]

    # Seleciona sub-arrays e prepara para uso na sklearn
    training_set = training.loc[:,'x1':].as_matrix()
    training_lbls = training['label'].as_matrix()
    testing_set = testing.loc[:,'x1':].as_matrix()
    testing_lbls = testing['label'].as_matrix()
    classifier = SVC(kernel='linear')
    classifier.fit(training_set, training_lbls)

    # Retorna o score da classificação em relação aos rótulos.
    score = classifier.score(testing_set, testing_lbls)
    class_score.append(score)
    print("#{0:2} run score = {1:.2f}".format(i+1, score))

print(("Mean value after 20 runs: {0:.2f}".format(np.mean(class_score))))

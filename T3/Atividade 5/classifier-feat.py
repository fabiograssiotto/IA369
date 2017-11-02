# Atividade 5 - Disciplina IA369Y 2S 2017
# Classificador de emoções para base de fotos rotulada.

# Biblioteca pandas para carregar arquivo csv em dataframes.
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

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

def classify(df):
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
    return np.mean(class_score)


# Leitura e preparação do data set
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


# Executa classificação.
avg_score = classify(df)

# Execução com scaling de features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[df_scaled.columns[2:]] = scaler.fit_transform(df_scaled[df_scaled.columns[2:]])
avg_score_scaled = classify(df_scaled)

# Execução com marcos relativos.
df_relative = df.copy()
df_relative.iloc[:,2::] = df_relative.iloc[:,2::].apply(pd.to_numeric, errors='coerce')

# Seleciona colunas com coordenadas x e calcula a média por linha (de uma mesma amostra)
dfX = df_relative.iloc[:,range(2,len(df_relative.columns),2)]
dfX = dfX.apply(pd.to_numeric, errors='coerce')
dfX['meanX'] = dfX.mean(axis=1)

# Seleciona colunas com coordenadas y e calcula a média por linha (de uma mesma amostra)
dfY = df_relative.iloc[:,range(2,len(df_relative.columns),2)]
dfY = dfY.apply(pd.to_numeric, errors='coerce')
dfY['meanY'] = dfY.mean(axis=1)

# Ajuste das features baseadas em marcos relativos em relação a um ponto central na face.
for col in df_relative.columns[2::2]:
    df_relative[col] = abs(df_relative[col] - dfX['meanX'])

for col in df_relative.columns[3::2]:
    df_relative[col] = abs(df_relative[col] - dfY['meanY'])

avg_score_relative = classify(df_relative)

# Score após scaling das features
print(("Média de acurácia (sem scaling): {0:.2f}".format(avg_score)))
print(("Média de acurácia (com scaling): {0:.2f}".format(avg_score_scaled)))
print(("Média de acurácia (marcos relativos): {0:.2f}".format(avg_score_relative)))
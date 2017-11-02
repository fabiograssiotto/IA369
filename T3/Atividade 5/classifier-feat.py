# Atividade 5 - Disciplina IA369Y 2S 2017
# Classificador de emoções para base de fotos rotulada.
# Este arquivo executa manipulações das fetaures de marcos faciais para tentar atingir uma acurácia maior.

# Biblioteca pandas para carregar arquivo csv em dataframes.
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

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
    lbl_pred = classifier.predict(testing_set)
    print(classification_report(testing_lbls, lbl_pred))
        
    score = classifier.score(testing_set, testing_lbls)

    return score

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
print("Classificação inicial:")
score = classify(df)
print(("Acurácia (sem scaling): {0:.2f}".format(score)))

# Execução com scaling de features
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[df_scaled.columns[2:]] = scaler.fit_transform(df_scaled[df_scaled.columns[2:]])
print("Classificação com scaling:")
score = classify(df_scaled)
print(("Acurácia (com scaling): {0:.2f}".format(score)))

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
    df_relative[col] = (df_relative[col] - dfX['meanX'])

for col in df_relative.columns[3::2]:
    df_relative[col] = (df_relative[col] - dfY['meanY'])

# Adicione novas colunas ao fim do dataframe original
for col in df_relative.columns[2:]:
    df[col+'r'] = df_relative[col]

print("Classificação com marcos relativos:")
score = classify(df)
print(("Acurácia (com marcos relativos): {0:.2f}".format(score)))
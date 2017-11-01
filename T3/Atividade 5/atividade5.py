# Atividade 5 - Disciplina IA369Y 2S 2017
# Classificador de emoções para base de fotos rotulada.

# Biblioteca pandas para carregar arquivo csv em dataframes.
import pandas as pd
import numpy as np

def read_csv():
    "Leitura das colunas: nome do arquivo, rótulo, 56 colunas de coordenadas."
    all_cols = list(range(0,65))
    remove_cols = set(range(2,9))
    cols = [col for col in all_cols if col not in remove_cols]
    return pd.read_csv('Faces_Disciplina\imagedb_CH_disciplina.csv', header=None, usecols=cols)

df = read_csv()

# Renomear colunas
new_cols = []
new_cols.append('file_name')
new_cols.append('label')
for i in range(1, len(df.columns)-1):
    new_cols.append('p' + str(i))
df.columns = new_cols

# Fazer o split das coordenadas, substituindo o formato (x1; y1) por x1 y1
for col in df.columns[2:]:
    s = df[col].apply(lambda x: x.split('; '))
    df['x'+col[1:]] = s.apply(lambda x: x[0].strip('('))
    df['y'+col[1:]] = s.apply(lambda x: x[1].strip(')'))
    del df[col]

# Divisão do dataset em treinamento (80%) e validação (20%)
msk = np.random.rand(len(df)) < 0.8
training = df[msk]
testing = df[~msk]

# Para sklearn:
testing['label'].as_matrix()
testing.ix[:,'x1':].as_matrix()

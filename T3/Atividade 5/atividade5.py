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
    df = pd.read_csv('Faces_Disciplina\imagedb_CH_disciplina.csv', header=None, usecols=cols)
    return df

df = read_csv()

pd.set_option('display.max_rows', 3)
pd.set_option('display.max_columns', 6)

# Divisão do dataset em treinamento e validação
msk = np.random.rand(len(df)) < 0.8
training = df[msk]
testing = df[~msk]

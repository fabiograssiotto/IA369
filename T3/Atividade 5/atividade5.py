# Atividade 5 - Disciplina IA369Y 2S 2017
# Classificador de emoções para base de fotos rotulada.

# Biblioteca pandas para carregar arquivo csv em dataframes.
import pandas as pd

def read_csv():
    "Leitura das colunas: nome do arquivo, rótulo, 56 colunas de coordenadas."
    all_cols = list(range(0,65))
    remove_cols = set(range(2,9))
    cols = [col for col in all_cols if col not in remove_cols]
    df = pd.read_csv('Faces_Disciplina\imagedb_CH_disciplina.csv', header=None, usecols=cols)
    df.set_index(['arquivo', 'rótulo', 'p1', 'p2', 'p3'])
    return df

df = read_csv()

pd.set_option('display.max_rows', 3)
pd.set_option('display.max_columns', 6)


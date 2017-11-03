# Atividade 5 - Disciplina IA369Y 2S 2017
# Classificador de emoções para base de fotos rotulada.
# Este arquivo executa exploração dos dados de entrada usando matplotlib.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint

# Para reformatar coordenadas.
def get_coord(x):
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

# Definição de subplots
f, axarr = plt.subplots(1, 2)

row = randint(0, len(df.index))
x_list = []
y_list = []
for col in range(9,65):
    x, y = get_coord(df.get_value(row,col))
    x_list.append(x)
    y_list.append(y)
axarr[0].scatter(x_list, y_list, c=np.random.rand(3,), alpha=0.5)

for row in range(0, len(df.index)):
    x_list = []
    y_list = []
    for col in range(9,65):
        x, y = get_coord(df.get_value(row,col))
        x_list.append(x)
        y_list.append(y)
    axarr[1].scatter(x_list, y_list, c=np.random.rand(3,), alpha=0.5)
 
# Plot
axarr[0].set_title('Marcos faciais de uma amostra')
axarr[0].set_xlabel('x')
axarr[0].set_ylabel('y')
axarr[0].invert_yaxis()
axarr[0].set_xlim([0, 640])
axarr[0].set_ylim([480, 0])
axarr[1].set_title('Marcos faciais da base completa')
axarr[1].set_xlabel('x')
axarr[1].set_ylabel('y')
axarr[1].invert_yaxis()
axarr[1].set_xlim([0, 640])
axarr[1].set_ylim([480, 0])
plt.tight_layout()
plt.show()

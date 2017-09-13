# -*- coding: utf-8 -*-
# Problema 1 - Determinação de Valência em Manchetes de Jornais Brasileiros no 1o Semestre de 2017
# Dados necessários da NLTK: stopwords, rslp stemmer para a língua portuguesa.
import csv
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import NaiveBayesClassifier
from nltk.stem import RSLPStemmer
import matplotlib.pyplot as pyplot
import numpy as np
import datetime
import time
import os.path

# Para mostrar uma barra de progresso durante iterações.
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

# Função utilitária para conversão de meses em português.
# evita precisar a configuração do locale em outras máquinas.
def monthToNumber(monthName):
    if (monthName == 'janeiro'): 
        return 1
    elif (monthName == 'fevereiro'):
        return 2
    elif (monthName == 'março'):
        return 3
    elif (monthName == 'abril'):
        return 4
    elif (monthName == 'maio'):
        return 5
    elif (monthName == 'junho'):
        return 6
    elif (monthName == 'julho'):
        return 7
    elif (monthName == 'agosto'):
        return 8
    elif (monthName == 'setembro'):
        return 9
    elif (monthName == 'outubro'):
        return 10
    elif (monthName == 'novembro'):
        return 11
    else:
        return 12

# Retorna no formato de dicionário utilizado pelo classificador.
def word_feats(words):
        return dict([(word, True) for word in words])

# Inicialização
print('T2: Análise de Emoções em Textos\n\n')

# Leitura do arquivo csv. Utilizar encoding UTF8 para preservar acentuação.
with open('manchetesBrasildatabase.csv', encoding='utf8') as csvFile:
    readCsv = csv.reader(csvFile, delimiter=',',  quotechar="'")
    org_headlines = []
    dates = []
    sources = []
    headlines = []
    count = 0
    for row in readCsv:
        printProgressBar(count, 499, prefix = 'Lendo Manchetes: ', suffix = 'OK', length = 50)
        dates.append(datetime.datetime.strptime((row[0]+'/'+str(monthToNumber(row[1]))+'/'+row[2]), "%d/%m/%Y"))
        sources.append(row[3])
        headline = row[4]
        org_headlines.append(headline)
        # Remover nesta etapa de pre-processamento as 'stopwords' para restringir a análise, e fazer todas as palavras minúsculas.
        stop_words = stopwords.words('portuguese')
        tokenizer = RegexpTokenizer(r'\w+')
        # Considere apenas os stems das palavras para melhorar o score de valência.
        st = RSLPStemmer()
        content = [st.stem(word.lower()) for word in tokenizer.tokenize(headline) if word.lower() not in stop_words]
        headlines.append(content)
        count = count + 1
        time.sleep(0.001)
csvFile.close()

# Leitura do corpus OpLexicon 3.0 
#### Referência:
####          Souza, M., Vieira, R., Chishman, R., & Alves, I. M. (2011).
####          Construction of a Portuguese Opinion Lexicon from multiple resources.
####          8th Brazilian Symposium in Information and Human Language Technology - STIL. Mato Grosso, Brazil.
####
with open('lexico_v3.0.txt', encoding='utf8') as csvFile:
    readCsv = csv.reader(csvFile, delimiter=',')
    features = [[]]
    count = 0
    for row in readCsv:
        printProgressBar(count, 32210, prefix = 'Lendo Corpus:    ', suffix = 'OK', length = 50)
        feature = []
        feature.append(row[0])
        feature.append(row[1])
        feature.append(row[2])
        feature.append(row[3])
        features.append(feature)
        count = count + 1
csvFile.close()
features = features[1:]

# Análise da quantidade de palavras classificadas no Corpus em neutras, positivas ou negativas.
featVal = [val for (word, pos, val, sth) in features]
featPos = featVal.count('1')
featNeg = featVal.count('-1')
featNtr = featVal.count('0')

featLbls = ('Positivo', 'Negativo', 'Neutro')
featSizes = []
featSizes.append(featPos)
featSizes.append(featNeg)
featSizes.append(featNtr)
featColors = ['lightskyblue', 'lightcoral', 'gold']
 
# Plot
fig = pyplot.figure(figsize=(6,6))
fig.suptitle('Análise do Corpus', fontsize=14)
pyplot.pie(featSizes, labels=featLbls, colors=featColors,
        autopct='%1.1f%%', shadow=False)
 
pyplot.axis('equal')
pyplot.savefig('análise_corpus.png')

# Criação de um training set para o classificador de Bayes
# Use apenas o stem das palavras no training set.
training_set = [(word_feats([st.stem(word)]), valence) for (word, pos, valence,sth) in features]
classifier = NaiveBayesClassifier.train(training_set)

# Classificação das headlines do jornal e apresentação dos resultados.
with open('resultados.txt', 'w', encoding='utf8') as wfile:
    valences = []
    intensities = []
    wfile.write('{0} {1}\n'.format('Manchete'.center(53), 'Valência (%)'))
    wfile.write('{0}\n'.format("-"*66))
    count = 0
    for headline, org_headline in zip(headlines, org_headlines):
        printProgressBar(count, 499, prefix = 'Classificando:   ', suffix = 'OK', length = 50)
        valence = classifier.classify(word_feats(headline))
        dist = classifier.prob_classify(word_feats(headline))
        valences.append(int(valence))
        if valence == '1':
            valStr = 'Positiva'
            intensity = dist.prob('1')
        elif valence == '-1':
            valStr = 'Negativa'
            intensity = (-1)*dist.prob('-1')
        else:
            valStr = 'Neutra'
            intensity = 0
        # Conversão
        intPercent = 100*(intensity+1)/2
        intensities.append(intPercent)
        wfile.write('{0:60.58} {1:5.2f}\n'.format(org_headline, intPercent))
        count = count + 1;
        time.sleep(0.001)
wfile.close()

# Geração de uma lista de amostras para análise do algoritmo.
sampleList = np.random.randint(0, len(headlines)-1, size=10)
with open('amostras.txt', 'w', encoding='utf8') as wfile:
    wfile.write('{0} {1}\n'.format('Manchete'.center(53), 'Valência (%)'))
    for sample in sampleList:
        wfile.write('{0:60.58} {1:5.2f}\n'.format(org_headlines[sample], intensities[sample]))
wfile.close()

# Grafíco com a dispersão das notícias ao longo do semestre
# Primeiro obtemos a média da intensidade das valências para um determinado mês.
l = list(zip(dates, intensities))
dez16Val = []
jan17Val = []
feb17Val = []
mar17Val = []
apr17Val = []
may17Val = []
jun17Val = []
jul17Val = []
ago17Val = []
for tup in l:
    if (tup[0].date().month == 12 and tup[0].date().year == 2016):
        dez16Val.append(tup[1])
    elif (tup[0].date().month == 1 and tup[0].date().year == 2017) :
        jan17Val.append(tup[1])
    elif (tup[0].date().month == 2 and tup[0].date().year == 2017) :
        feb17Val.append(tup[1])
    elif (tup[0].date().month == 3 and tup[0].date().year == 2017) :
        mar17Val.append(tup[1])
    elif (tup[0].date().month == 4 and tup[0].date().year == 2017) :
        apr17Val.append(tup[1])
    elif (tup[0].date().month == 5 and tup[0].date().year == 2017) :
        may17Val.append(tup[1])
    elif (tup[0].date().month == 6 and tup[0].date().year == 2017) :
        jun17Val.append(tup[1])  
    elif (tup[0].date().month == 7 and tup[0].date().year == 2017) :
        jul17Val.append(tup[1])
    elif (tup[0].date().month == 8 and tup[0].date().year == 2017) :
        ago17Val.append(tup[1]) 
monthValAverages = []
# Converte a média para escala 0-100%
monthValAverages.append(np.mean(dez16Val))
monthValAverages.append(np.mean(jan17Val))
monthValAverages.append(np.mean(feb17Val))
monthValAverages.append(np.mean(mar17Val))
monthValAverages.append(np.mean(apr17Val))
monthValAverages.append(np.mean(may17Val))
monthValAverages.append(np.mean(jun17Val))
monthValAverages.append(np.mean(jul17Val))
monthValAverages.append(np.mean(ago17Val))

# Criação do gráfico de barras
N = len(monthValAverages)
x = range(N)
width = 0.5
fig = pyplot.figure(figsize=(8, 4))
fig.suptitle('Valências por mês', fontsize=14)
pyplot.xlabel('Mês')
pyplot.ylabel('Valência (%)')
monthLabels = ['Dez/16', 'Jan/17', 'Fev/17', 'Mar/17', 'Abr/17', 'Maio/17', 'Jun/17', 'Jul/17', 'Ago/17']
pyplot.xticks(x, monthLabels)
pyplot.bar(x, monthValAverages, width, color="blue")
pyplot.savefig('valências_por_mês.png')

# Gráfico com dispersão de acordo com a fonte da manchete.
sourceValAverages = []
sourceSet = set(sources)
srcValList = list(zip(sources, intensities))
for source in sourceSet:
    valList = []
    for tup in srcValList:
        if (tup[0] == source) :
            valList.append(tup[1])
    sourceValAverages.append(np.mean(valList))

# Criação do gráfico de barras
N = len(sourceValAverages)
x = range(N)
width = 0.5
fig = pyplot.figure(figsize=(8,4))
fig.suptitle('Valências por publicação', fontsize=14)
pyplot.xlabel('Publicação')
pyplot.ylabel('Valência (%)')
sourceLabels = list(sourceSet)
pyplot.xticks(x, sourceLabels)
pyplot.bar(x, sourceValAverages, width, color="red")
pyplot.savefig('valências_por_publicação.png')

print('\nResultados:\n')
if (os.path.isfile('análise_corpus.png')):
    print('análise_corpus.png - Análise da distribuição de valências do corpus.')
else:
    print('Erro nos resultados')
if (os.path.isfile('amostras.txt')):
    print('amostras.txt - Seleção de 10 amostras de valências.')
else:
    print('Erro nos resultados')
if (os.path.isfile('resultados.txt')):
    print('resultados.txt - Classificação de valência (0-100%) das manchetes.')
else:
    print('Erro nos resultados')
if (os.path.isfile('valências_por_mês.png')):
    print('valências_por_mês.png - Médias das valências (0-100%) das manchetes por mês.')
else:
    print('Erro nos resultados')
if (os.path.isfile('valências_por_publicação.png')):
    print('valências_por_publicação.png - Médias das valências (0-100%) das manchetes por publicação.\n')
else:
    print('Erro nos resultados')
# -*- coding: utf-8 -*-
# Problema 1 - Determinação de Valência em Manchetes de Jornais Brasileiros no 1o Semestre de 2017
# Dados necessários da NLTK: stopwords, rslp de-stemmer para a língua portuguesa.
import csv
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import NaiveBayesClassifier
from nltk.stem import RSLPStemmer
import matplotlib.pyplot as pyplot
import numpy as np
import datetime

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
def word_feats(word):
        return dict([(word, True)])

# Leitura do arquivo csv. Utilizar encoding UTF8 para preservar acentuação.
with open('manchetesBrasildatabase.csv', encoding='utf8') as csvFile:
    readCsv = csv.reader(csvFile, delimiter=',',  quotechar="'")
    org_headlines = []
    dates = []
    sources = []
    headlines = []
    for row in readCsv:
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
csvFile.close()

# Ordenação das manchetes e fontes por data, necessário para análise temporal.
org_headlines = [org for _, org in sorted(zip(dates, org_headlines))]
headlines = [head for _, head in sorted(zip(dates, headlines))]
sources = [source for _, source in sorted(zip(dates, sources))]
dates_sorted = sorted(dates)

# Leitura do corpus OpLexicon 3.0 
#### Referência:
####          Souza, M., Vieira, R., Chishman, R., & Alves, I. M. (2011).
####          Construction of a Portuguese Opinion Lexicon from multiple resources.
####          8th Brazilian Symposium in Information and Human Language Technology - STIL. Mato Grosso, Brazil.
####
with open('lexico_v3.0.txt', encoding='utf8') as csvFile:
    readCsv = csv.reader(csvFile, delimiter=',')
    features = [[]]
    for row in readCsv:
        features = list(list(rec) for rec in csv.reader(csvFile, delimiter=','))
csvFile.close()

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
pyplot.savefig('análise_corpus.jpg')

# Criação de um training set para o classificador de Bayes
# Use apenas o stem das palavras no training set.
training_set = [(word_feats(st.stem(word)), valence) for (word,pos,valence,sth) in features]
classifier = NaiveBayesClassifier.train(training_set)

# Classificação das headlines do jornal e apresentação dos resultados.
with open('resultados.txt', 'w', encoding='utf8') as wfile:
    valences = []
    wfile.write('{0} {1}\n'.format('Manchete'.center(73), 'Valência (0-100%)'))
    wfile.write('{0}\n'.format("-"*91))
    for headline,org_headline in zip(headlines, org_headlines):
        valSum = 0
        for word in headline:
            valSum = valSum + int(classifier.classify(word_feats(word.lower())))
        # Encontra a média para as palavras classificadas e 
        # converte para escala 0-100 a valência encontrada
        valence = 100*((valSum/len(headline))+1)/2
        valences.append(valence)
        wfile.write('{0:85} {1:2}%\n'.format(org_headline, int(valence)))
wfile.close()

# Grafíco com a dispersão das notícias ao longo do semestre
# Primeiro obtemos a média das valências para um determinado mês.
l = list(zip(dates_sorted, valences))
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
fig = pyplot.figure(figsize=(8,4))
fig.suptitle('Valências por mês', fontsize=14)
pyplot.xlabel('Mês')
pyplot.ylabel('Valência (%)')
monthLabels = ['Dez/16', 'Jan/17', 'Fev/17', 'Mar/17', 'Apr/17', 'Maio/17', 'Jun/17', 'Jul/17', 'Ago/17']
pyplot.xticks(x, monthLabels)
pyplot.bar(x, monthValAverages, width, color="blue")
pyplot.savefig('valências_por_mês.jpg')

# Gráfico com dispersão de acordo com a fonte da manchete.
sourceValAverages = []
sourceSet = set(sources)
srcValList = list(zip(sources, valences))
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
pyplot.savefig('valências_por_publicação.jpg')

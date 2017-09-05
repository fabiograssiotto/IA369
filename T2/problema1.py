# -*- coding: utf-8 -*-
# Problema 1 - Determinação de Valência em Manchetes de Jornais Brasileiros no 1o Semestre de 2017
# Dados necessários da NLTK: stopwords.
import nltk, csv
import nltk.data
from nltk.corpus import stopwords
from nltk.corpus.reader import WordListCorpusReader
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

# Leitura do arquivo csv. Utilizar encoding UTF8 para preservar acentuação.
with open('manchetesBrasildatabase.csv', encoding='utf8') as csvFile:
    readCsv = csv.reader(csvFile, delimiter=',')
    headlines = []
    for row in readCsv:
        # O único campo relevante para o algoritmo é a manchete em si.
        headline = row[4]
        # Remover nesta etapa de pre-processamento as 'stopwords' para restringir a análise, e fazer todas as palavras minúsculas.
        stop_words = stopwords.words('portuguese')
        tokenizer = RegexpTokenizer(r'\w+')
        content = [word.lower() for word in tokenizer.tokenize(headline) if word.lower() not in stop_words]
        headlines.append(content)

# Lista de headlines contém agora as palavras a serem processadas para encontrar a valência relativa.

# Leitura do corpus OpLexicon 3.0 
#### Referência:
####          Souza, M., Vieira, R., Chishman, R., & Alves, I. M. (2011).
####          Construction of a Portuguese Opinion Lexicon from multiple resources.
####          8th Brazilian Symposium in Information and Human Language Technology - STIL. Mato Grosso, Brazil.
####

reader = WordListCorpusReader('.', ['lexico_v3.0.txt'])
lines = [word_tokenize(x) for x in reader.words()]
features = [(x[0],x[4]) for x in lines]


#from itertools import chain
#vocabulary = set(chain(*[word_tokenize(i.lower()) for i in lines]))

#x = {i:(i in headlines[0]) for i in vocabulary}
#from nltk import NaiveBayesClassifier as nbc
#classifier = nbc.train(x)
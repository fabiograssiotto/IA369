# -*- coding: utf-8 -*-
# Problema 1 - Determinação de Valência em Manchetes de Jornais Brasileiros no 1o Semestre de 2017
# Dados necessários da NLTK: stopwords.
import nltk, csv
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk import NaiveBayesClassifier

# Retorna no formato de dicionário utilizado pelo classificador.
def word_feats(word):
        return dict([(word, True)])

# Leitura do arquivo csv. Utilizar encoding UTF8 para preservar acentuação.
with open('manchetesBrasildatabase.csv', encoding='utf8') as csvFile:
    readCsv = csv.reader(csvFile, delimiter=',')
    org_headlines = []
    headlines = []
    for row in readCsv:
        # O único campo relevante para o algoritmo é a manchete em si.
        headline = row[4]
        org_headlines.append(headline)
        # Remover nesta etapa de pre-processamento as 'stopwords' para restringir a análise, e fazer todas as palavras minúsculas.
        stop_words = stopwords.words('portuguese')
        tokenizer = RegexpTokenizer(r'\w+')
        content = [word.lower() for word in tokenizer.tokenize(headline) if word.lower() not in stop_words]
        headlines.append(content)
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
    for row in readCsv:
        features = list(list(rec) for rec in csv.reader(csvFile, delimiter=','))

csvFile.close()

# Criação de um training set para o classificador de Bayes
training_set = [(word_feats(word), valence) for (word,pos,valence,sth) in features]
classifier = NaiveBayesClassifier.train(training_set)

# Classificação das headlines do jornal.
for headline,org_headline in zip(headlines, org_headlines):
    valSum = 0
    for word in headline:
        valSum = valSum + int(classifier.classify(word_feats(word.lower())))
    print (org_headline + ' : ' + str(valSum/len(headline)))

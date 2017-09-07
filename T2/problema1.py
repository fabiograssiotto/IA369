# -*- coding: utf-8 -*-
# Problema 1 - Determinação de Valência em Manchetes de Jornais Brasileiros no 1o Semestre de 2017
# Dados necessários da NLTK: stopwords, rslp de-stemmer para a língua portuguesa.
import csv
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import NaiveBayesClassifier
from nltk.stem import RSLPStemmer

# Retorna no formato de dicionário utilizado pelo classificador.
def word_feats(word):
        return dict([(word, True)])

# Leitura do arquivo csv. Utilizar encoding UTF8 para preservar acentuação.
with open('manchetesBrasildatabase.csv', encoding='utf8') as csvFile:
    readCsv = csv.reader(csvFile, delimiter=',',  quotechar="'")
    org_headlines = []
    headlines = []
    for row in readCsv:
        # O único campo relevante para o algoritmo é a manchete em si.
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
# Use apenas o stem das palavras no training set.
training_set = [(word_feats(st.stem(word)), valence) for (word,pos,valence,sth) in features]
classifier = NaiveBayesClassifier.train(training_set)

# Classificação das headlines do jornal e apresentação dos resultados.
with open('resultados.txt', 'w', encoding='utf8') as wfile:
    wfile.write('{0} {1}\n'.format('Manchete'.center(73), 'Valência (0-100%)'))
    wfile.write('{0}\n'.format("-"*91))
    for headline,org_headline in zip(headlines, org_headlines):
        valSum = 0
        for word in headline:
            valSum = valSum + int(classifier.classify(word_feats(word.lower())))
        # Encontra a média para as palavras classificadas e 
        # converte para escala 0-100 a valência encontrada
        valence = 100*((valSum/len(headline))+1)/2
        wfile.write('{0:85} {1:2}%\n'.format(org_headline, int(valence)))
wfile.close()
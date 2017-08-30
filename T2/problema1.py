# Problema 1 - Determinacao de Valência em Manchetes de Jornais Brasileiros no 1 Semestre de 2017
# Dados necessários da NLTK: stopwords.
import nltk, csv
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

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


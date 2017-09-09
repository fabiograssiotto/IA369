T2 - Detecção de Emoções em Textos

Instruções para execução do código:

    1. Colocar no mesmo diretório os arquivos:
        
        lexico_v3.0.txt
        manchetesBrasildatabase.csv
        problema1.py

    2. Utilizar Python 3.
    3. Instalar as bibliotecas NLTK e matplotlib utilizando pip: 

        sudo pip3 install nltk
        sudo pip3 install matplotlib

    4. Fazer o download dos dados da NLTK:

        No terminal executar python3
        >> import nltk
        >> nltk.download()
        >> selecionar os componentes stopwords e rslp

    5. Executar no terminal:

        python3 problema1.py

    6. Coletar os resultados da classificação das manchetes nos arquivos:
    
         resultados.txt - Classificação de valência (Positiva/Negativa/Neutra) das manchetes.
         análise_corpus.jpg - Análise da distribuição de valências no Corpus utilizado.
         valências_por_mes.jpg - Médias das valências (0-100%) das manchetes por mês
         valências_por_publicação.jpg - Médias das valências (0-100%) das manchetes por publicação.
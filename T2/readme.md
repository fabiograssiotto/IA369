T2 - Detecção de Emoções em Textos

Instruções para execução do código:

    1. Unzipar o arquivo T2.zip e verificar que os arquivos estão presentes:
    
        unzip T2.zip

        lexico_v3.0.txt
        manchetesBrasildatabase.csv
        problema1.py
        readme.md (este arquivo)

    2. Utilizar Python 3, instalar os componentes pip3 e python3-tk (para os gráficos de análise).

        sudo apt install python3-pip
        sudo apt install python3-tk

    3. Instalar as bibliotecas NLTK e matplotlib utilizando pip: 

        sudo pip3 install nltk
        sudo pip3 install matplotlib
        
        Se estiver executando o código no Windows, instalar a biblioteca pillow (para geração de arquivos em jpg):
    
        pip3 install pillow

    4. Fazer o download dos dados da NLTK:

        No terminal executar python3
        >> import nltk
        >> nltk.download()
        >> selecionar os componentes stopwords e rslp

    5. Executar no terminal:

        python3 problema1.py

    6. Coletar os resultados da classificação das manchetes nos arquivos:
    
         análise_corpus.jpg - Análise da distribuição de valências no Corpus utilizado.
         amostras.txt - Seleção de 10 amostras de valências para determinação da performane do classificador. 
         resultados.txt - Classificação de valência (0-100%) das manchetes.
         valências_por_mes.jpg - Médias das valências (0-100%) das manchetes por mês
         valências_por_publicação.jpg - Médias das valências (0-100%) das manchetes por publicação.

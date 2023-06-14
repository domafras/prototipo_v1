##  UTILIZANDO BERT PARA PREVER A POPULARIDADE DE POSTAGENS NO YOUTUBE

Esta proposta de pesquisa científica é um trabalho de conclusão de curso de Ciência da Computação, desenvolvido pelos alunos:

-   Bryan Yassunori Tanaka
-   Leonardo Mafra Salin

O objetivo deste protótipo é compreender as bases de estudos que são referentes à coleta de dados de canais e vídeos no YouTube realizada em estudo anterior pelo PPGIA, e em sequência gerar previsões de popularidade nas publicações, se beneficiando do processamento de linguagem natural nas transcrições dos conteúdos extraídos.

Nesta primeira entrega, não será utilizado Processamento de Linguagem Natural, buscando estabelecer uma base sólida para comparação com as técnicas que serão desenvolvidas no segundo semestre, as quais levarão em conta padrões semânticos e estruturais dos conteúdos compreendidos por meio do PLN.

Acompanhe neste README o passo a passo para o desenvolvimento deste projeto.


## Arquivos

Disponíveis neste repositório
- Dados de canais: dataset_canais.csv
- Dados de vídeos: dataset_videos.csv

## EDA

Para este estudo, foi feita a união dos dataframes 'df_canais' e 'df_videos' usando o método merge(), tendo a coluna 'channel_id' como chave para unir as duas tabelas.

As colunas que possuem o mesmo nome entre os dois dataframes terão o sufixo:
-   _x no nome para referenciar a coluna do dataframe esquerdo (df_canais)
-   _y para referenciar a coluna do dataframe direito (df_videos)

**Dimensão:**
- 103 canais
- 38427 vídeos

**Dicionário: Canais**

-   url_channel: Url do canal
-   category: Categoria do canal
-   gender: Gênero do canal
-   channel_name: Nome do canal
-   country: País do Canal
-   year: Ano de Criação
-   channel_description: Descrição do canal
-   comment_count: Número de comentários
-   video_count: Total de vídeos publicados
-   view_count: Total de visualizações do canal
-   subscriber_count: Total de inscritos
-   aux_status: Status de canal ativo
-   updates_at: Última atualização do canal

**Dicionário: Vídeos**

-   channel_id: Id do canal de origem
-   channel_name: Nome do canal
-   video_title: Título do vídeo
-   video_desc: Descrição do vídeo
-   comment_count: Número de comentários
-   dislike_count: Número de dislikes
-   view_count: Número de views
-   like_count: Númerdo de likes
-   video_duration: Duração do vídeo
-   published_at: Data de publicação do vídeo

**Verificação**
- dados duplicados
- valores ausentes (NaN, -999)
- invariâncias
- variáveis numéricas ou categóricas
- correlações

## Pré-processamento

Em geral, antes da divisão dos dados, algums procedimentos foram executados para simplificar o conjunto de dados:
- remoção de linhas
- remoção de colunas
- conversão de variáveis categóricas

Devidamente documentados no **prototipo_v1.ipynb** caso seja necessário reverter.

Após a divisão dos dados em treino e teste, foi aplicado aos dados de treino algumas transformações como:
- codificação das variáveis categóricas
	- OneHotEncoder
- normalizando variáveis numéricas
	- MinMaxScaler ou StandardScaler

*O tratamento foi feito após o split e somente nos dados de treino a fim de evitar vazamento dos dados (data leakage)

##  Métrica de popularidade

Seguindo o estudo prévio descrito no artigo "YouTube Videos Prediction: Will this video be popular", a fim de utilizar como variável alvo, criamos uma equação que calcula a pontuação de popularidade para vídeos no YouTube com base em algumas métricas presentes nos vídeos da plataforma:

    # Criando nova coluna 'score' de popularidade
    df['score'] = (df['comment_count_y'] / df['view_count_y']) * df['like_count'] - (1.5 * df['dislike_count'])


Onde:

-   'score' é a pontuação de popularidade do vídeo;
-   'comment_count_y' é o número de comentários no vídeo;
-   'view_count_y' é o número de visualizações do vídeo;
-   'like_count' é o número de likes do vídeo;
-   'dislike_count' é o número de dislikes do vídeo;

Entretanto, não esta sendo aplicado da mesma maneira que o estudo propôs. Por conta disso, optamos nesse momento por também considerar o número de visualizações do vídeo como a métrica de popularidade, até mesmo por ser o fator principal de sucesso numa publicação na plataforma. (Cabe revisão)

## Machine Learning

Nessa etapa, uma série de técnicas foram aplicadas para garantir que o modelo pudesse aprender da melhor forma possível a partir dos dados fornecidos. Tudo começou com a escolha do **'score' como alvo** do nosso modelo, ou seja, a variável que queremos prever.

Em seguida, foi feita a **divisão entre treino e teste**. Essa separação é importante para que possamos avaliar o desempenho do modelo em dados que ele ainda não viu, de maneira a **evitar 'Data Leakage'** (esse termo refere-se ao vazamento de informações dos dados de teste para os de treinamento, o que pode prejudicar a capacidade do modelo de generalizar corretamente os dados).

Com a base divida, foram aplicados em pipelines específicos tratamentos de acordo com o tipo da váriavel:

Para as **variáveis numéricas**, foi testado MinMaxScaler e **StandardScaler**. Esse processo é importante pois as variáveis numéricas podem ter diferentes escalas e ordens de grandeza, o que pode prejudicar a convergência do modelo. O MinMaxScaler transforma as variáveis numéricas para uma escala específica (por exemplo, 0 a 1), o que ajuda a tornar o modelo mais robusto e StandardScaler lida bem com outliers.

Já para as **variáveis categóricas**, foi utilizado o **OneHotEncoding**. Esse processo é importante pois as variáveis categóricas não podem ser representadas numericamente diretamente, já que não existe uma ordem entre as categorias. O OneHotEncoding cria uma nova coluna para cada categoria, atribuindo valor 1 para a categoria presente em cada observação e valor 0 para as demais categorias.

Em seguida, é utilizado **validação cruzada**, analisado as pontuações em treino e em seguida, com a hiperparametrização no **GridSearch** ocorre treinamento de alguns modelos e predição, em conjunto as pontuações obtidas nas métricas mais relevantes em um problema de regressão.

## Métricas

Em termos gerais, esses modelos apresentaram resultados razoáveis, mas ainda há bastante espaço para melhorias, pois nenhum dos modelos foi capaz de explicar de maneira satisfatória a variação nos dados. Talvez seja necessário considerar outras variáveis ou ajustar os hiperparâmetros dos modelos para melhorar sua performance.

Quando a variável dependente que está sendo considerada é **'view_count_y'**, ou seja, somente o número de visualizações, os resultados são os seguintes:
- Regressão linear
	- R2-score: 0,75
	- MAE: 282339 unidades (views)
- Árvore de decisão
	- R2-score: 0,81
	- MAE: 199469 unidades (views)

Quando se aplica **'score'** como o alvo, ou seja, uma equação mais elaborada entre métricas que o YouTube fornece para medir a popularidade, os resultados são melhores para explicar a variabilidade e apresentam menor erro:
-   Regressão Linear:
    -   R2-score: 0.99
    -   MAE: 53.27 unidades (views)
-   Árvore de decisão:
    -   R2-score: 0.97
    -   MAE: 82.12 unidades (views)

Entretanto, como nesse momento não temos compreensão completa dessa proposta de métrica de popularidade, cabe revisão e adaptação da equação que gera essa variável dependente.

## Trabalhos futuros

Com a conclusão desta primeira versão do projeto, o foco principal foi compreender as bases de dados disponíveis para estudo. Neste protótipo, realizamos um teste inicial que se baseou no problema de regressão, mas existem alguns detalhes importantes que precisam ser discutidos e analisados mais profundamente para que possamos avançar.

Por exemplo, é fundamental escolher uma métrica de popularidade adequada (como 'score') e explorar modelos de regressão com seus hiperparâmetros específicos. Durante os testes iniciais, outros modelos também foram avaliados, como Random Forest, SVM e Naive Bayes, mas, devido ao elevado tempo de execução, optamos por não utilizá-los nesta etapa.

Também é importante revisar algumas decisões tomadas inicialmente, como a remoção de linhas e o tratamento de colunas das quais foram devidamente documentadas nesse protótipo. Além disso, há oportunidades interessantes para explorarmos comparando o que temos hoje com o processamento de conteúdos textuais presentes nas bases (como títulos, comentários e transcrições que ainda não estão disponíveis) usando técnicas de PLN (BERT).

Outras técnicas, como Feature Selection/Engineering e regularização, podem ser aplicadas para melhorar ainda mais o resultado do nosso modelo preditivo.

Embora esta seja apenas a primeira entrega do projeto, acreditamos que ela tenha alcançado seu objetivo principal de fornecer uma compreensão inicial dos dados disponíveis e de destacar as oportunidades futuras de trabalho que poderão contribuir para o sucesso do projeto.

Com a primeira etapa desse projeto, foi possível também visualizar horizontes para o decorrer do projeto:
- revisão EDA (Q&A, gráficos, outliers, análises multivariadas, aprimorar visualização)
- revisão Métrica de popularidade
- revisão Pré-processamento
- revisão Modelos
- revisão Métricas de avaliação (R2, adj-R2, EV, MSE, RMSE, MAE)
- Feature Engineering/Selection
- Regularização
- Processamento de linguagem natural (BERT)


## Contato
Sem restrições ao uso, sinta-se à vontade para enviar sugestões.

Feito com  ❤️  por  [Leonardo Mafra](https://www.linkedin.com/in/leomafra/)


# Bibliotecas
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate  # -> print(tabulate(df.head(2), headers=df.columns, tablefmt='pretty'))
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib

### README:
### 1 - Entendimento do Desafio que você quer resolver
### 2 - Entendimento da Empresa/Área

### 3 - Extração/Obtenção de Dados
caminho_bases = pathlib.Path('../dataset')
base_airbnb = pd.DataFrame()
meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

# Iterar sobre os arquivos dentro do caminho
for arquivo in caminho_bases.iterdir():
    # Adicionando a coluna mês e ano
    nome_mes = arquivo.name[:3]
    numero_mes = meses[nome_mes]
    ano = int(arquivo.name[-8:].replace('.csv', ''))
    df = pd.read_csv(caminho_bases / arquivo.name)
    df['ano'] = ano
    df['mes'] = numero_mes
    # Unindo todos dfs em um grande df
    base_airbnb = base_airbnb._append(df)

### 4 - Ajustes de Dados:
# Para fazer a limpeza de dados é interessante utilizar o excel
# Lista o nome de cada coluna
print(list(base_airbnb.columns))

# Cria um arquivo excel com as 1000 primeiras linhas
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')

# Tirar colunas desnecessárias:
# 1 - Ids, Links e informações não relevantes para o modelo
# 2 - Colunas repetidas EX: Data vs Ano/Mês
# 3 - Colunas preenchidas com texto livre -> Não serve para a análise
# 4 - Colunas vazias, ou em que quase todos os valores são iguais
print(base_airbnb[['experiences_offered']].value_counts())
print((base_airbnb['host_listings_count'] == base_airbnb['host_total_listings_count']).value_counts())

# Após a verificação no excel, escolhemos as colunas significativas para a nossa análise:
colunas = [
    'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count',
    'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
    'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee',
    'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews',
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value', 'instant_bookable', 'is_business_travel_ready',
    'cancellation_policy', 'ano', 'mes'
]
base_airbnb = base_airbnb.loc[:, colunas]

# Tratar valores None
print(base_airbnb.isnull().sum())

# Excluir as colunas: reviews, tempo de resposta, security deposit e taxa de limpeza
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() >= 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)

# Excluir as linhas onde temos poucos valores None:
base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())

# Verificar o tipo de dados de cada coluna:
print('-'*60)
print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])

# Tratando 'price'
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)

# Tratando 'extra_people'
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)

### 5 - Análise exploratória:
# Ver a correlação entre as features
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(base_airbnb.corr(numeric_only=True), annot=True, cmap='Greens', fmt='.2f', annot_kws={"size": 15})
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.show()

# Excluir outliers
def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    limSup = q3 + 1.5 * amplitude
    limInf = q1 - 1.5 * amplitude
    return limInf, limSup

def excluirOutliers(df, nome_coluna):
    qtde_linhas = df.shape[0]
    limInf, limSup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= limInf) & (df[nome_coluna] <= limSup), :]
    linhas_removidas = qtde_linhas - df.shape[0]
    return df, linhas_removidas, nome_coluna

def boxPlot(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=coluna, ax=ax1)
    ax1.set_title('Com outliers')
    lim_inf, lim_sup = limites(coluna)
    ax2.set_xlim(lim_inf, lim_sup)
    sns.boxplot(x=coluna, ax=ax2)
    ax2.set_title('Sem outliers')
    fig.suptitle('Com outliers vs Sem outliers', fontsize=16)
    plt.show()

def histograma(base, coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.histplot(data=base, x=coluna, element='bars', kde=True, binwidth=50, ax=ax1)
    ax1.set_title('Com outliers')
    lim_inf, lim_sup = limites(base[coluna])
    ax2.set_xlim(lim_inf, lim_sup)
    sns.histplot(data=base, x=coluna, element='bars', kde=True, binwidth=50, ax=ax2)
    ax2.set_title('Sem outliers')
    fig.suptitle('Com outliers vs Sem outliers', fontsize=16)
    plt.show()

def barra(base, coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    if isinstance(coluna, str):
        coluna = base[coluna]
    sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts(), ax=ax1)
    ax1.set_title('Com outliers')
    sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts(), ax=ax2)
    ax2.set_xlim(limites(coluna))
    ax2.set_title('Sem outliers')
    fig.suptitle('Com outliers vs Sem outliers', fontsize=16)
    plt.show()

def countplot(base, coluna):
    plt.figure(figsize=(15, 5))
    grafico = sns.countplot(x=coluna, data=base)
    grafico.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()

# Análise coluna price(contínuo)
boxPlot(base_airbnb['price'])
histograma(base_airbnb, 'price')
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'price')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna extra_people(contínuo)
boxPlot(base_airbnb['extra_people'])
histograma(base_airbnb, 'extra_people')
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'extra_people')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna host_listings_count(discreto)
boxPlot(base_airbnb['host_listings_count'])
barra(base_airbnb, base_airbnb['host_listings_count'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'host_listings_count')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna accommodates(discreto)
boxPlot(base_airbnb['accommodates'])
barra(base_airbnb, base_airbnb['accommodates'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'accommodates')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna bathrooms(discreto)
boxPlot(base_airbnb['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())
plt.show()
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'bathrooms')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna bedrooms(discreto)
boxPlot(base_airbnb['bedrooms'])
barra(base_airbnb, base_airbnb['bedrooms'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'bedrooms')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna beds(discreto)
boxPlot(base_airbnb['beds'])
barra(base_airbnb, base_airbnb['beds'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'beds')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna guests_included(discreto)
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())
base_airbnb = base_airbnb.drop('guests_included', axis=1)

# Análise coluna minimum_nights(discreto)
boxPlot(base_airbnb['minimum_nights'])
barra(base_airbnb, base_airbnb['minimum_nights'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'minimum_nights')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna maximum_nights(discreto)
boxPlot(base_airbnb['maximum_nights'])
barra(base_airbnb, base_airbnb['maximum_nights'])
base_airbnb = base_airbnb.drop('maximum_nights', axis=1)

# Análise coluna number_of_reviews(discreto)
boxPlot(base_airbnb['number_of_reviews'])
barra(base_airbnb, base_airbnb['number_of_reviews'])
base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)

# Análise coluna property_type(categórica)
print(base_airbnb['property_type'].value_counts())
countplot(base_airbnb, 'property_type')
tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = [tipo for tipo in tabela_tipos_casa.index if tabela_tipos_casa[tipo] < 2000]
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type'] == tipo, 'property_type'] = 'Other'
print(base_airbnb['property_type'].value_counts())

# Análise coluna room_type (categórica)
print(base_airbnb['room_type'].value_counts())
countplot(base_airbnb, 'room_type')

# Análise coluna bed_type(categórica)
print(base_airbnb['bed_type'].value_counts())
countplot(base_airbnb, 'bed_type')
colunas_agrupar = [tipo for tipo in base_airbnb['bed_type'].value_counts().index if base_airbnb['bed_type'].value_counts()[tipo] < 10000]
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type'] == tipo, 'bed_type'] = 'Other'
print(base_airbnb['bed_type'].value_counts())

# Análise coluna cancellation_policy(categórica)
print(base_airbnb['cancellation_policy'].value_counts())
countplot(base_airbnb, 'cancellation_policy')
colunas_agrupar = [tipo for tipo in base_airbnb['cancellation_policy'].value_counts().index if base_airbnb['cancellation_policy'].value_counts()[tipo] < 10000]
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy'] == tipo, 'cancellation_policy'] = 'strict'
print(base_airbnb['cancellation_policy'].value_counts())

# Análise coluna amenities(categórica)
print(base_airbnb['amenities'].value_counts())
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
base_airbnb = base_airbnb.drop('amenities', axis=1)

# Análise nova coluna n_amenities
boxPlot(base_airbnb['n_amenities'])
barra(base_airbnb, base_airbnb['n_amenities'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'n_amenities')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Criação do mapa de densidade para visualização das áreas com maiores preços
amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5, center=centro_mapa, zoom=10, mapbox_style='stamen-terrain')
mapa.update_layout(mapbox_style="open-street-map")
mapa.show()

### Encoding
# Boolean -> V/F = 1/0
colunas_vf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_vf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 'f', coluna] = 0

# Categóricas -> OneHotEncoding ou DummyVariables
colunas_cat = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_cat)

### Passos para criar construir um modelo de previsão:
# 1: Escolher o tipo de machine learning: Classificação X Regressão
#    - Classificação: Categorias (Separar entre A, B e C) (Ex: Diagnóstico doença, Spam, ...)
#    - Regressão: Valor específico (Número) (Ex: Preço, Velocidade, ...)
# 2: Definir as métricas para avaliação do modelo:
#    - R²
#        - De 0 a 1 -> Quanto maior, melhor
#        - Explicação: Mede o "quanto" dos valores o modelo consegue explicar
#        - Ex: 92% significa que o modelo consegue explicar 92% da variância dos dados a partir das informações que são dadas
#    - RSME (Raiz do erro quadrático médio)
#        - Pode ser qualquer valor
#        - Mede o "quanto" o modelo erra
# 3: Definir os modelos de regressão que iremos usar:
#    - Regressão linear
#        - Traça a "melhor" reta entre os pontos minimizando os erros
#    - Random forest
#        - Escolhe aleatoriamente características e procura o melhor lugar para dividir os dados. Usa amostras dos dados com repetição. Boa escolha geral, especialmente para dados menores e limpos
#    - Extra Trees
#        - Escolhe completamente aleatório onde dividir os dados. Pode usar todos os dados ou amostras sem repetição. Mais rápido e pode lidar bem com dados muito ruidosos
# 4: Treinar e testar o modelo
#    - Dividir a base de dados em 2 conjuntos: Treino e Teste
#    - 80% treino - Usam para aprender (Possuem acesso às características do imóvel (X) e ao preço (Y))
#    - 20% teste  - Usam para testar (Possuem as características do imóvel (X) e tentam precificá-lo (Y))
# 5: Comparar os modelos e escolher o melhor
#    - Calcularemos as duas métricas para cada modelo
#    - Escolhemos 1 métrica para ser a principal e usaremos a outra para critério de desempate
#    - Além disso, devemos levar em conta o tempo e a complexidade
# 6: Analisar o melhor modelo mais a fundo
#    - Entender a importância de cada feature para ver oportunidades de melhora
#    - Se uma feature/coluna não é utilizada, ou pouco importante, podemos retirar e ver o resultado
#    - Lembrar de avaliar: Métricas, velocidade e simplicidade do modelo
# 7: Fazer ajustes no melhor modelo
#    - Após tirarmos as features, treinar novamente o modelo e analisar as métricas
#    - Caso não haja diferenças significativas nas métricas, talvez a diferença de tempo ou complexidade seja razoável

# Definindo as métricas:
def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    rsme = np.sqrt(mean_squared_error(y_teste, previsao))
    return f"Modelo {nome_modelo}\n- R²: {r2:.2%}\n- RSME: {rsme:.2f}"

# Separação das variáveis
y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

# Separa os dados em treino e teste + treino do modelo
modelo_RandomForest = RandomForestRegressor()
modelo_LinearRegression = LinearRegression()
modelo_ExtraTrees = ExtraTreesRegressor()

modelos = {'RandomForest': modelo_RandomForest, 'LinearRegression': modelo_LinearRegression, 'ExtraTrees': modelo_ExtraTrees}

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

def escolher_modelo(modelos):
    for nome_modelo, modelo in modelos.items():
        # Treinar
        modelo.fit(x_train, y_train)
        # Testar
        previsao = modelo.predict(x_test)
        print(avaliar_modelo(nome_modelo, y_test, previsao))

#escolher_modelo(modelos)  # Essa linha foi comentada após a escolha do modelo

# # Após avaliarmos, o modelo ExtraTrees é o melhor modelo tanto no R² quanto no RSME
# print(modelo_ExtraTrees.feature_importances_)
# print(x_train.columns)
# importancia_features = pd.DataFrame(modelo_ExtraTrees.feature_importances_, x_train.columns)
#
# # Ordenar:
# importancia_features = importancia_features.sort_values(by=0, ascending=False)

# # Exibir em gráfico:
# plt.figure(figsize=(15, 5))
# ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
# ax.tick_params(axis='x', rotation=90)
# plt.tight_layout()
# plt.show()
# print(importancia_features)

# Analisando a importância das features percebemos:
#    - A importância da localização, quantidade de quartos e número de comodidades (tv, ar condicionado, etc...) para o preço
#    - Outras colunas como banheiros, pessoas extras e acomodações também possuem grande influência
#    - Features com menor importância serão removidas tais como: is_business_travel_ready, room_type_Hotel room, property_type_Hostel, property_type_Guest suite, cancellation_policy_strict, property_type_Guesthouse, property_type_Bed and breakfast, room_type_Shared room, property_type_Loft, property_type_Serviced apartment, property_type_Other
lista_remover = [
    'is_business_travel_ready', 'room_type_Hotel room', 'property_type_Hostel',
    'property_type_Guest suite', 'cancellation_policy_strict', 'property_type_Guesthouse',
    'property_type_Bed and breakfast', 'room_type_Shared room', 'property_type_Loft',
    'property_type_Serviced apartment', 'property_type_Other'
]

# Remoção colunas
for coluna in lista_remover:
    base_airbnb_cod = base_airbnb_cod.drop(coluna, axis=1)

y = base_airbnb_cod['price']
x = base_airbnb_cod.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)
modelo_ExtraTrees.fit(x_train, y_train)
previsao = modelo_ExtraTrees.predict(x_test)
print(avaliar_modelo('modelo_ExtraTrees', y_test, previsao))

importancia_features = pd.DataFrame(modelo_ExtraTrees.feature_importances_, x_train.columns)
importancia_features = importancia_features.sort_values(by=0, ascending=False)

# Exibir em gráfico:
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=importancia_features.index, y=importancia_features[0])
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()
print(importancia_features)
# Após a remoção dessas colunas tornamos o modelo bem mais simples sem alterar tanto o seu poder de previsão

### Deploy:
# Passo 1 -> Criar um arquivo do modelo (joblib)
# Passo 2 -> Escolher a forma de deploy:
# . Arquivo executável + tkinter
# . Deploy em um microsite (Flask)
# . Deploy apenas para o uso direto (streamlit)
# Passo 3 -> Outro arquivo python
# Passo 4 -> Importar streamlit e criar código
# Passo 5 -> Deploy feito

x['price'] = y
x.to_csv('dados.csv')

#joblib.dump(modelo_ExtraTrees, 'modelo.joblib')
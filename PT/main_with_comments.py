# Bibliotecas
import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate  # -> print(tabulate(df.head(2), headers=df.columns, tablefmt='pretty'))
import plotly.express as px

### README:
### 1 - Entendimento do Desafio que você quer resolver
### 2 - Entendimento da Empresa/Área

### 3 - Extração/Obtenção de Dados
caminho_bases = pathlib.Path('../dataset')

base_airbnb = pd.DataFrame()

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

# iterdir: Lista de arquivos dentro do caminho
for arquivo in caminho_bases.iterdir():
    # Adicionando a coluna mês e ano
    nome_mes = arquivo.name[:3]
    numero_mes = meses[nome_mes]

    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))

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

# Verificar se duas colunas são iguais:
print((base_airbnb['host_listings_count'] == base_airbnb['host_total_listings_count']).value_counts())


# Após a verificação no excel, escolhemos as colunas significativas para a nossa análise:
colunas = ['host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count', 'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'ano', 'mes']
base_airbnb = base_airbnb.loc[:, colunas]

# Tratar valores None
# Verificar quantidade de linhas vazias
print(base_airbnb.isnull().sum())

# Excluir as colunas: reviews, tempo de resposta, security deposit e taxa de limpeza
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() >= 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)

# Excluir as linhas onde temos poucos valores None:
base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())

## Verificar o tipo de dados de cada coluna:
print('-'*60)
print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])
# Tratando 'price'
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)  # float32 é mais ágil, em projetos assim é interessante
# Tratando 'extra_people'
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)  # float32 é mais ágil, em projetos assim é interessante

### 5 - Análise exploratória:
##. Ver a correlação entre as features(Caso tenha colunas muito correlacionadas, poderíamos excluir uma delas)
plt.figure(figsize=(15, 10))  # Ajustando o tamanho da figura

# Criando o heatmap
heatmap = sns.heatmap(base_airbnb.corr(numeric_only=True), annot=True, cmap='Greens', fmt='.2f', annot_kws={"size": 15})

# Rotacionar os rótulos do eixo x em 45 graus e ajustar a posição
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)

# Adicionando mais espaço em branco
plt.subplots_adjust(left=0.2)
plt.subplots_adjust(bottom=0.2)

plt.show()

##. Excluir outliers(Q1 - 1.5*Amplitude e Q3 + 1.5*Amplitude)(Amplitude = Q3 - Q1)
def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    limSup = q3 + 1.5 * amplitude
    limInf = q1 - 1.5 * amplitude
    return limInf, limSup


def excluirOutliers(df, nome_coluna):
    # Exclusão dos outliers
    qtde_linhas = df.shape[0]  # Indice 0 -> Linhas 1 -> Colunas
    limInf, limSup = limites(df[nome_coluna])
    df = df.loc[(df[nome_coluna] >= limInf) & (df[nome_coluna] <= limSup), :]
    # Conta quantas linhas foram excluídas
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

    # Histograma sem limites em x
    sns.histplot(data=base, x=coluna, element='bars', kde=True, binwidth=50, ax=ax1)
    ax1.set_title('Com outliers')

    # Histograma com limites em x
    lim_inf, lim_sup = limites(base[coluna])
    ax2.set_xlim(lim_inf, lim_sup)
    sns.histplot(data=base, x=coluna, element='bars', kde=True, binwidth=50, ax=ax2)
    ax2.set_title('Sem outliers')

    fig.suptitle('Com outliers vs Sem outliers', fontsize=16)
    plt.show()


def barra(base, coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    # Ensure coluna is a pandas Series object
    if isinstance(coluna, str):
        coluna = base[coluna]  # Assuming 'coluna' is a column name in the DataFrame 'base'

    # Gráfico de barras sem limites em x (com outliers)
    sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts(), ax=ax1)
    ax1.set_title('Com outliers')

    # Gráfico de barras com limites em x (sem outliers)
    sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts(), ax=ax2)
    ax2.set_xlim(limites(coluna))  # Assuming you have a function named 'limites' defined elsewhere
    ax2.set_title('Sem outliers')

    fig.suptitle('Com outliers vs Sem outliers', fontsize=16)
    plt.show()


def countplot(base, coluna):
    plt.figure(figsize=(15, 5))
    grafico = sns.countplot(x=coluna, data=base)
    grafico.tick_params(axis='x', rotation=90)
    plt.tight_layout()  # Ajusta automaticamente o layout para evitar cortes
    plt.show()

# Análise coluna price(contínuo):
boxPlot(base_airbnb['price'])
histograma(base_airbnb, 'price')
# Como estamos construindo um modelo para imóveis comuns, valores acima do limite superior são considerados de luxo, fugindo do objetivo principal.
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'price')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna extra_people(contínuo):
boxPlot(base_airbnb['extra_people'])
histograma(base_airbnb, 'extra_people')
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'extra_people')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna host_listings_count(discreto):
boxPlot(base_airbnb['host_listings_count'])
barra(base_airbnb, base_airbnb['host_listings_count'])
# Podemos excluir os outliers, pois hosts com mais de 6 imóveis no airbnb não é o público alvo do objetivo(imobiliárias, etc).
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'host_listings_count')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna accommodates(discreto):
boxPlot(base_airbnb['accommodates'])
barra(base_airbnb, base_airbnb['accommodates'])
# Casas com 9 acomodações serão excluídas, não é o objetivo do projeto
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'accommodates')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna bathrooms(discreto):
boxPlot(base_airbnb['bathrooms'])
# A função de gráfico de barra estava dando erro para esse caso, logo iremos tratar especificamente:
plt.figure(figsize=(15, 5))
sns.barplot(x=base_airbnb['bathrooms'].value_counts().index, y=base_airbnb['bathrooms'].value_counts())
plt.show()
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'bathrooms')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna bedrooms(discreto):
boxPlot(base_airbnb['bedrooms'])
barra(base_airbnb, base_airbnb['bedrooms'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'bedrooms')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna beds(discreto):
boxPlot(base_airbnb['beds'])
barra(base_airbnb, base_airbnb['beds'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'beds')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna guests_included(discreto):
sns.barplot(x=base_airbnb['guests_included'].value_counts().index, y=base_airbnb['guests_included'].value_counts())
# Não parece uma boa métrica, vamos remover a coluna da análise
base_airbnb = base_airbnb.drop('guests_included', axis=1)

# Análise coluna minimum_nights(discreto):
boxPlot(base_airbnb['minimum_nights'])
barra(base_airbnb, base_airbnb['minimum_nights'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'minimum_nights')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Análise coluna maximum_nights(discreto):
boxPlot(base_airbnb['maximum_nights'])
barra(base_airbnb, base_airbnb['maximum_nights'])
# Parece haver o mal preenchimento desse campo por parte dos usuários, sendo assim, devemos remover essa coluna para não atrapalhar a análise
base_airbnb = base_airbnb.drop('maximum_nights', axis=1)

# Análise coluna number_of_reviews(discreto):
boxPlot(base_airbnb['number_of_reviews'])
barra(base_airbnb, base_airbnb['number_of_reviews'])
# Poderíamos tirar ou não essa coluna da análise, a fim de desempenho computacional, irei tirar:
base_airbnb = base_airbnb.drop('number_of_reviews', axis=1)

# Análise coluna property_type(categórica):
print(base_airbnb['property_type'].value_counts())
countplot(base_airbnb, 'property_type')
tabela_tipos_casa = base_airbnb['property_type'].value_counts()
colunas_agrupar = []
for tipo in tabela_tipos_casa.index:
    if tabela_tipos_casa[tipo] < 2000:
        colunas_agrupar.append(tipo)
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['property_type'] == tipo, 'property_type'] = 'Other'
print(base_airbnb['property_type'].value_counts())

# Análise coluna room_type (categórica):
print(base_airbnb['room_type'].value_counts())
countplot(base_airbnb, 'room_type')

# Análise coluna bed_type(categórica):
print(base_airbnb['bed_type'].value_counts())
countplot(base_airbnb, 'bed_type')
# Agrupando tipos de cama que são minorias
colunas_agrupar = []
tabela_tipos_cama = base_airbnb['bed_type'].value_counts()
for tipo in tabela_tipos_cama.index:
    if tabela_tipos_cama[tipo] < 10000:
        colunas_agrupar.append(tipo)
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['bed_type'] == tipo, 'bed_type'] = 'Other'
print(base_airbnb['bed_type'].value_counts())

# Análise coluna cancellation_policy(categórica):
print(base_airbnb['cancellation_policy'].value_counts())
countplot(base_airbnb, 'cancellation_policy')
# Agrupando tipos de políticas de cancelamento muito parecidas
colunas_agrupar = []
tabela_tipos_politica = base_airbnb['cancellation_policy'].value_counts()
for tipo in tabela_tipos_politica.index:
    if tabela_tipos_politica[tipo] < 10000:
        colunas_agrupar.append(tipo)
for tipo in colunas_agrupar:
    base_airbnb.loc[base_airbnb['cancellation_policy'] == tipo, 'cancellation_policy'] = 'strict'
print(base_airbnb['cancellation_policy'].value_counts())

# Análise coluna amenities(categórica):
print(base_airbnb['amenities'].value_counts())
# Iremos avaliar com base na quantidade de comodidades, e não nas comodidades em si, devido à grande quantidade de tipos e maneiras de escrita de uma mesma comodidade
base_airbnb['n_amenities'] = base_airbnb['amenities'].str.split(',').apply(len)
# Remover a coluna amenities
base_airbnb = base_airbnb.drop('amenities', axis=1)

# Análise nova coluna n_amenities
boxPlot(base_airbnb['n_amenities'])
barra(base_airbnb, base_airbnb['n_amenities'])
base_airbnb, linhas_removidas, nome_coluna = excluirOutliers(base_airbnb, 'n_amenities')
print(f'{nome_coluna} - Foram excluídas {linhas_removidas} linhas de Outliers')

# Criação do mapa de densidade para visualização das áreas com maiores preços
amostra = base_airbnb.sample(n=50000)
centro_mapa = {'lat': amostra.latitude.mean(), 'lon': amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude', z='price', radius=2.5,
            center=centro_mapa, zoom=10,
            mapbox_style='stamen-terrain')
mapa.update_layout(mapbox_style="open-street-map")
mapa.show()

##. Encoding(Ajusta as features para facilitar o trabalho do modelo: Categóricas, Boolean, etc)
#Boolean -> V/F = 1/0
colunas_vf = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
base_airbnb_cod = base_airbnb.copy()
for coluna in colunas_vf:
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 't', coluna] = 1
    base_airbnb_cod.loc[base_airbnb_cod[coluna] == 'f', coluna] = 0

#Categóricas -> OneHotEncoding ou DummyVariables
colunas_cat = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
base_airbnb_cod = pd.get_dummies(data=base_airbnb_cod, columns=colunas_cat)
##. Confirmar que todas as features que temos fazem realmente sentido para o nosso modelo



#Bibliotecas
import pandas as pd
from tabulate import tabulate #-> print(tabulate(df.head(2), headers=df.columns, tablefmt='pretty'))
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

###README:
###1 - Entendimento do Desafio que você quer resolver
###2 - Entendimento da Empresa/Área

###3 - Extração/Obtenção de Dados
caminho_bases = pathlib.Path('../dataset')

base_airbnb = pd.DataFrame()

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

#iterdir: Lista de arquivos dentro do caminho
for arquivo in caminho_bases.iterdir():
    #Adicionando a coluna mês e ano
    nome_mes = arquivo.name[:3]
    numero_mes = meses[nome_mes]

    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))

    df = pd.read_csv(caminho_bases / arquivo.name)

    df['ano'] = ano
    df['mes'] = numero_mes
    #Unindo todos dfs em um grande df
    base_airbnb = base_airbnb._append(df)


### 4 - Ajustes de Dados:
#Para fazer a limpeza de dados é interessante utilizar o excel
#Lista o nome de cada coluna
print(list(base_airbnb.columns))

#Cria um arquivo excel com as 1000 primeiras linhas
base_airbnb.head(1000).to_csv('primeiros_registros.csv', sep=';')

#Tirar colunas desnecessárias:
# 1 - Ids, Links e informações não relevantes para o modelo
# 2 - Colunas repetidas EX: Data vs Ano/Mês
# 3 - Colunas preenchidas com texto livre -> Não serve para a análise
# 4 - Colunas vazias, ou em que quase todos os valores são iguais

print(base_airbnb[['experiences_offered']].value_counts())

#Verificar se duas colunas são iguais:
print((base_airbnb['host_listings_count'] == base_airbnb['host_total_listings_count']).value_counts())


#Após a verificação no excel, escolhemos as colunas significativas para a nossa análise:
colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','ano','mes']
base_airbnb = base_airbnb.loc[:, colunas]

#Tratar valores None
#Verificar quantidade de linhas vazias
print(base_airbnb.isnull().sum())

#Excluir as colunas: reviews, tempo de resposta, security deposit e taxa de limpeza
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull().sum() >= 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)

#Excluir as linhas onde temos poucos valores None:
base_airbnb = base_airbnb.dropna()
print(base_airbnb.isnull().sum())

##Verificar o tipo de dados de cada coluna:
print('-'*60)
print(base_airbnb.dtypes)
print('-'*60)
print(base_airbnb.iloc[0])
#Tratando 'price'
base_airbnb['price'] = base_airbnb['price'].str.replace('$', '')
base_airbnb['price'] = base_airbnb['price'].str.replace(',', '')
base_airbnb['price'] = base_airbnb['price'].astype(np.float32, copy=False)#float32 é mais ágil, em projetos assim é interessante
#Tratando 'extra_people'
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace('$', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].str.replace(',', '')
base_airbnb['extra_people'] = base_airbnb['extra_people'].astype(np.float32, copy=False)#float32 é mais ágil, em projetos assim é interessante

### 5 - Análise exploratória:
##. Ver a correlação entre as features(Caso tenha colunas muito correlacionadas, poderiamos excluir uma delas)
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
    return limSup, limInf

def boxPlot(coluna):
    fig, (ax1, ax2) = plt.subplot(1, 2)
    fig.set_size_inches(15, 5)

    sns.boxplot(x=coluna, ax=ax1)

    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)

    plt.show()

def histograma(coluna):
    plt.figure(figsize=(15, 5))
    sns.displot(coluna, hist=True)

    plt.show()

##. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo
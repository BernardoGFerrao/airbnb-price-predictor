#Bibliotecas
import pandas as pd
from tabulate import tabulate #-> print(tabulate(df.head(2), headers=df.columns, tablefmt='pretty'))
import pathlib

###README:
###1 - Entendimento do Desafio que você quer resolver
###2 - Entendimento da Empresa/Área

###3 - Extração/Obtenção de Dados
caminho_bases = pathlib.Path('dataset')

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


##Limpeza dos dados:
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

#Excluir os reviews, tempo de resposta, security deposit e taxa de limpeza
for coluna in base_airbnb:
    if base_airbnb[coluna].isnull.sum() >= 300000:
        base_airbnb = base_airbnb.drop(coluna, axis=1)
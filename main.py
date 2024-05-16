##README:
##1 - Entendimento do Desafio que você quer resolver
##2 - Entendimento da Empresa/Área

##3 - Extração/Obtenção de Dados
import pandas as pd
from tabulate import tabulate #-> print(tabulate(df.head(2), headers=df.columns, tablefmt='pretty'))
import pathlib

caminho_bases = pathlib.Path('dataset')

base_airbnb = pd.DataFrame()

meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

#iterdir: Lista de arquivos dentro do caminho
for arquivo in caminho_bases.iterdir():
    nome_mes = arquivo.name[:3]
    numero_mes = meses[nome_mes]

    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv', ''))

    df = pd.read_csv(caminho_bases / arquivo.name)

    df['ano'] = ano
    df['mes'] = numero_mes

    base_airbnb = base_airbnb._append(df)

print(tabulate(base_airbnb.head(2), headers=base_airbnb.columns, tablefmt='pretty'))





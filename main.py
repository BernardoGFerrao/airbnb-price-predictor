##README:
##1 - Entendimento do Desafio que você quer resolver
##2 - Entendimento da Empresa/Área

##3 - Extração/Obtenção de Dados
import pandas as pd
from tabulate import tabulate #-> print(tabulate(df.head(2), headers=df.columns, tablefmt='pretty'))
import pathlib

caminho_bases = pathlib.Path('dataset')

base_airbnb = pd.DataFrame()

#iterdir: Lista de arquivos dentro do caminho
for arquivo in caminho_bases.iterdir():
    df = pd.read_csv(caminho_bases / arquivo.name)
    base_airbnb = base_airbnb._append(df)

print(tabulate(base_airbnb.head(2), headers=base_airbnb.columns, tablefmt='pretty'))





import pandas as pd
import streamlit as st
import joblib

##Variáveis:
#Numeric
x_numeric = {'latitude': 0,
             'longitude': 0,
             'accommodates': 0,
             'bathrooms': 0,
             'bedrooms': 0,
             'beds': 0,
             'extra_people': 0,
             'minimum_nights': 0,
             'ano': 0, 'mes': 0,
             'n_amenities': 0,
             'host_listings_count': 0}
#Boolean
x_boolean = {'host_is_superhost': 0,
             'instant_bookable': 0}
#Categorical
x_categorical = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Other', 'Serviced apartment'],
                 'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
                 'cancellation_policy': ['flexible', 'moderate', 'strict', 'strict_14_with_grace_period']}

dict = {}
for item in x_categorical:
    for valor in x_categorical[item]:
         dict[f'{item}_{valor}'] = 0

#Botões/Buttons
for item in x_numeric:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format="%.5f")
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
    else:
        valor = st.number_input(f'{item}', value=0)
    x_numeric[item] = valor

for item in x_boolean:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    if valor == "Sim":
        x_boolean[item] = 1
    else:
        x_boolean[item] = 0

for item in x_categorical:
    valor = st.selectbox(f'{item}', x_categorical[item])
    dict[f'{item}_{valor}'] = 1

botao = st.button('Prever valor do imóvel')

if botao:
    dict.update(x_numeric)
    dict.update(x_boolean)
    valores_x = pd.DataFrame(dict, index=[0])

    dados = pd.read_csv('PT/dados.csv')
    colunas = list(dados.columns)[1:-1]
    valores_x = valores_x[colunas]

    ##Criar o nosso arquivo/Create our file
    modelo = joblib.load('PT/modelo_compression_level_1.joblib')
    preco = modelo.predict(valores_x)
    st.write(preco[0])

#cd C:\Github\airbnb-price-predictor\PT
#streamlit run C:\Github\airbnb-price-predictor\PT\deploy.py

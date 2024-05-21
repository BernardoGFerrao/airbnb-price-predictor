import pandas as pd
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib

path_to_datasets = pathlib.Path('../dataset')
airbnb_data = pd.DataFrame()
months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

for file in path_to_datasets.iterdir():
    month_name = file.name[:3]
    month_number = months[month_name]
    year = int(file.name[-8:].replace('.csv', ''))
    df = pd.read_csv(path_to_datasets / file.name)
    df['year'] = year
    df['month'] = month_number
    airbnb_data = airbnb_data._append(df)

print(list(airbnb_data.columns))

airbnb_data.head(1000).to_csv('first_records.csv', sep=';')

print(airbnb_data[['experiences_offered']].value_counts())
print((airbnb_data['host_listings_count'] == airbnb_data['host_total_listings_count']).value_counts())

columns = [
    'host_response_time', 'host_response_rate', 'host_is_superhost', 'host_listings_count',
    'latitude', 'longitude', 'property_type', 'room_type', 'accommodates', 'bathrooms',
    'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee',
    'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews',
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value', 'instant_bookable', 'is_business_travel_ready',
    'cancellation_policy', 'year', 'month'
]
airbnb_data = airbnb_data.loc[:, columns]

print(airbnb_data.isnull().sum())

for column in airbnb_data:
    if airbnb_data[column].isnull().sum() >= 300000:
        airbnb_data = airbnb_data.drop(column, axis=1)

airbnb_data = airbnb_data.dropna()
print(airbnb_data.isnull().sum())

print('-'*60)
print(airbnb_data.dtypes)
print('-'*60)
print(airbnb_data.iloc[0])

airbnb_data['price'] = airbnb_data['price'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)
airbnb_data['extra_people'] = airbnb_data['extra_people'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)

plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(airbnb_data.corr(numeric_only=True), annot=True, cmap='Greens', fmt='.2f', annot_kws={"size": 15})
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.show()

def limits(column):
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    range_ = q3 - q1
    upper_limit = q3 + 1.5 * range_
    lower_limit = q1 - 1.5 * range_
    return lower_limit, upper_limit

def exclude_outliers(df, column_name):
    num_rows = df.shape[0]
    lower_limit, upper_limit = limits(df[column_name])
    df = df.loc[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit), :]
    rows_removed = num_rows - df.shape[0]
    return df, rows_removed, column_name

def box_plot(column):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.boxplot(x=column, ax=ax1)
    ax1.set_title('With outliers')
    lower_limit, upper_limit = limits(column)
    ax2.set_xlim(lower_limit, upper_limit)
    sns.boxplot(x=column, ax=ax2)
    ax2.set_title('Without outliers')
    fig.suptitle('With outliers vs Without outliers', fontsize=16)
    plt.show()

def histogram(base, column):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    sns.histplot(data=base, x=column, element='bars', kde=True, binwidth=50, ax=ax1)
    ax1.set_title('With outliers')
    lower_limit, upper_limit = limits(base[column])
    ax2.set_xlim(lower_limit, upper_limit)
    sns.histplot(data=base, x=column, element='bars', kde=True, binwidth=50, ax=ax2)
    ax2.set_title('Without outliers')
    fig.suptitle('With outliers vs Without outliers', fontsize=16)
    plt.show()

def bar_chart(base, column):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    if isinstance(column, str):
        column = base[column]
    sns.barplot(x=column.value_counts().index, y=column.value_counts(), ax=ax1)
    ax1.set_title('With outliers')
    sns.barplot(x=column.value_counts().index, y=column.value_counts(), ax=ax2)
    ax2.set_xlim(limits(column))
    ax2.set_title('Without outliers')
    fig.suptitle('With outliers vs Without outliers', fontsize=16)
    plt.show()

def count_plot(base, column):
    plt.figure(figsize=(15, 5))
    plot = sns.countplot(x=column, data=base)
    plot.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.show()

box_plot(airbnb_data['price'])
histogram(airbnb_data, 'price')
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'price')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

box_plot(airbnb_data['extra_people'])
histogram(airbnb_data, 'extra_people')
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'extra_people')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

box_plot(airbnb_data['host_listings_count'])
bar_chart(airbnb_data, airbnb_data['host_listings_count'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'host_listings_count')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

box_plot(airbnb_data['accommodates'])
bar_chart(airbnb_data, airbnb_data['accommodates'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'accommodates')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

box_plot(airbnb_data['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=airbnb_data['bathrooms'].value_counts().index, y=airbnb_data['bathrooms'].value_counts())
plt.show()
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'bathrooms')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

box_plot(airbnb_data['bedrooms'])
bar_chart(airbnb_data, airbnb_data['bedrooms'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'bedrooms')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

box_plot(airbnb_data['beds'])
bar_chart(airbnb_data, airbnb_data['beds'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'beds')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

sns.barplot(x=airbnb_data['guests_included'].value_counts().index, y=airbnb_data['guests_included'].value_counts())
airbnb_data = airbnb_data.drop('guests_included', axis=1)

box_plot(airbnb_data['minimum_nights'])
bar_chart(airbnb_data, airbnb_data['minimum_nights'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'minimum_nights')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

box_plot(airbnb_data['maximum_nights'])
bar_chart(airbnb_data, airbnb_data['maximum_nights'])
airbnb_data = airbnb_data.drop('maximum_nights', axis=1)

box_plot(airbnb_data['number_of_reviews'])
bar_chart(airbnb_data, airbnb_data['number_of_reviews'])
airbnb_data = airbnb_data.drop('number_of_reviews', axis=1)

print(airbnb_data['property_type'].value_counts())
count_plot(airbnb_data, 'property_type')
property_type_counts = airbnb_data['property_type'].value_counts()
columns_to_group = [ptype for ptype in property_type_counts.index if property_type_counts[ptype] < 2000]
for ptype in columns_to_group:
    airbnb_data.loc[airbnb_data['property_type'] == ptype, 'property_type'] = 'Other'
print(airbnb_data['property_type'].value_counts())

print(airbnb_data['room_type'].value_counts())
count_plot(airbnb_data, 'room_type')

print(airbnb_data['bed_type'].value_counts())
count_plot(airbnb_data, 'bed_type')
columns_to_group = [btype for btype in airbnb_data['bed_type'].value_counts().index if airbnb_data['bed_type'].value_counts()[btype] < 10000]
for btype in columns_to_group:
    airbnb_data.loc[airbnb_data['bed_type'] == btype, 'bed_type'] = 'Other'
print(airbnb_data['bed_type'].value_counts())

print(airbnb_data['cancellation_policy'].value_counts())
count_plot(airbnb_data, 'cancellation_policy')
columns_to_group = [ctype for ctype in airbnb_data['cancellation_policy'].value_counts().index if airbnb_data['cancellation_policy'].value_counts()[ctype] < 10000]
for ctype in columns_to_group:
    airbnb_data.loc[airbnb_data['cancellation_policy'] == ctype, 'cancellation_policy'] = 'strict'
print(airbnb_data['cancellation_policy'].value_counts())

print(airbnb_data['amenities'].value_counts())
airbnb_data['n_amenities'] = airbnb_data['amenities'].str.split(',').apply(len)
airbnb_data = airbnb_data.drop('amenities', axis=1)

box_plot(airbnb_data['n_amenities'])
bar_chart(airbnb_data, airbnb_data['n_amenities'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'n_amenities')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

sample = airbnb_data.sample(n=50000)
map_center = {'lat': sample.latitude.mean(), 'lon': sample.longitude.mean()}
map_fig = px.density_mapbox(sample, lat='latitude', lon='longitude', z='price', radius=2.5, center=map_center, zoom=10, mapbox_style='stamen-terrain')
map_fig.update_layout(mapbox_style="open-street-map")
map_fig.show()

boolean_columns = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
airbnb_data_encoded = airbnb_data.copy()
for column in boolean_columns:
    airbnb_data_encoded.loc[airbnb_data_encoded[column] == 't', column] = 1
    airbnb_data_encoded.loc[airbnb_data_encoded[column] == 'f', column] = 0

categorical_columns = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
airbnb_data_encoded = pd.get_dummies(data=airbnb_data_encoded, columns=categorical_columns)

def evaluate_model(model_name, y_test, prediction):
    r2 = r2_score(y_test, prediction)
    rsme = np.sqrt(mean_squared_error(y_test, prediction))
    return f"Model {model_name}\n- R²: {r2:.2%}\n- RSME: {rsme:.2f}"

y = airbnb_data_encoded['price']
x = airbnb_data_encoded.drop('price', axis=1)

model_RandomForest = RandomForestRegressor()
model_LinearRegression = LinearRegression()
model_ExtraTrees = ExtraTreesRegressor()

models = {'RandomForest': model_RandomForest, 'LinearRegression': model_LinearRegression, 'ExtraTrees': model_ExtraTrees}

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

def choose_model(models):
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        print(evaluate_model(model_name, y_test, prediction))

print(model_ExtraTrees.feature_importances_)
print(x_train.columns)
feature_importance = pd.DataFrame(model_ExtraTrees.feature_importances_, x_train.columns)

feature_importance = feature_importance.sort_values(by=0, ascending=False)

plt.figure(figsize=(15, 5))
ax = sns.barplot(x=feature_importance.index, y=feature_importance[0])
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()
print(feature_importance)

columns_to_remove = [
    'is_business_travel_ready', 'room_type_Hotel room', 'property_type_Hostel',
    'property_type_Guest suite', 'cancellation_policy_strict', 'property_type_Guesthouse',
    'property_type_Bed and breakfast', 'room_type_Shared room', 'property_type_Loft',
    'property_type_Serviced apartment', 'property_type_Other'
]

for column in columns_to_remove:
    airbnb_data_encoded = airbnb_data_encoded.drop(column, axis=1)

y = airbnb_data_encoded['price']
x = airbnb_data_encoded.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)
model_ExtraTrees.fit(x_train, y_train)
prediction = model_ExtraTrees.predict(x_test)
print(evaluate_model('model_ExtraTrees', y_test, prediction))

feature_importance = pd.DataFrame(model_ExtraTrees.feature_importances_, x_train.columns)
feature_importance = feature_importance.sort_values(by=0, ascending=False)

plt.figure(figsize=(15, 5))
ax = sns.barplot(x=feature_importance.index, y=feature_importance[0])
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()
print(feature_importance)

x['price'] = y
x.to_csv('data.csv')

#joblib.dump(modelo_ExtraTrees, 'modelo_compression_level_1.joblib', compress=1)
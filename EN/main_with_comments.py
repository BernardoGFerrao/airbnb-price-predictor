# Libraries
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
### 1 - Understanding the challenge you want to solve
### 2 - Understanding the Company/Area

### 3 - Data Extraction/Acquisition
path_to_datasets = pathlib.Path('../dataset')
airbnb_data = pd.DataFrame()
months = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

# Iterate over the files in the path
for file in path_to_datasets.iterdir():
    # Adding the month and year columns
    month_name = file.name[:3]
    month_number = months[month_name]
    year = int(file.name[-8:].replace('.csv', ''))
    df = pd.read_csv(path_to_datasets / file.name)
    df['year'] = year
    df['month'] = month_number
    # Merging all dfs into one large df
    airbnb_data = airbnb_data._append(df)

### 4 - Data Adjustments:
# To clean data it is useful to use Excel
# List the name of each column
print(list(airbnb_data.columns))

# Create an Excel file with the first 1000 rows
airbnb_data.head(1000).to_csv('first_records.csv', sep=';')

# Remove unnecessary columns:
# 1 - Ids, Links and irrelevant information for the model
# 2 - Duplicate columns EX: Date vs Year/Month
# 3 - Columns filled with free text -> Not useful for analysis
# 4 - Empty columns, or columns where almost all values are the same
print(airbnb_data[['experiences_offered']].value_counts())
print((airbnb_data['host_listings_count'] == airbnb_data['host_total_listings_count']).value_counts())

# After verification in Excel, we chose the significant columns for our analysis:
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

# Handle None values
print(airbnb_data.isnull().sum())

# Remove columns: reviews, response time, security deposit, and cleaning fee
for column in airbnb_data:
    if airbnb_data[column].isnull().sum() >= 300000:
        airbnb_data = airbnb_data.drop(column, axis=1)

# Remove rows with few None values:
airbnb_data = airbnb_data.dropna()
print(airbnb_data.isnull().sum())

# Check the data type of each column:
print('-'*60)
print(airbnb_data.dtypes)
print('-'*60)
print(airbnb_data.iloc[0])

# Handling 'price'
airbnb_data['price'] = airbnb_data['price'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)

# Handling 'extra_people'
airbnb_data['extra_people'] = airbnb_data['extra_people'].str.replace('$', '').str.replace(',', '').astype(np.float32, copy=False)

### 5 - Exploratory Analysis:
# Check the correlation between features
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(airbnb_data.corr(numeric_only=True), annot=True, cmap='Greens', fmt='.2f', annot_kws={"size": 15})
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.show()

# Exclude outliers
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

# Analyzing column price (continuous)
box_plot(airbnb_data['price'])
histogram(airbnb_data, 'price')
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'price')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column extra_people (continuous)
box_plot(airbnb_data['extra_people'])
histogram(airbnb_data, 'extra_people')
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'extra_people')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column host_listings_count (discrete)
box_plot(airbnb_data['host_listings_count'])
bar_chart(airbnb_data, airbnb_data['host_listings_count'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'host_listings_count')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column accommodates (discrete)
box_plot(airbnb_data['accommodates'])
bar_chart(airbnb_data, airbnb_data['accommodates'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'accommodates')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column bathrooms (discrete)
box_plot(airbnb_data['bathrooms'])
plt.figure(figsize=(15, 5))
sns.barplot(x=airbnb_data['bathrooms'].value_counts().index, y=airbnb_data['bathrooms'].value_counts())
plt.show()
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'bathrooms')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column bedrooms (discrete)
box_plot(airbnb_data['bedrooms'])
bar_chart(airbnb_data, airbnb_data['bedrooms'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'bedrooms')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column beds (discrete)
box_plot(airbnb_data['beds'])
bar_chart(airbnb_data, airbnb_data['beds'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'beds')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column guests_included (discrete)
sns.barplot(x=airbnb_data['guests_included'].value_counts().index, y=airbnb_data['guests_included'].value_counts())
airbnb_data = airbnb_data.drop('guests_included', axis=1)

# Analyzing column minimum_nights (discrete)
box_plot(airbnb_data['minimum_nights'])
bar_chart(airbnb_data, airbnb_data['minimum_nights'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'minimum_nights')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Analyzing column maximum_nights (discrete)
box_plot(airbnb_data['maximum_nights'])
bar_chart(airbnb_data, airbnb_data['maximum_nights'])
airbnb_data = airbnb_data.drop('maximum_nights', axis=1)

# Analyzing column number_of_reviews (discrete)
box_plot(airbnb_data['number_of_reviews'])
bar_chart(airbnb_data, airbnb_data['number_of_reviews'])
airbnb_data = airbnb_data.drop('number_of_reviews', axis=1)

# Analyzing column property_type (categorical)
print(airbnb_data['property_type'].value_counts())
count_plot(airbnb_data, 'property_type')
property_type_counts = airbnb_data['property_type'].value_counts()
columns_to_group = [ptype for ptype in property_type_counts.index if property_type_counts[ptype] < 2000]
for ptype in columns_to_group:
    airbnb_data.loc[airbnb_data['property_type'] == ptype, 'property_type'] = 'Other'
print(airbnb_data['property_type'].value_counts())

# Analyzing column room_type (categorical)
print(airbnb_data['room_type'].value_counts())
count_plot(airbnb_data, 'room_type')

# Analyzing column bed_type (categorical)
print(airbnb_data['bed_type'].value_counts())
count_plot(airbnb_data, 'bed_type')
columns_to_group = [btype for btype in airbnb_data['bed_type'].value_counts().index if airbnb_data['bed_type'].value_counts()[btype] < 10000]
for btype in columns_to_group:
    airbnb_data.loc[airbnb_data['bed_type'] == btype, 'bed_type'] = 'Other'
print(airbnb_data['bed_type'].value_counts())

# Analyzing column cancellation_policy (categorical)
print(airbnb_data['cancellation_policy'].value_counts())
count_plot(airbnb_data, 'cancellation_policy')
columns_to_group = [ctype for ctype in airbnb_data['cancellation_policy'].value_counts().index if airbnb_data['cancellation_policy'].value_counts()[ctype] < 10000]
for ctype in columns_to_group:
    airbnb_data.loc[airbnb_data['cancellation_policy'] == ctype, 'cancellation_policy'] = 'strict'
print(airbnb_data['cancellation_policy'].value_counts())

# Analyzing column amenities (categorical)
print(airbnb_data['amenities'].value_counts())
airbnb_data['n_amenities'] = airbnb_data['amenities'].str.split(',').apply(len)
airbnb_data = airbnb_data.drop('amenities', axis=1)

# Analyzing new column n_amenities
box_plot(airbnb_data['n_amenities'])
bar_chart(airbnb_data, airbnb_data['n_amenities'])
airbnb_data, rows_removed, column_name = exclude_outliers(airbnb_data, 'n_amenities')
print(f'{column_name} - {rows_removed} rows of outliers were removed')

# Creating the density map to visualize areas with higher prices
sample = airbnb_data.sample(n=50000)
map_center = {'lat': sample.latitude.mean(), 'lon': sample.longitude.mean()}
map_fig = px.density_mapbox(sample, lat='latitude', lon='longitude', z='price', radius=2.5, center=map_center, zoom=10, mapbox_style='stamen-terrain')
map_fig.update_layout(mapbox_style="open-street-map")
map_fig.show()

### Encoding
# Boolean -> T/F = 1/0
boolean_columns = ['host_is_superhost', 'instant_bookable', 'is_business_travel_ready']
airbnb_data_encoded = airbnb_data.copy()
for column in boolean_columns:
    airbnb_data_encoded.loc[airbnb_data_encoded[column] == 't', column] = 1
    airbnb_data_encoded.loc[airbnb_data_encoded[column] == 'f', column] = 0

# Categorical -> OneHotEncoding or DummyVariables
categorical_columns = ['property_type', 'room_type', 'bed_type', 'cancellation_policy']
airbnb_data_encoded = pd.get_dummies(data=airbnb_data_encoded, columns=categorical_columns)

### Steps to build a prediction model:
# 1: Choose the type of machine learning: Classification vs Regression
#    - Classification: Categories (Separate between A, B, and C) (Ex: Disease diagnosis, Spam, ...)
#    - Regression: Specific value (Number) (Ex: Price, Speed, ...)
# 2: Define metrics to evaluate the model:
#    - R²
#        - From 0 to 1 -> The higher, the better
#        - Explanation: Measures "how much" the model can explain the values
#        - Ex: 92% means that the model can explain 92% of the data variance from the given information
#    - RSME (Root Mean Squared Error)
#        - Can be any value
#        - Measures "how much" the model errs
# 3: Define the regression models to use:
#    - Linear regression
#        - Draws the "best" line among the points minimizing errors
#    - Random forest
#        - Randomly chooses features and looks for the best place to split the data. Uses data samples with replacement. Good general choice, especially for smaller and clean data
#    - Extra Trees
#        - Randomly chooses where to split the data. Can use all data or samples without replacement. Faster and can handle very noisy data well
# 4: Train and test the model
#    - Split the dataset into 2 sets: Training and Testing
#    - 80% training - Used to learn (Has access to the features of the property (X) and the price (Y))
#    - 20% testing  - Used to test (Has access to the features of the property (X) and tries to price it (Y))
# 5: Compare the models and choose the best
#    - Calculate the two metrics for each model
#    - Choose 1 metric to be the main and use the other as a tiebreaker
#    - Besides that, we should consider the time and complexity
# 6: Analyze the best model more in-depth
#    - Understand the importance of each feature to see improvement opportunities
#    - If a feature/column is not used or is of little importance, we can remove it and see the result
#    - Remember to evaluate: Metrics, speed, and simplicity of the model
# 7: Make adjustments to the best model
#    - After removing the features, retrain the model and analyze the metrics
#    - If there are no significant differences in the metrics, maybe the difference in time or complexity is reasonable

# Defining metrics:
def evaluate_model(model_name, y_test, prediction):
    r2 = r2_score(y_test, prediction)
    rsme = np.sqrt(mean_squared_error(y_test, prediction))
    return f"Model {model_name}\n- R²: {r2:.2%}\n- RSME: {rsme:.2f}"

# Variable separation
y = airbnb_data_encoded['price']
x = airbnb_data_encoded.drop('price', axis=1)

# Split data into training and testing + model training
model_RandomForest = RandomForestRegressor()
model_LinearRegression = LinearRegression()
model_ExtraTrees = ExtraTreesRegressor()

models = {'RandomForest': model_RandomForest, 'LinearRegression': model_LinearRegression, 'ExtraTrees': model_ExtraTrees}

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

def choose_model(models):
    for model_name, model in models.items():
        # Train
        model.fit(x_train, y_train)
        # Test
        prediction = model.predict(x_test)
        print(evaluate_model(model_name, y_test, prediction))

# # choose_model(models)  # This line was commented out after choosing the model
#
# # After evaluating, the ExtraTrees model is the best model both in R² and RSME
# print(model_ExtraTrees.feature_importances_)
# print(x_train.columns)
# feature_importance = pd.DataFrame(model_ExtraTrees.feature_importances_, x_train.columns)
#
# # Sort:
# feature_importance = feature_importance.sort_values(by=0, ascending=False)
#
# # Display in chart:
# plt.figure(figsize=(15, 5))
# ax = sns.barplot(x=feature_importance.index, y=feature_importance[0])
# ax.tick_params(axis='x', rotation=90)
# plt.tight_layout()
# plt.show()
# print(feature_importance)

# Analyzing the importance of the features we realized:
#    - The importance of location, number of rooms, and number of amenities (TV, air conditioning, etc.) for the price
#    - Other columns such as bathrooms, extra people, and accommodations also have a significant influence
#    - Features with lower importance will be removed such as: is_business_travel_ready, room_type_Hotel room, property_type_Hostel, property_type_Guest suite, cancellation_policy_strict, property_type_Guesthouse, property_type_Bed and breakfast, room_type_Shared room, property_type_Loft, property_type_Serviced apartment, property_type_Other
columns_to_remove = [
    'is_business_travel_ready', 'room_type_Hotel room', 'property_type_Hostel',
    'property_type_Guest suite', 'cancellation_policy_strict', 'property_type_Guesthouse',
    'property_type_Bed and breakfast', 'room_type_Shared room', 'property_type_Loft',
    'property_type_Serviced apartment', 'property_type_Other'
]

# Removing columns
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

# Display in chart:
plt.figure(figsize=(15, 5))
ax = sns.barplot(x=feature_importance.index, y=feature_importance[0])
ax.tick_params(axis='x', rotation=90)
plt.tight_layout()
plt.show()
print(feature_importance)
# After removing these columns we made the model much simpler without significantly altering its predictive power

### Deploy:
# Step 1 -> Create a model file (joblib)
# Step 2 -> Choose the deployment method:
# . Executable file + tkinter
# . Deploy on a microsite (Flask)
# . Deploy for direct use (streamlit)
# Step 3 -> Another python file
# Step 4 -> Import streamlit and create code
# Step 5 -> Deployment done

x['price'] = y
x.to_csv('data.csv')

joblib.dump(model_ExtraTrees, 'modelo.joblib')

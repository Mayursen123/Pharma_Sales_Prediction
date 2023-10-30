import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

df = pd.read_csv('D:/pharma/salesdaily.csv')

df.head()

"""About Dataset
The dataset is built from the initial dataset consisted of 600000 transactional data collected in 6 years (period 2014-2019), indicating date and time of sale, pharmaceutical drug brand name and sold quantity, exported from Point-of-Sale system in the individual pharmacy. Selected group of drugs from the dataset (57 drugs) is classified to the following Anatomical Therapeutic Chemical (ATC) Classification System categories:

M01AB - Anti-inflammatory and antirheumatic products, non-steroids, Acetic acid derivatives and related substances
M01AE - Anti-inflammatory and antirheumatic products, non-steroids, Propionic acid derivatives
N02BA - Other analgesics and antipyretics, Salicylic acid and derivatives
N02BE/B - Other analgesics and antipyretics, Pyrazolones and Anilides
N05B - Psycholeptics drugs, Anxiolytic drugs
N05C - Psycholeptics drugs, Hypnotics and sedatives drugs
R03 - Drugs for obstructive airway diseases
R06 - Antihistamines for systemic use
Sales data are resampled to the hourly, daily, weekly and monthly periods. Data is already pre-processed, where processing included outlier detection and treatment and missing data imputation.

### EDA
"""

df.shape

df.isna().sum()

# Convert the 'datum' column to datetime
df['datum'] = pd.to_datetime(df['datum'])

import plotly.express as px

# Plot the quantity of M01AB against the datum using Plotly Express
fig = px.line(df, x='datum', y='M01AB', title='Quantity of M01AB over time')
fig.show()

### Monthly sales volume

df_m01ab = df[['M01AB','Year','Month']]
df_m01ab

df_m01ab = df_m01ab.groupby(['Year', 'Month']).sum().reset_index()
df_m01ab

# Plot the 'M01AB' values against month and year
fig = px.bar(df_m01ab, x='Month', y='M01AB', color='Year', barmode='group')

# Update the axis labels and title
fig.update_layout(
    xaxis_title='Month',
    yaxis_title='M01AB',
    title='M01AB vs Month and Year'
)

# Show the plot
fig.show()

# Convert Year and Month columns to datetime format
df_m01ab['Date'] = pd.to_datetime(df_m01ab[['Year', 'Month']].assign(day=1))

# Plot M01AB vs date
fig = px.line(df_m01ab, x='Date', y='M01AB', title='M01AB vs Date')
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='M01AB')

# Show the plot
fig.show()

df['day'] = df['datum'].dt.day
df

print(df['datum'].min())
print(df['datum'].max())

# Reshape the data to have separate columns for each category
melted_df = pd.melt(df, id_vars=['Year', 'Month'], value_vars=['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06'],
                    var_name='Category', value_name='Consumption')

# Group the data by Category and Month and calculate the total consumption
grouped_df = melted_df.groupby(['Category', 'Month']).sum().reset_index()

# Create the bar chart
fig = px.bar(grouped_df, x='Month', y='Consumption', color='Category', barmode='group')

# Customize the layout
fig.update_layout(
    title='Total Monthly Consumption of Each Category',
    xaxis_title='Month',
    yaxis_title='Consumption',
    legend_title='Category',
)

# Show the chart
fig.show()

df[df['datum'].dt.month == 1]['M01AB'].sum()

df_m01ab

df_m01ab.groupby('Year')['M01AB'].sum().reset_index()

# Calculate the total yearly consumption of M01AB
df_m01ab_yearly = df_m01ab.groupby('Year')['M01AB'].sum().reset_index()

# Create the bar chart
fig = px.bar(df_m01ab_yearly, x='Year', y='M01AB', color='Year')

# Customize the layout
fig.update_layout(
    title='Total Yearly Consumption of M01AB',
    xaxis_title='Year',
    yaxis_title='Consumption',
    showlegend=False
)

# Show the chart
fig.show()

"""### Predictive Data Analysis"""

df.head()

# Reshape the dataframe
df_new = df.melt(id_vars=['datum', 'Year', 'Month', 'Hour', 'Weekday Name', 'day'],
             var_name='Drug',
             value_name='Quantity')

# Print the updated dataframe
df_new.head()

df_new.shape

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df_new['Weekday Name'] = le.fit_transform(df_new['Weekday Name'])
df_new['Drug'] = le.fit_transform(df_new['Drug'])
df_new

df_new.set_index('datum')

train = df_new[df_new['Year'] < 2019]
test = df_new[df_new['Year'] >= 2019]

print(train.shape)
print(test.shape)

X_train = train.drop(['Hour','Quantity','datum'],axis = 1)
y_train = train['Quantity']

X_test = test.drop(['Hour','Quantity','datum'],axis = 1)
y_test = test['Quantity']

X_train

import xgboost as xgb

reg = xgb.XGBRegressor(n_estimators = 1000,early_stopping_rounds = 50, learning_rate = 0.005)
reg.fit(X_train,y_train,
       eval_set = [(X_train,y_train),(X_test,y_test)],
       verbose = 10)

fi = pd.DataFrame(data = reg.feature_importances_, index = reg.feature_names_in_,columns = ['Importance'])
fi

fi.sort_values('Importance').plot(kind = 'barh', title = 'Feature Importance')
plt.show()

### Random Forest

from sklearn.ensemble import RandomForestRegressor

reg_rf = RandomForestRegressor(max_depth=2, random_state=0)
reg_rf.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

rf_pred = reg_rf.predict(X_test)
mse = mean_squared_error(y_test,rf_pred)
rmse = np.sqrt(mse)
rmse

### outliers detection and removel

Q1 = df_new['Quantity'].quantile(0.25)
Q3 = df_new['Quantity'].quantile(0.75)
Q1,Q3

IQR = Q3 - Q1
IQR

lower_lim = Q1 - 1.5*IQR
upper_lim = Q3 + 1.5*IQR
lower_lim,upper_lim

df_new_no_out = df_new[df_new['Quantity'] < 17.27]
df_new_no_out

train = df_new_no_out[df_new_no_out['Year'] < 2019]
test = df_new_no_out[df_new_no_out['Year'] >= 2019]

print(train.shape)
print(test.shape)

X_train = train.drop(['Hour','Quantity','datum'],axis = 1)
y_train = train['Quantity']

X_test = test.drop(['Hour','Quantity','datum'],axis = 1)
y_test = test['Quantity']

reg = xgb.XGBRegressor(n_estimators = 1000,early_stopping_rounds = 50, learning_rate = 0.005)
reg.fit(X_train,y_train,
       eval_set = [(X_train,y_train),(X_test,y_test)],
       verbose = 10)

#### Hyperparameter tuning

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 500, 1000],    # Number of trees in the forest
    'learning_rate': [0.01, 0.1, 0.2],   # Learning rate
    'max_depth': [3, 5, 7],               # Maximum depth of each tree
    'subsample': [0.8, 1.0],              # Subsample ratio of the training instances
    'colsample_bytree': [0.8, 1.0]        # Subsample ratio of columns when constructing each tree
}

# Create the XGBoost regressor
xgb = xgb.XGBRegressor(random_state=42)

from sklearn.model_selection import GridSearchCV

# Perform grid search with cross-validation
grid_search = GridSearchCV(xgb, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding mean squared error
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Mean Squared Error:", -grid_search.best_score_)

# Evaluate the model on the test set using the best hyperparameters
best_xgb = grid_search.best_estimator_
y_pred = best_xgb.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Test Root Mean Squared Error:", rmse)

test['Quantity Predictions'] = y_pred
test

import pickle

# save the model to disk
filename = 'pharma_model.sav'

pickle.dump(best_xgb, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

y_pred = loaded_model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
rmse

# Drug range from 0 to 7
def predict_sales(start_date,end_date,drug): # dates selected from celander and category(int) from options
    # Generate a range of dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create the DataFrame with dates as the index
    df_test = pd.DataFrame(index=dates)
    df_test['Year'] = df_test.index.year
    df_test['Month'] = df_test.index.month
    df_test['Weekday Name'] = df_test.index.weekday
    df_test['day'] = df_test.index.day
    df_test['Drug'] = drug
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
#   df_test['Weekday Name'] = le.fit_transform(df_test['Weekday Name'])
    df_test['predicted_quantity'] = loaded_model.predict(df_test)
    return df_test

predict_sales(start_date = '2023-01-01',end_date = '2023-01-31',drug = 0)

"""! Thank You"""






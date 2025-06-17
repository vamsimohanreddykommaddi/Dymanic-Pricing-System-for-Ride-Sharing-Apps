'''
Problem Statement : 
    
    A ride-sharing company wants to implement a dynamic pricing strategy to optimize fares 
    based on real-time market conditions. 
    The company only uses ride duration to decide ride fares currently. 
    The company aims to leverage data-driven techniques to analyze historical data 
    and develop a predictive model that can dynamically adjust prices in response 
    to changing factors.
    
    Business Objective : Maximize revenue

    Business Constraints : Minimize price discrimination

    Success Criteria :
        Business success criteria : Increased average revenue per ride by 15%
        
        ML success criteria : Achive an accuracy of atleast 85%
        
        Economic success criteria : Achieved at least a 10–20% increase in average revenue per 
        ride compared to the previous static pricing model (based only on ride duration).
        
    Data Collection :
        A dataset containing historical ride data has been provided. 
        The dataset includes features such as the 
        number of riders, 
        number of drivers, 
        location category, 
        customer loyalty status, 
        number of past rides, 
        average ratings, 
        time of booking, 
        vehicle type, 
        expected ride duration, and 
        historical cost of the rides.

'''

# importing necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from feature_engine.outliers import Winsorizer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OrdinalEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split

import sklearn.metrics as skmet

from sklearn.ensemble import RandomForestRegressor

import joblib

import pickle



import sweetviz


# loading the data

data = pd.read_csv(r"D:\Dynamic Pricing strategy\dynamic_pricing.csv")

data.info()


'''
Exploratory Data Analysis
'''

# generating descriptive statistics of the data

data.describe()

# we will look at the relation between Expected_Ride_Duration and Historical_Cost_of_Ride

sns.regplot(data = data, x = 'Expected_Ride_Duration', y = 'Historical_Cost_of_Ride',
                       line_kws={"color": "red"})

plt.title('Expected_Ride_Duration vs Historical_Cost_of_Ride')

plt.show()

#let’s have a look at the distribution of the historical cost of rides based on the vehicle type:

sns.boxplot(data = data, x = 'Vehicle_Type', y = 'Historical_Cost_of_Ride')

plt.title('Historical cost of rides distribution by  vehicle type')

plt.show()

# Now we will perform AutoEDA using Sweetviz 

report = sweetviz.analyze(data)

report.show_html('dynamic_pricing_data_report.html')

'''
from report we can confirm that there are no missing values and 

all features are not correlated much that means much variance which in turn more information
'''

'''
The data provided by the company states that the company uses a pricing model that only takes 
the expected ride duration as a factor to determine the price for a ride. 

Now, we will implement a dynamic pricing strategy aiming to adjust the ride costs dynamically 
based on the demand and supply levels observed in the data. 

It will capture high-demand periods and low-supply scenarios to increase prices, 
low-demand periods and high-supply situations will lead to price reductions.

'''

# Implementing dynamic pricing strategy

# Calculate demand_multiplier based on percentile for high and low demand

high_demand_percentile = 75

low_demand_percentile = 25

'''
we're creating a new column in the DataFrame data called demand_multiplier.

np.where(condition, value_if_true, value_if_false) works like an if-else statement across 
a NumPy array or Series.

Condition: If the number of riders is greater than the 75th percentile → this is considered high demand.

If True (High Demand): You scale the number of riders relative to the 75th percentile.

 -> For example, if 75th percentile is 200 riders and current record has 250 → multiplier = 250 / 200 = 1.25

If False (Low or Normal Demand): You scale the number of riders relative to the 25th percentile.

 -> If 25th percentile is 100 and current value is 80 → multiplier = 80 / 100 = 0.8
 
Here We're calculating a relative demand multiplier:

>1 for high-demand situations,

<1 for low-demand,

≈1 for average demand.

This can then be used to adjust ride pricing dynamically — for instance, 
increasing prices during high demand and decreasing or keeping stable during low demand.

'''
data['demand_multiplier'] = np.where(data['Number_of_Riders'] > np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                     data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], high_demand_percentile),
                                     data['Number_of_Riders'] / np.percentile(data['Number_of_Riders'], low_demand_percentile))


# Calculating Supply multiplier for high and low supply based on percentile(inverse : more supply -> less price)

high_supply_percentile = 75

low_supply_percentile = 25

data['supply_multiplier'] = np.where(data['Number_of_Drivers'] > np.percentile(data['Number_of_Drivers'],high_supply_percentile),
                                     data['Number_of_Drivers'] / np.percentile(data['Number_of_Drivers'], high_supply_percentile),
                                     data['Number_of_Drivers'] / np.percentile(data['Number_of_Drivers'], low_supply_percentile))



# mapping loyality factor for price adjustments

loyalty_map = {'Regular': 1.0, # 0% discount
                'Silver': 0.95, # 5% discount
                'Gold': 0.90 # 10% discount
                }

data['loyality_factor'] = data['Customer_Loyalty_Status'].map(loyalty_map)

# mapping the time factor for pricing adjustments

time_map = {'Morning' : 1.00, # base price
            'Afternoon' : 1.05, # slight increase(mild demand)
            'Evening' : 1.10, # High demand
            'Night' : 1.15 # High demand/low driver availability
    }

data['time_factor'] = data['Time_of_Booking'].map(time_map)


# vehicle type mapping for price adjustments

vehicle_map = {'Economy' : 1.0, # base price
               'Premium' : 1.15 # 15% higher fare for premium vehicles
    }

data['vehicle_type_factor'] = data['Vehicle_Type'].map(vehicle_map)

# defining a rating factor for adjusting prices

data['rating_factor'] = 1 - ((data['Average_Ratings'] - 4.0) * 0.05)



# Define price adjustment factors for high and low demand/supply

#demand_threshold_high = 1.5  # Higher demand threshold

#demand_threshold_low = 0.8  # Lower demand threshold

#supply_threshold_high = 0.8  # Higher supply threshold

#supply_threshold_low = 1.5  # Lower supply threshold

# Calculate adjusted_ride_cost for dynamic pricing

data['adjusted_ride_cost'] = data['Historical_Cost_of_Ride']* (
    np.clip(data['demand_multiplier'], 0.8, 1.5) *
    np.clip(data['supply_multiplier'], 0.8, 1.5) *
    data['loyality_factor'] *
    data['time_factor'] *
    data['vehicle_type_factor'] *
    data['rating_factor']
)

'''
Steps followed to implement dynamic pricing

-> first calculated the demand multiplier by comparing the number of riders to percentiles 
   representing high and low demand levels. If the number of riders exceeds the percentile for 
   high demand, the demand multiplier is set as the number of riders divided by the high-demand 
   percentile. Otherwise, if the number of riders falls below the percentile for low demand, 
   the demand multiplier is set as the number of riders divided by the low-demand percentile.
   
-> Next, we calculated the supply multiplier by comparing the number of drivers to percentiles 
   representing high and low supply levels. If the number of drivers exceeds the low-supply 
   percentile, the supply multiplier is set as the high-supply percentile divided by the 
   number of drivers. On the other hand, if the number of drivers is below the low-supply 
   percentile, the supply multiplier is set as the low-supply percentile divided by the 
   number of drivers.
   
-> Finally, we calculated the adjusted ride cost for dynamic pricing.

'''

'''
np.clip(value, min, max)
It limits the value between a minimum and a maximum.

If the value is lower than min, it becomes min.

If the value is higher than max, it becomes max.

np.clip(data['demand_multiplier'], 0.8, 1.5)
    -> Keeps demand_multiplier between 0.8 and 1.5

    ->If demand is very low (<0.8), it uses 0.8 → avoids underpricing.

    ->If demand is very high (>1.5), it uses 1.5 → avoids overpricing.

np.clip(data['supply_multiplier'], 0.8, 1.5)

    ->Keeps supply_multiplier in the same range — protecting against extreme fare drops or 
      surges due to unusual supply values.
'''

# Now we will calculate profit percentage we got after implementing dynamic pricing

data['profit_percentage'] = ((data['adjusted_ride_cost'] - data['Historical_Cost_of_Ride'])/
                             data['Historical_Cost_of_Ride']) * 100


# find profitable rides where profit percentage is positive 

profit_rides = data[data['profit_percentage'] > 0]

# find loss rides where profit percentage is negative

loss_rides = data[data['profit_percentage'] < 0]

# visualizing profotable and loss rides using plotly

import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'  # Forces Plotly to open in your web browser


# Calculate the count of profitable and loss rides

profitable_count = len(profit_rides)

loss_count = len(loss_rides)

# Create a donut chart to show the distribution of profitable and loss rides

labels = ['Profitable Rides', 'Loss Rides']

values = [profitable_count, loss_count]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])

fig.update_layout(title='Profitability of Rides (Dynamic Pricing vs. Historical Pricing)')

fig.show()

# Now let’s have a look at the relationship between the expected ride duration and the 
# cost of the ride based on the dynamic pricing strategy:

sns.regplot(data = data, x = 'Expected_Ride_Duration', y = 'adjusted_ride_cost',
                       line_kws={"color": "red"})

plt.title('Expected_Ride_Duration vs adjusted_ride_cost')

plt.show()

'''
Now we will preprocess data to prepare it for model building

Data Preprocessing : 

'''

# from eda report we can confirm that therea re no missing values in the data

# creating a new dataframes which cotain input features and target variable  for model 

data_X = pd.DataFrame(data[['Number_of_Riders','Number_of_Drivers','Expected_Ride_Duration','Vehicle_Type']])

data_Y = pd.DataFrame(data['adjusted_ride_cost'])

# now we will seperate numeric and categoric features

numeric_features = data_X.select_dtypes(exclude = 'object').columns

categoric_features = data_X.select_dtypes(include = 'object').columns

# performing box plots on numeric features in input and output data

# Set up a grid of 3 rows x 3 columns
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()  # Flatten to easily index axes in a loop

# Loop through each feature and its corresponding subplot axis
for i, feature in enumerate(numeric_features):
    sns.boxplot(y = data_X[feature], ax=axes[i], color='skyblue')
    axes[i].set_title(f'Boxplot of {feature}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Value')

# Adjust layout
plt.tight_layout()
plt.show()

sns.boxplot(y = data_Y['adjusted_ride_cost'])
plt.show()

'''
from box plots we can see that there are few outliers in the features Number_of_Drivers and 
adjusted_ride_cost

'''
# defining a winsorizer object to handle outliers

winsor_iqr = Winsorizer(capping_method = 'iqr', tail = 'both', fold = 1.5)

# handling outliers

data_X = winsor_iqr.fit_transform(data_X)

data_Y = winsor_iqr.fit_transform(data_Y)

# defining a minmax scaler to scale the numeric features

scaler = MinMaxScaler()

# defining a label encoder to encode categorical features

encoder = OrdinalEncoder()


# Defining a pipeline for numeric features

numeric_pipeline = Pipeline([('scaling', scaler)])

# Defining a pipeline for categoric features

categoric_pipeline = Pipeline([('encoding',encoder)])


# Defining a column transformer to create an entire preprocessing pipeline

preprocessing_pipeline = ColumnTransformer([('num_pipeline',numeric_pipeline, numeric_features),
                                            ('categoric_pipeline',categoric_pipeline, categoric_features)],
                                           remainder = 'passthrough')

# fitting the data to the column transformer and saving the pipeline for future use

dynamic_preprocess = preprocessing_pipeline.fit(data_X)

# saving the preprocessing pipeline

joblib.dump(dynamic_preprocess, 'preprocess_dynamic')

# transforming the data using column transformer

data_X_clean = pd.DataFrame(dynamic_preprocess.transform(data_X), columns = dynamic_preprocess.get_feature_names_out())

data_X_clean.columns = data_X_clean.columns.str.replace('^num_pipeline__|^categoric_pipeline__|remainder__','', regex = True)

'''
We will now build a ML model using the cleaned data

'''

# splitting the data into train and test sets

X_train, X_test, Y_train, Y_test = train_test_split(data_X_clean, data_Y, test_size = 0.2,
                                                    random_state = 42)

# reshaping Y into 1-d array

Y_train = Y_train.to_numpy()

Y_test = Y_test.to_numpy()

# Defining a random forest model and training it using training data

model = RandomForestRegressor()

model.fit(X_train, Y_train)

# making predictions using trained data

predicted_price = model.predict(X_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


mae = mean_absolute_error(Y_train, predicted_price)
mse = mean_squared_error(Y_train, predicted_price)
rmse = root_mean_squared_error(Y_train, predicted_price)
r2 = r2_score(Y_train, predicted_price)

print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R² Score: {r2:.2f}')

# making predictions using test data

predicted_test_price = model.predict(X_test)

mae_test = mean_absolute_error(Y_test, predicted_test_price)
mse_test = mean_squared_error(Y_test, predicted_test_price)
rmse_test = root_mean_squared_error(Y_test, predicted_test_price)
r2_test = r2_score(Y_test, predicted_test_price)

print(f'MAE: {mae_test:.2f}')
print(f'MSE: {mse_test:.2f}')
print(f'RMSE: {rmse_test:.2f}')
print(f'R² Score: {r2_test:.2f}')

'''
Hyperparameter tuning

'''

from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2',
                           n_jobs=-1,
                           verbose=2)

# Fit the model
grid_search.fit(X_train, Y_train.ravel())

print("Best Parameters:", grid_search.best_params_)
print("Best R² Score on Train Set:", grid_search.best_score_) # 0.872

# Get the best model
best_model = grid_search.best_estimator_

# Save the model
with open('best_dynamic_pricing_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Optionally, test the model on test data
test_pred = best_model.predict(X_test)
print("Test R² Score:", r2_score(Y_test, test_pred)) # 0.869



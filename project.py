import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import datetime as dt

# Import the data
df = pd.read_csv(r"C:\Users\himan\Desktop\CUTM DOMAIN WORK\CLASSIFICATION\Cyclone Data Classification\Literature\Cyclone Data Classification.csv")

# Convert date column as datetime
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

# Extract year from the date column
df['Year'] = df['Date'].dt.year

# Function to create hemisphere columns
def hemisphere(coord):
    hem = re.findall(r'[NSWE]', coord)[0]
    return 0 if hem in ['N', 'E'] else 1

# Creating the columns Latitude_Hemisphere and Longitude_Hemisphere
df['Latitude_Hemisphere'] = df['Latitude'].apply(hemisphere)
df['Longitude_Hemisphere'] = df['Longitude'].apply(hemisphere)
df['Latitude_Hemisphere'] = df['Latitude_Hemisphere'].astype('category')
df['Longitude_Hemisphere'] = df['Longitude_Hemisphere'].astype('category')

# Convert the latitude and longitude columns to numeric type
df['Latitude'] = df['Latitude'].apply(lambda x: re.match(r'\d{1,3}\.\d?', x)[0])
df['Longitude'] = df['Longitude'].apply(lambda x: re.match(r'\d{1,3}\.\d?', x)[0])

# Handle missing values
for column in df.columns:
    missing_cnt = df[column][df[column] == -999].count()
    print(f'Missing Values in column {column} = ', missing_cnt)
    if missing_cnt != 0:
        mean = round(df[column][df[column] != -999].mean())
        df.loc[df[column] == -999, column] = mean

# Restructure the dataframe for visibility and remove columns ID and Event
df = df[['ID', 'Name', 'Date', 'Time', 'Event', 'Status', 'Latitude', 'Latitude_Hemisphere',
         'Longitude', 'Longitude_Hemisphere', 'Maximum Wind', 'Minimum Pressure', 'Low Wind NE',
         'Low Wind SE', 'Low Wind SW', 'Low Wind NW', 'Moderate Wind NE',
         'Moderate Wind SE', 'Moderate Wind SW', 'Moderate Wind NW',
         'High Wind NE', 'High Wind SE', 'High Wind SW', 'High Wind NW']]

# Change all time to format HHMM
df['Time'] = df['Time'].astype('object')

def hhmm(time):
    time = str(time)
    digits = re.findall(r'\d', time)
    if len(digits) == 1:
        return f'0{time}00'
    elif len(digits) == 2:
        return f'{time}00'
    elif len(digits) == 3:
        return f'0{time}'
    else:
        return time

# Apply the function
df['Time'] = df['Time'].apply(hhmm)

# Convert the column into Datetime
df['Time'] = pd.to_datetime(df['Time'], format='%H%M').dt.time

# Convert the status column to categorical
df['Status'] = df['Status'].astype('category')

data = df.drop(columns=['ID', 'Event'])


# Find the top ten cyclones which have occurred the maximum number of times
lst = [x.strip() for x in data.groupby('Name').count().sort_values(by='Date', ascending=False).index[:10]]
val = data.groupby('Name').count().sort_values(by='Date', ascending=False)[:10]['Date'].values
font = {'family': 'monospace', 'weight': 'bold', 'size': 22}
plt.rc('font', **font)
fig, ax = plt.subplots()
fig.set_size_inches(12, 12)
ax.pie(labels=lst, x=val, autopct='%.1f%%', explode=[0.1 for _ in range(10)])
plt.title('Top Ten Hurricanes by Frequency.', fontsize=30)


# Year-Wise Frequency of Hurricanes
data['Month'] = data['Date'].apply(lambda x: x.month)
data['Year'] = data['Date'].apply(lambda x: x.year)
mnt = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temp = data.groupby('Month').count()
temp.loc[4] = 0
temp = temp.sort_values(by='Month', ascending=False)
font = {'family': 'monospace', 'weight': 'bold', 'size': 22}
plt.rc('font', **font)
plt.figure(figsize=(10, 10))
sns.set_style("whitegrid")
ax = sns.barplot(x=temp.index, y='Date', data=temp, palette='RdBu')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], mnt, rotation=90)
plt.ylabel('Frequency')
plt.title('Frequency of Cyclones by Month.')

# Probability Distribution Function of Frequency
temp = data.groupby('Year').count().sort_values(by='Date', ascending=False)
plt.figure(figsize=(15, 15))
ax = sns.histplot(temp['Date'].values, kde=True, stat='density', bins=30)
ax.set_xlabel('Probability Distribution of Frequency of Cyclones.')

# Frequency of Cyclones by Category
temp = data.groupby('Status').count().sort_values(by='Date', ascending=False)
fig, ax = plt.subplots()
fig.set_size_inches(12, 12)
sns.barplot(y=list(temp.index), x='Date', data=temp, palette='pastel')
plt.xlabel('Frequency')
plt.ylabel('Category')
plt.title('Category-wise Frequency Distribution of Cyclones.')



# 1. Apply normal feature selstcion techniques and find its importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns

# Assuming we are predicting 'Status' of the cyclone

# Drop unnecessary columns and target variable
X = data.drop(columns=['Name', 'Status', 'Date', 'Time'])
y = data['Status']

# Correlation Matrix with Heatmap
plt.figure(figsize=(15,10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a RandomForestClassifier to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for i in range(X.shape[1]):
    print(f"{i + 1}. feature {X.columns[indices[i]]} ({importances[indices[i]]})")

# Plot the feature importance
plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.barh(range(X.shape[1]), importances[indices], align="center")
plt.yticks(range(X.shape[1]), [X.columns[i] for i in indices])
plt.gca().invert_yaxis()


# Use SelectFromModel to reduce the features
sfm = SelectFromModel(rf, threshold=0.1)  # Threshold can be adjusted
sfm.fit(X_train, y_train)
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

#Decision Tree
# Import Decision Tree Classifier.
from sklearn.tree import DecisionTreeClassifier

# Import train-test split.
from sklearn.model_selection import train_test_split

# Import accuracy Score.
from sklearn.metrics import accuracy_score

#Import Recall Score.
from sklearn.metrics import recall_score

#Import Precision Score.
from sklearn.metrics import precision_score

# Form the model.
dt = DecisionTreeClassifier(min_samples_leaf=50 , criterion='entropy')


# Set the dependent and independent variables.
x_train = data[['Latitude', 'Latitude_Hemisphere',
       'Longitude', 'Longitude_Hemisphere', 'Maximum Wind', 'Minimum Pressure',
       'Low Wind NE', 'Low Wind SE', 'Low Wind SW', 'Low Wind NW',
       'Moderate Wind NE', 'Moderate Wind SE', 'Moderate Wind SW',
       'Moderate Wind NW', 'High Wind NE', 'High Wind SE', 'High Wind SW',
       'High Wind NW' , 'Month' , 'Year']]
y_train = data['Status']

#Random Forest
# Import Random Forest
from sklearn.ensemble import RandomForestClassifier

# First I want to determine the important features.
rf = RandomForestClassifier(oob_score=True , n_estimators=1000)
rf.fit(x_train , y_train)
features = pd.Series(rf.feature_importances_ , index= x_train.columns).sort_values(ascending=False)



# Set the dependent and independent variables.
x_trainf = data[features.index[:5]]
y_train = data['Status']

# Import necessary libraries for metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
# Split the data into train and test sets for Random Forest
X_trainf, X_testf, y_trainf, y_testf = train_test_split(x_trainf, y_train, test_size=0.3, random_state=42)
# Fit Decision Tree Model
dt = DecisionTreeClassifier(min_samples_leaf=50, criterion='entropy', random_state=42)
dt.fit(X_trainf, y_trainf)
# Predictions using Decision Tree
y_pred_dt = dt.predict(X_testf)
# Fit Random Forest Model
rf = RandomForestClassifier(oob_score=True, n_estimators=1000, random_state=42)
rf.fit(X_trainf, y_trainf)
# Predictions using Random Forest
y_pred_rf = rf.predict(X_testf)
# Save the features along with the model
import joblib
joblib.dump((rf, features), 'random_forest_model.pkl')




# Evaluate Decision Tree model and store the metrics
accuracy_dt = accuracy_score(y_testf, y_pred_dt)
precision_dt = precision_score(y_testf, y_pred_dt, average='weighted')
recall_dt = recall_score(y_testf, y_pred_dt, average='weighted')
f1_dt = f1_score(y_testf, y_pred_dt, average='weighted')



# Evaluate Random Forest model and store the metrics
accuracy_rf = accuracy_score(y_testf, y_pred_rf)
precision_rf = precision_score(y_testf, y_pred_rf, average='weighted')
recall_rf = recall_score(y_testf, y_pred_rf, average='weighted')
f1_rf = f1_score(y_testf, y_pred_rf, average='weighted')

# Create a DataFrame for comparison
comparison_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Decision Tree': [accuracy_dt, precision_dt, recall_dt, f1_dt],
    'Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf]
})




# Assuming your model requires these five features
features = ['Maximum Wind', 'Minimum Pressure', 'Latitude', 'Year', 'Longitude']

# Cyclone type descriptions
cyclone_descriptions = {
    "DB": "A weak, unorganized system of showers and thunderstorms without defined circulation. It poses minimal threats, usually causing localized rain and minor winds but can potentially develop into a more significant storm.",
    "ET": "A system driven by temperature contrasts rather than ocean warmth, usually affecting higher latitudes. It brings strong winds, heavy rain, and sometimes snow, causing flooding, property damage, and transportation disruptions.",
    "EX": "A tropical cyclone that fully transitions into an extratropical system, causing heavy rain, high winds, and sometimes snow. It impacts both coastal and inland areas with significant damage potential.",
    "HU": "A tropical cyclone with winds of 74 mph (119 km/h) or higher, categorized on the Saffir-Simpson scale. Hurricanes can cause devastating damage from storm surges, flooding, and wind, especially in stronger categories.",
    "LO": "A low-pressure system that could develop into a tropical or extratropical cyclone. It brings minimal impact, usually just rain and minor winds, but can evolve into more hazardous weather.",
    "PT": "A former hurricane that has lost its tropical characteristics but can still bring strong winds, rain, and high seas. It remains dangerous but generally less intense than its tropical form.",
    "SD": "A system with features of both tropical and extratropical cyclones, typically producing weaker winds. It mainly brings rain and localized flooding but can develop into more intense storms.",
    "SS": "A hybrid storm with characteristics of tropical and extratropical systems, with winds of at least 39 mph. It can cause moderate winds, heavy rainfall, and flooding, potentially becoming a stronger tropical storm.",
    "ST": "A powerful tropical cyclone in the western Pacific with winds exceeding 150 mph. It causes catastrophic damage, widespread destruction, storm surges, and long-term impacts on infrastructure and life.",
    "TD": "A weak tropical cyclone with winds below 39 mph. It primarily brings rain and minor flooding but can strengthen into a more intense system under favorable conditions.",
    "TS": "A tropical cyclone with winds between 39 and 73 mph. It can cause significant rainfall, coastal flooding, and wind damage, and may strengthen into a hurricane under the right conditions."
}

# Function for user input and prediction
def classify_cyclone(max_wind, min_pressure, latitude, year, longitude):
    input_data = pd.DataFrame([[max_wind, min_pressure, latitude, year, longitude]], columns=features)
    prediction = rf.predict(input_data)
    return prediction[0].strip()  # Strip extra spaces from the prediction result

# Example user input
max_wind = float(input("Enter Maximum Wind: "))
min_pressure = float(input("Enter Minimum Pressure: "))
latitude = float(input("Enter Latitude: "))
year = int(input("Enter Year: "))
longitude = float(input("Enter Longitude: "))

# Classify the cyclone type
predicted_cyclone = classify_cyclone(max_wind, min_pressure, latitude, year, longitude)

# Get the description for the predicted cyclone type
description = cyclone_descriptions.get(predicted_cyclone, "No description available for this cyclone type.")

# Print the predicted cyclone type and its description
print(f"The predicted cyclone type is: {predicted_cyclone}")
print(f"Description: {description}")

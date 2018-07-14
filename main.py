import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

'''Data Retrieva'''
# read csv
df = pd.read_csv('E0.csv')
df = df.dropna(subset=['FTHG', 'FTAG'])
df['Date'] = pd.to_datetime(df['Date'])
df['DATE_DELTA'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')

'''Data Preparation'''

"""Feature Extraction and Engineering"""
feature_names = ['DATE_DELTA', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
training_features = df[feature_names]
outcome_name = ['FTR']
outcome_labels = df[outcome_name]

"""Now that we have extracted our initial available features from the data and their corresponding outcome
labels, let’s separate out our available features based on their type (numerical and categorical)"""
numeric_feature_names = ['DATE_DELTA', 'B365H', 'B365D', 'B365A']
categoricial_feature_names = ['HomeTeam', 'AwayTeam']

"""We will now use a standard scalar from scikit-learn to scale or normalize our two numeric scorebased attributes """
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# fit scaler on numeric features
ss.fit(training_features[numeric_feature_names])

# scale numeric features now
training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])
training_features = pd.get_dummies(training_features, columns=categoricial_feature_names)

# get list of new categorical features
categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))

'''Modeling'''
"""We will now build a simple classification (supervised) model based on our feature set by using the logistic
regression algorithm"""
from sklearn.linear_model import LogisticRegression
import numpy as np

# fit the model
lr = LogisticRegression()
model = lr.fit(training_features, np.array(outcome_labels['FTR']))
# view model parameters

'''Model Evaluation'''
# simple evaluation on training data
pred_labels = model.predict(training_features)
actual_labels = np.array(outcome_labels['FTR'])
# evaluate model performance
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# print('Accuracy:', float(accuracy_score(actual_labels, pred_labels)) * 100, '%')
# print('Classification Stats:')
# print(classification_report(actual_labels, pred_labels))

'''Model Deployment'''
from sklearn.externals import joblib
import os

# save models to be deployed on your server
if not os.path.exists('Model'):
    os.mkdir('Model')
if not os.path.exists('Scaler'):
    os.mkdir('Scaler')
joblib.dump(model, r'Model/model.pickle')
joblib.dump(ss, r'Scaler/scaler.pickle')

"""Prediction in Action"""

# load model and scaler objects
model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')

## data retrieval
new_data = pd.DataFrame(
    [{'DATE_DELTA': '300', 'HomeTeam': 'Man United', 'AwayTeam': 'Leicester', 'B365H': 1.36, 'B365D': 4.5, 'B365A': 9}])

## We will now carry out the tasks relevant to data preparation—feature extraction, engineering, and scaling
## data preparation
prediction_features = new_data[feature_names]
# scaling
prediction_features[numeric_feature_names] = scaler.transform(prediction_features[numeric_feature_names])
# engineering categorical variables
prediction_features = pd.get_dummies(prediction_features, columns=categoricial_feature_names)
# view feature set

# add missing categorical feature columns
current_categorical_engineered_features = set(prediction_features.columns) - set(numeric_feature_names)
missing_features = set(categorical_engineered_features) - current_categorical_engineered_features

for feature in missing_features:
    # add zeros since feature is absent in these data samples
    prediction_features[feature] = [0] * len(prediction_features)
    # view final feature set

## predict using model
predictions = model.predict(prediction_features)
## display results
new_data['FTR'] = predictions

print(new_data)

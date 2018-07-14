from sklearn.externals import joblib
import pandas as pd

"""Prediction in Action"""

# load model and scaler objects
model = joblib.load(r'Model/model.pickle')
scaler = joblib.load(r'Scaler/scaler.pickle')

## data retrieval
new_data = pd.DataFrame([{'Date': '10/08/18', 'HomeTeam': 'Man United', 'AwayTeam': 'Leicester', 'B365H': 1.36, 'B365D': 4.5, 'B365A': 9}])

## We will now carry out the tasks relevant to data preparation—feature extraction, engineering, and scaling
## data preparation
prediction_features = new_data[feature_names]
# scaling
prediction_features[numeric_feature_names] = scaler.transform(prediction_features[numeric_feature_names])
# engineering categorical variables
prediction_features = pd.get_dummies(prediction_features,columns=categoricial_feature_names)
# view feature set


print(prediction_features)

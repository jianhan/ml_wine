import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

'''Data Retrieva'''
# read csv
df = pd.read_csv('epl.csv')
df = df.dropna(subset=['FTHG', 'FTAG'])
df['YEAR'] = pd.to_datetime(df['Date']).map(lambda x: x.year)
df['MONTH'] = pd.to_datetime(df['Date']).map(lambda x: x.month)
df['DAY'] = pd.to_datetime(df['Date']).map(lambda x: x.day)

'''Data Preparation'''

"""Feature Extraction and Engineering"""
feature_names = ['YEAR', 'MONTH', 'DAY', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
training_features = df[feature_names]
outcome_name = ['FTR']
outcome_labels = df[outcome_name]

from sklearn.preprocessing import LabelEncoder

gle = LabelEncoder()

# Home team
home_team_labels = gle.fit_transform(training_features['HomeTeam'])
home_team_mappings = {index: label for index, label in enumerate(gle.classes_)}
print(home_team_mappings)
training_features['HomeTeamLabel'] = home_team_labels
training_features[['YEAR', 'MONTH', 'DAY', 'B365H', 'B365D', 'B365A', 'HomeTeamLabel']].iloc[
1:len(training_features)]

# Away team
away_team_labels = gle.fit_transform(training_features['AwayTeam'])
away_team_mappings = {index: label for index, label in enumerate(gle.classes_)}
training_features['AwayTeamLabel'] = away_team_labels
training_features[
    ['YEAR', 'MONTH', 'DAY', 'B365H', 'B365D', 'B365A', 'HomeTeamLabel', 'AwayTeamLabel']].iloc[
1:len(training_features)]
print(away_team_mappings)
training_features = training_features.drop('HomeTeam', axis=1)
training_features = training_features.drop('AwayTeam', axis=1)


# 4. Split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(training_features, outcome_labels,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=outcome_labels)
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# 6. Declare hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

# 7. Tune model using cross-validation pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np
#
# fit the model
lr = LogisticRegression()
model = lr.fit(X_train, y_train.values.ravel())

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
pred = model.predict(X_test)
print('Accuracy:', float(accuracy_score(y_test, pred)) * 100, '%')
print('Classification Stats:')
print(classification_report(y_test, pred))

from sklearn.tree import DecisionTreeClassifier
dct = DecisionTreeClassifier()
dctModel = dct.fit(X_train, y_train.values.ravel())
dctPred = dctModel.predict(X_test)
print('Accuracy:', float(accuracy_score(y_test, dctPred)) * 100, '%')
print('Classification Stats:')
print(classification_report(y_test, dctPred))

# 8. Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)

# 9. Evaluate model pipeline on test data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#
# pred = clf.predict(X_test)
# print('Accuracy:', float(accuracy_score(y_test, pred)) * 100, '%')
# print('Classification Stats:')
# print(classification_report(y_test, pred))

# print(mean_squared_error(y_test, pred))

#
# """Now that we have extracted our initial available features from the data and their corresponding outcome
# labels, let’s separate out our available features based on their type (numerical and categorical)"""
# numeric_feature_names = ['YEAR', 'MONTH', 'DAY', 'B365H', 'B365D', 'B365A']
# categoricial_feature_names = ['HomeTeam', 'AwayTeam']
#
# """We will now use a standard scalar from scikit-learn to scale or normalize our two numeric scorebased attributes """
# from sklearn.preprocessing import StandardScaler
#
# ss = StandardScaler()
# # fit scaler on numeric features
# ss.fit(training_features[numeric_feature_names])
#
# # scale numeric features now
# training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])
# training_features = pd.get_dummies(training_features, columns=categoricial_feature_names)
#
# # get list of new categorical features
# categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))
#
# '''Modeling'''
# """We will now build a simple classification (supervised) model based on our feature set by using the logistic
# regression algorithm"""
# from sklearn.linear_model import LogisticRegression
# import numpy as np
#
# # fit the model
# lr = LogisticRegression()
# model = lr.fit(training_features, np.array(outcome_labels['FTR']))
# # view model parameters
#
# '''Model Evaluation'''
# # simple evaluation on training data
# pred_labels = model.predict(training_features)
# actual_labels = np.array(outcome_labels['FTR'])
# # evaluate model performance
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import classification_report
#
# print('Accuracy:', float(accuracy_score(actual_labels, pred_labels)) * 100, '%')
# print('Classification Stats:')
# print(classification_report(actual_labels, pred_labels))
#
# '''Model Deployment'''
# from sklearn.externals import joblib
# import os
#
# # save models to be deployed on your server
# if not os.path.exists('Model'):
#     os.mkdir('Model')
# if not os.path.exists('Scaler'):
#     os.mkdir('Scaler')
# joblib.dump(model, r'Model/model.pickle')
# joblib.dump(ss, r'Scaler/scaler.pickle')
#
# """Prediction in Action"""
#
# # load model and scaler objects
# model = joblib.load(r'Model/model.pickle')
# scaler = joblib.load(r'Scaler/scaler.pickle')
#
# ## data retrieval
# new_data = pd.DataFrame(
#     [{'YEAR': "2018", 'MONTH': '08', 'DAY': '11', 'HomeTeam': 'Man United', 'AwayTeam': 'Leicester', 'B365H': 1.36,
#       'B365D': 4.5, 'B365A': 9}])
#
# ## We will now carry out the tasks relevant to data preparation—feature extraction, engineering, and scaling
# ## data preparation
# prediction_features = new_data[feature_names]
# # scaling
# prediction_features[numeric_feature_names] = scaler.transform(prediction_features[numeric_feature_names])
# # engineering categorical variables
# prediction_features = pd.get_dummies(prediction_features, columns=categoricial_feature_names)
# # view feature set
#
# # add missing categorical feature columns
# current_categorical_engineered_features = set(prediction_features.columns) - set(numeric_feature_names)
# missing_features = set(categorical_engineered_features) - current_categorical_engineered_features
#
# for feature in missing_features:
#     # add zeros since feature is absent in these data samples
#     prediction_features[feature] = [0] * len(prediction_features)
#     # view final feature set
#
# ## predict using model
# predictions = model.predict(prediction_features)
# ## display results
# new_data['FTR'] = predictions
#
# print(new_data)

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

'''Data Retrieva'''
# read csv
df = pd.read_csv('E0.csv')
df = df.dropna(subset=['FTHG', 'FTAG'])

'''Data Preparation'''

"""Feature Extraction and Engineering"""
feature_names = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
training_features = df[feature_names]
outcome_name = ['FTR']
outcome_labels = df[outcome_name]

"""Now that we have extracted our initial available features from the data and their corresponding outcome
labels, let’s separate out our available features based on their type (numerical and categorical)"""
numeric_feature_names = ['FTHG', 'FTAG']
categoricial_feature_names = ['HomeTeam', 'AwayTeam']

"""We will now use a standard scalar from scikit-learn to scale or normalize our two numeric scorebased attributes """
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# fit scaler on numeric features
ss.fit(training_features[numeric_feature_names])

# scale numeric features now
training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])
training_features = pd.get_dummies(training_features,columns=categoricial_feature_names)

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
print('Accuracy:', float(accuracy_score(actual_labels,pred_labels))*100, '%')
print('Classification Stats:')
print(classification_report(actual_labels, pred_labels))

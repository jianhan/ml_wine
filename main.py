import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

'''Data Retrieva'''
# read csv
df = pd.read_csv('E0.csv')
df = df.dropna(subset=['FTHG', 'FTAG'])

'''Data Preparation'''

"""Feature Extraction and Engineering"""
feature_names = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']
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

print(training_features)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def cleanup_column_names(df,rename_dict={},do_inplace=True):
    """This function renames columns of a pandas dataframe
    It converts column names to snake case if rename_dict is not passed.
    Args:
    rename_dict (dict): keys represent old column names and values point to
    newer ones
    do_inplace (bool): flag to update existing dataframe or return a new one
    Returns:
    pandas dataframe if do_inplace is set to False, None otherwise
    """
    if not rename_dict:
        return df.rename(columns={col: col.lower().replace(' ','_')
        for col in df.columns.values.tolist()},inplace=do_inplace)
    else:
        return df.rename(columns=rename_dict,inplace=do_inplace)

def dataCollection():
    # read csv
    df = pd.read_csv('epl.csv')
    df = df.dropna(subset=['FTHG', 'FTAG'])
    df['YEAR'] = pd.to_datetime(df['Date']).map(lambda x: x.year)
    df['MONTH'] = pd.to_datetime(df['Date']).map(lambda x: x.month)
    df['DAY'] = pd.to_datetime(df['Date']).map(lambda x: x.day)
    cleanup_column_names(df)
    print("Number of rows::", df.shape[0])
    print("Number of columns::", df.shape[1])
    print("Column Names::", df.columns.values.tolist())
    print("Column Data Types::\n", df.dtypes)
    print("Columns with Missing Values::", df.columns[df.isnull().any()].tolist())
    print("Number of rows with Missing Values::", len(pd.isnull(df).any(1).nonzero()[0].tolist()))
    print("Sample Indices with missing data::", pd.isnull(df).any(1).nonzero()[0].tolist()[0:5])
    print("General Stats::")
    print(df.info())
    print("Summary Stats::")
    print(df.describe())
    return df

def main():
    df = dataCollection()

if __name__ == '__main__':
    main()

exit(0)


'''Data Preparation'''

"""Feature Extraction and Engineering"""
feature_names = ['YEAR', 'MONTH', 'DAY', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
training_features = df[feature_names]
outcome_name = ['FTR']
outcome_labels = df[outcome_name]

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


X_train, X_test, y_train, y_test = train_test_split(training_features, outcome_labels,
                                                    test_size=0.2,
                                                    random_state=123,
                                                    stratify=outcome_labels)

#####################################################################
## SCALING

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

# 6. Declare hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

# 7. Tune model using cross-validation pipeline

import numpy as np

#
# fit the model
lr = LogisticRegression()
model = lr.fit(X_train, y_train.values.ravel())

pred = model.predict(X_test)
print('Accuracy:', float(accuracy_score(y_test, pred)) * 100, '%')
print('Classification Stats:')
print(classification_report(y_test, pred))

dct = DecisionTreeClassifier()
dctModel = dct.fit(X_train, y_train.values.ravel())
dctPred = dctModel.predict(X_test)
print('Accuracy:', float(accuracy_score(y_test, dctPred)) * 100, '%')
print('Classification Stats:')
print(classification_report(y_test, dctPred))

knc = KNeighborsClassifier()
kncModel = knc.fit(X_train, y_train.values.ravel())
kncPred = kncModel.predict(X_test)
print('Accuracy:', float(accuracy_score(y_test, kncPred)) * 100, '%')
print('Classification Stats:')
print(classification_report(y_test, kncPred))

gnb = GaussianNB()
gnbModel = gnb.fit(X_train, y_train.values.ravel())
gnbPred = gnbModel.predict(X_test)
print('Accuracy:', float(accuracy_score(y_test, gnbPred)) * 100, '%')
print('Classification Stats:')
print(classification_report(y_test, gnbPred))

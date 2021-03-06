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
import matplotlib.pyplot as plt


def cleanup_column_names(df, rename_dict={}, do_inplace=True):
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
        return df.rename(columns={col: col.lower().replace(' ', '_')
                                  for col in df.columns.values.tolist()}, inplace=do_inplace)
    else:
        return df.rename(columns=rename_dict, inplace=do_inplace)


def format_result_type(r_type):
    if r_type == "H":
        return 1
    elif r_type == "D":
        return 2
    elif r_type == "A":
        return 3
    else:
        return 'error'


def collection():
    # read csv
    df = pd.read_csv('epl.csv')
    df = df.dropna(subset=['FTHG', 'FTAG', 'BWH', 'BWD', 'IWH', 'IWD', 'IWA', 'LBH'])
    df['YEAR'] = pd.to_datetime(df['Date']).map(lambda x: x.year)
    df['MONTH'] = pd.to_datetime(df['Date']).map(lambda x: x.month)
    df['DAY'] = pd.to_datetime(df['Date']).map(lambda x: x.day)
    df['FTR_NUM'] = df['FTR'].map(format_result_type)
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


def visulization(df):
    # TODO: get some useful graph up
    pass


def featureEngineering(df):
    feature_names = ['year', 'month', 'day', 'hometeam', 'awayteam', 'b365h', 'b365d', 'b365a']
    training_features = df[feature_names]
    outcome_name = ['ftr_num']
    outcome_labels = df[outcome_name]

    gle = LabelEncoder()

    # Home team
    home_team_labels = gle.fit_transform(training_features['hometeam'])
    home_team_mappings = {index: label for index, label in enumerate(gle.classes_)}
    print(home_team_mappings)

    training_features['home_team_label'] = home_team_labels
    training_features[['year', 'month', 'day', 'b365h', 'b365d', 'b365a', 'home_team_label']].iloc[
    1:len(training_features)]

    # Away team
    away_team_labels = gle.fit_transform(training_features['awayteam'])
    away_team_mappings = {index: label for index, label in enumerate(gle.classes_)}
    training_features['away_team_label'] = away_team_labels
    training_features[
        ['year', 'month', 'day', 'b365h', 'b365d', 'b365a', 'home_team_label', 'away_team_label']].iloc[
    1:len(training_features)]
    print(away_team_mappings)

    training_features = training_features.drop('hometeam', axis=1)
    training_features = training_features.drop('awayteam', axis=1)
    return training_features, outcome_labels


def splitDataset(df, outcome_labels):
    X_train, X_test, y_train, y_test = train_test_split(df, outcome_labels,
                                                        test_size=0.2,
                                                        random_state=123,
                                                        stratify=outcome_labels)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test


def gaussianNB(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnbModel = gnb.fit(X_train, y_train)
    gnbPred = gnbModel.predict(X_test)
    print('Accuracy:', float(accuracy_score(y_test, gnbPred)) * 100, '%')
    print('Classification Stats:')
    print(classification_report(y_test, gnbPred))


def logisticRegression(X_train, X_test, y_train, y_test):
    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)

    pred = model.predict(X_test)
    print('Accuracy:', float(accuracy_score(y_test, pred)) * 100, '%')
    print('Classification Stats:')
    print(classification_report(y_test, pred))


def main():
    df = collection()
    visulization(df)
    df, outcome_labels = featureEngineering(df)
    X_train, X_test, y_train, y_test = splitDataset(df, outcome_labels)
    logisticRegression(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    main()

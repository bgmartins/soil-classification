import pandas as pd
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data = pd.read_csv('../data/china_depth.csv')
profiles = pd.read_csv('../data/profiles.csv')

profiles = profiles[['profile_id', 'cwrb_reference_soil_group']]


clf = classifier = sklearn.ensemble.RandomForestClassifier(
    n_estimators=1000, n_jobs=5)

profiles = profiles.merge(data, how="inner", left_on=[
    'profile_id'], right_on=['profile_id'])


y = profiles.cwrb_reference_soil_group
X = profiles[profiles.columns.values[3:]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


print(cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1))

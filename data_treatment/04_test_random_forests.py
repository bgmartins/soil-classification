import sys
import getopt
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir
from os.path import basename
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# Set variables and defaults
inputfile = '../data/depth_merged_data.csv'
profile_file = '../data/profiles.csv'
input_folder = ''

h = '04_test_random_forests.py -h <help> -i <input file> -f <input folder> -p <profiles file>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:f:p:")
except getopt.GetoptError:
    print(h)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print(h)
        sys.exit()
    elif opt in ('-i'):
        inputfile = arg
    elif opt in ('-f'):
        input_folder = arg
    elif opt in ('-p'):
        profile_file = arg

if input_folder != '':
    files = listdir(input_folder)
else:
    files = [inputfile]

profiles = pd.read_csv(profile_file)
profiles = profiles[['profile_id', 'cwrb_reference_soil_group']]

for file in files:
    print('Training Random Forest on {}'.format(file))
    if input_folder != '':
        data = pd.read_csv(input_folder + '/' + file)
    else:
        data = pd.read_csv(file)

    # After some tests these seem to be the best params
    clf = RandomForestClassifier(
        n_estimators=1200, n_jobs=5, min_samples_leaf=2, oob_score=True)

    data = profiles.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    y = data.cwrb_reference_soil_group
    X = data[data.columns.values[3:]]

    X = X.drop(columns=list(
        X.loc[:, X.columns.str.contains('longitude')]))
    X = X.drop(columns=list(
        X.loc[:, X.columns.str.contains('latitude')]))
    X = X.drop(columns=list(
        X.loc[:, X.columns.str.contains('profile_layer_id')]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(accuracy_score(y_test, preds))

    df_ = pd.DataFrame(X.columns, columns=['feature'])
    df_['fscore'] = clf.feature_importances_[:, ]

    df_.sort_values('fscore', ascending=False, inplace=True)
    df_ = df_[0:30]
    df_.sort_values('fscore', ascending=True, inplace=True)
    df_.plot(kind='barh', x='feature', y='fscore',
             color='blue', legend=False, figsize=(12, 6))
    plt.title('Random forest feature importance', fontsize=24)
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig('feature_importance_{}.png'.format(basename(file)))

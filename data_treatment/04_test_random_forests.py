import sys
import getopt
import pandas as pd
from os import listdir
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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
    clf = classifier = RandomForestClassifier(
        n_estimators=1200, n_jobs=5, min_samples_leaf=2, oob_score=True)

    data = profiles.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    y = data.cwrb_reference_soil_group
    X = data[data.columns.values[3:]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    print(cross_val_score(clf, X_train, y_train, cv=10, n_jobs=-1))

    # 50 average 0,555180235
    # 25 average 0,557494699
    # 10 average 0,55182848

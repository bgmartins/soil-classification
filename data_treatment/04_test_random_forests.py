import sys
import getopt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from os import listdir
from os.path import splitext, basename
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, make_scorer


def classification_report_with_accuracy_score(y_true, y_pred):
    print("==== REPORT OF RESULTS FOR ONE OF THE DATA FOLDS  ====")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print(classification_report(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    pre_micro = precision_score(y_true, y_pred, average='micro')
    rec_micro = recall_score(y_true, y_pred, average='micro')
    f_1_micro = f1_score(y_true, y_pred, average='micro')
    pre_macro = precision_score(y_true, y_pred, average='macro')
    rec_macro = recall_score(y_true, y_pred, average='macro')
    f_1_macro = f1_score(y_true, y_pred, average='macro')
    pre_weighted = precision_score(y_true, y_pred, average='weighted')
    rec_weighted = recall_score(y_true, y_pred, average='weighted')
    f_1_weighted = f1_score(y_true, y_pred, average='weighted')
    return acc


def remove_lat_lon(X):
    print("Removing latitude and longitude")

    X = X.drop(columns=list(
        X.loc[:, X.columns.str.contains('longitude')]))
    X = X.drop(columns=list(
        X.loc[:, X.columns.str.contains('latitude')]))

    return X


def remove_small_classes(df, min):
    uniques = df.cwrb_reference_soil_group.unique()
    for u in uniques:
        cnt = df[df.cwrb_reference_soil_group == u].shape[0]
        if cnt < min:
            df = df[df.cwrb_reference_soil_group != u]
            print('Deleting {} with {} occurrences'.format(u, cnt))

    return df


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(13, 13))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.savefig(title)


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

    data = remove_small_classes(data, 15)

    y = data.cwrb_reference_soil_group
    X = data[data.columns.values[3:]]

    # Remove unecessary columns
    # X = remove_lat_lon(X)
    X = X.drop(columns=list(
        X.loc[:, X.columns.str.contains('profile_layer_id')]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25)

    # , scoring=make_scorer(classification_report_with_accuracy_score))
    res = cross_val_score(clf, X_train, y_train, cv=10)
    print(res, 'avg: ', np.mean(res))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print(accuracy_score(y_test, y_pred))

    labels = list(y.unique())
    labels.sort()
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plot_confusion_matrix(
        cm, labels, title='confusion_matrix_{}.pdf'.format(splitext(basename(file))[0]), normalize=True)

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

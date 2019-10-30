import sys
import getopt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.metrics import cohen_kappa_score, precision_recall_curve, auc, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from os.path import splitext, basename
from os import listdir
from xgboost import XGBClassifier


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def plot_confusion_matrix_2(y_true, y_pred, classes,
                            normalize=True,
                            title=None,
                            cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    # plt.savefig(title+'.pdf')
    return ax


def classification_report_with_accuracy_score(y_true, y_pred, print_score=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if print_score:
        print("==== REPORT OF RESULTS FOR ONE OF THE DATA FOLDS  ====")
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
    kappa = cohen_kappa_score(y_true, y_pred)
    global test_results
    global test_results_y_true
    global test_results_y_pred
    test_results_y_true += list(y_true)
    test_results_y_pred += list(y_pred)
    test_results.append(
        (acc, pre_micro, rec_micro, f_1_micro, pre_macro, rec_macro, f_1_macro, kappa))
    return acc


def remove_small_classes(df, min):
    uniques = df.cwrb_reference_soil_group.unique()
    for u in uniques:
        cnt = df[df.cwrb_reference_soil_group == u].shape[0]
        if cnt < min:
            df = df[df.cwrb_reference_soil_group != u]
            print('Deleting {} with {} occurrences'.format(u, cnt))
    return df


# Set variables and defaults
inputfile = '../data/test/mexico_k_1_layers_5.csv'
profile_file = '../data/profiles.csv'
input_folder = ''
return_results = False
plot_stuff = False

h = '04_test_gradie_boosting.py -h <help> -i <input file> -f <input folder> -p <profiles file> -g <plot>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:f:p:rg")
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
    elif opt in ('-r'):
        return_results = True
    elif opt in ('-g'):
        plot_stuff = True


if input_folder != '':
    files = listdir(input_folder)
else:
    files = [inputfile]

profiles = pd.read_csv(profile_file)
profiles = profiles[['profile_id', 'cwrb_reference_soil_group']]

final_results = []
for file in files:
    print(f'testing {file}')
    test_results = []
    test_results_y_true = list()
    test_results_y_pred = list()

    print('Training XGB on {}'.format(file))
    if input_folder != '':
        data = pd.read_csv(input_folder + '/' + file)
    else:
        data = pd.read_csv(file)

    data.dropna(inplace=True)
    data = profiles.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    data = remove_small_classes(data, 15)

    y = data.cwrb_reference_soil_group
    X = data.drop(['profile_id', 'cwrb_reference_soil_group'], axis=1)

    # Remove unecessary columns
    X = X.drop(columns=list(
        X.loc[:, X.columns.str.contains('profile_layer_id')]))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)

    clf = XGBClassifier(n_estimators=300, min_child_weight=7,
                        gamma=0.2, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)

    res = cross_val_score(clf, X_train, y_train, cv=10, scoring=make_scorer(
        classification_report_with_accuracy_score))

    clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)
    # print(accuracy_score(y_test, y_pred))

    print('Results --------------\n\n\n\n')
    classification_report_with_accuracy_score(
        test_results_y_true, test_results_y_pred, print_score=True)

    kappa = cohen_kappa_score(test_results_y_true, test_results_y_pred)
    print('Fold accuracy', res, '\nAverage: ',
          np.mean(res), 'Kappa Score', kappa)
    final_results.append((file, str(np.mean(res)), kappa))
    print('Results  --------------\n\n\n\n')

    # Plot things
    if plot_stuff:
        labels = list(y.value_counts().index)
        plot_confusion_matrix_2(test_results_y_true,
                                test_results_y_pred, classes=labels)

        df_ = pd.DataFrame(X.columns, columns=['feature'])
        df_['fscore'] = clf.feature_importances_[:, ]
        df_.sort_values('fscore', ascending=False, inplace=True)
        df_ = df_[0:10]
        df_.sort_values('fscore', ascending=True, inplace=True)
        df_.plot(kind='barh', x='feature', y='fscore',
                 color='blue', legend=False)
        plt.xlabel('Relative Importance')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()
        # plt.savefig('feature_importance_{}.pdf'.format(
        #    basename(file)))


# Print and store final results
if return_results:
    for line in final_results:
        print(line)
    results_df = pd.DataFrame(final_results, columns=[
        "Filename", "Accuracy", "Kappa"])
    results_df.to_csv('results.csv', index=False)

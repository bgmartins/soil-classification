import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, precision_recall_curve, auc, accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import cohen_kappa_score
from keras.utils import np_utils
from sklearn.utils.class_weight import compute_class_weight


def plot_confusion_matrix(y_true, y_pred, classes,
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

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
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
    plt.savefig(title+'.pdf')
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
    # TODO TEM DE SER FIXED PORQUE ISTO AGORA ESTA POR CAMADAS E NAO PERFIS
    uniques = df.cwrb_reference_soil_group.unique()
    for u in uniques:
        cnt = df[df.cwrb_reference_soil_group == u].shape[0]
        if cnt < min:
            df = df[df.cwrb_reference_soil_group != u]
            print('Deleting {} with {} occurrences'.format(u, cnt))

    return df


def get_data():
    inputfile = '../data/test/mexico_k_1_layers_5.csv'
    profile_file = '../data/profiles.csv'
    profiles_file = pd.read_csv(profile_file)
    profiles_file = profiles_file[['profile_id', 'cwrb_reference_soil_group']]
    data = pd.read_csv(inputfile)
    data = profiles_file.merge(data, how="inner", left_on=[
        'profile_id'], right_on=['profile_id'])

    data = remove_small_classes(data, 15)
    # data = scale_data(data)

    y = data.cwrb_reference_soil_group.astype(str)
    X = data.drop(['profile_id', 'cwrb_reference_soil_group'], axis=1)

    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    return X, y


# Actual Code
test_results = []
test_results_y_true = list()
test_results_y_pred = list()

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)


clf = XGBClassifier(n_estimators=300, min_child_weight=7,
                    gamma=0.2, subsample=0.8, colsample_bytree=0.8, n_jobs=-1)

res = cross_val_score(clf, X_train, y_train, cv=10, scoring=make_scorer(
    classification_report_with_accuracy_score))


print('Results --------------\n\n\n\n')
classification_report_with_accuracy_score(
    test_results_y_true, test_results_y_pred, print_score=True)

kappa = cohen_kappa_score(test_results_y_true, test_results_y_pred)
print('Fold accuracy', res, '\nAverage: ',
      np.mean(res), 'Kappa Score', kappa)
print(f"acc {str(np.mean(res))}, kappa: {kappa}")

# Unique classes sorted by their number of occurrences
labels = list(y.value_counts().index)
plot_confusion_matrix(test_results_y_true,
                      test_results_y_pred, classes=labels)


"""

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"kappa: {cohen_kappa_score(y_test, y_pred)}")


parameters = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
              "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
              "min_child_weight": [1, 3, 5, 7],
              "gamma": [0.0, 0.1, 0.25, 0.4],
              "colsample_bytree": [0.3, 0.4, 0.5, 0.7]}



              'gamma':[i/10.0 for i in range(0,5)]

parameters = {
    "learning_rate": [0.1],
    "n_estimators": [300],
    "max_depth": [5],
    "min_child_weight": [7],
    'gamma': [0.2],
    'subsample': [0.8],
    'colsample_bytree': [0.8]

}

clf = GridSearchCV(XGBClassifier(
    n_estimators=150, n_jobs=-1), parameters, cv=5, n_jobs=-1, verbose=2)
clf.fit(X_train, y_train)

print(clf.best_params_)

y_pred = clf.besta_estimator_.predict(X_test)

print(f"kappa: {cohen_kappa_score(y_test, y_pred)}")
"""

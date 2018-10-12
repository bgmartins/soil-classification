import numpy as np
import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn_pandas import DataFrameMapper, cross_val_score
#from gcforest.gcforest import GCForest

def get_gcforest_config(n_classes=10):
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = n_classes
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5, "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

def classification_report_with_accuracy_score(y_true, y_pred):
    print("==== REPORT OF RESULTS FOR ONE OF THE DATA FOLDS  ====")
    print(classification_report(y_true, y_pred))
    acc=accuracy_score(y_true, y_pred)
    return acc

table_y = pandas.read_csv("TAXNWRB_selection.csv", header=0)
table = pandas.read_csv("PROPS_selection.csv", header=0, dtype={col: np.float32 for col in list(['LATWGS84',
                                                                                                  'LONWGS84',
                                                                                                  'DEPTH',
                                                                                                  'UHDICM.f',
                                                                                                  'LHDICM.f',
                                                                                                  'DEPTH.f',
                                                                                                  'UHDICM',
                                                                                                  'LHDICM',
                                                                                                  'CRFVOL',
                                                                                                  'SNDPPT',
                                                                                                  'SLTPPT',
                                                                                                  'CLYPPT',
                                                                                                  'BLD',
                                                                                                  'PHIHOX',
                                                                                                  'PHIKCL',
                                                                                                  'ORCDRC',
                                                                                                  'CECSUM',
                                                                                                  'HZDTXT',
                                                                                                  'PHICAL'])})
table = table.merge(table_y, how="inner", left_on='CLEAN_ID', right_on='LOC_ID').fillna(0) # TODO: consider using fillna(0) instead
mapper = DataFrameMapper( [ ('WRB_2006_NAMEf', sklearn.preprocessing.LabelEncoder()),
							(['LONWGS84_x'], sklearn.preprocessing.StandardScaler()), 
							(['LATWGS84_x'], sklearn.preprocessing.StandardScaler()), 
							('DEPTH', None),
							('UHDICM.f', None),
							('LHDICM.f', None),
							('DEPTH.f', None),
							('UHDICM', None),
							('LHDICM', None),
							('CRFVOL', None),
							('SNDPPT', None),
							('SLTPPT', None),
							('CLYPPT', None),
							('BLD', None),
							('PHIHOX', None),
							('PHIKCL', None),
							('ORCDRC', None),
							('CECSUM', None) ], sparse=True )
classifier = sklearn.ensemble.RandomForestClassifier()
#classifier = GCForest(get_gcforest_config())
pipe = sklearn.pipeline.Pipeline( [ ('featurize', mapper), ('classify', classifier)] )
cross_val_score(pipe, X=table, y=table.WRB_2006_NAMEf, scoring=make_scorer(classification_report_with_accuracy_score), cv=10)

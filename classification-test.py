import numpy as np
import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn_pandas import DataFrameMapper, cross_val_score

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
pipe = sklearn.pipeline.Pipeline( [ ('featurize', mapper), ('rf', sklearn.ensemble.RandomForestClassifier())] )
cross_val_score(pipe, X=table, y=table.WRB_2006_NAMEf, scoring=make_scorer(classification_report_with_accuracy_score), cv=10)
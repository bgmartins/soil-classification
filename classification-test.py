import re
import numpy as np
import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
import inflect
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
                                                                                                  'PHICAL'])}, low_memory=False)

table = table.assign(CLEAN_ID = [ str(x).replace("ID_","") for x in table.LOC_ID ] )
#table_y = table_y.assign(SOILCLASS = [ re.compile('s$').sub('',re.sub('[()]', ' ',str(x)).strip().split(' ')[-1].lower().strip()) for x in table_y.TAXNWRB ] )
table_y['SOILCLASS'] = table_y['TAXNWRB.f'].apply(lambda x: x.split(" ")[1])
table = table.merge(table_y, how="inner", left_on='CLEAN_ID', right_on='LOC_ID')
table = table.dropna(subset=['DEPTH'])
mapper = DataFrameMapper( [ ('SOILCLASS', None), ('CLEAN_ID', None),
							('LONWGS84_x', None), 
							('LATWGS84_x', None), 
							(['DEPTH'], sklearn.preprocessing.KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='quantile')),
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
							('CECSUM', None) ], df_out=True)
newtable = mapper.fit_transform(table).pivot_table(columns=['DEPTH'],index=['CLEAN_ID','DEPTH'])
newtable.columns = [ re.compile('[^a-zA-Z0-9_]').sub('',''.join(str(col))) for col in newtable.columns.values ]
table = table[['CLEAN_ID','SOILCLASS']].drop_duplicates()
newtable = newtable.merge(table, how="inner", left_on='CLEAN_ID', right_on='CLEAN_ID')
table = newtable.dropna(subset=['SOILCLASS']).fillna(0)
mapper = DataFrameMapper( [ ('SOILCLASS', sklearn.preprocessing.LabelEncoder()),
                            (['LONWGS84_x00'], sklearn.preprocessing.StandardScaler()), 
                            (['LATWGS84_x00'], sklearn.preprocessing.StandardScaler()), 
                            ('UHDICMf00', None), ('UHDICMf10', None), ('UHDICMf20', None), ('UHDICMf30', None), ('UHDICMf40', None),
                            ('LHDICMf00', None), ('LHDICMf10', None), ('LHDICMf20', None), ('LHDICMf30', None), ('LHDICMf40', None),
                            ('DEPTHf00', None), ('DEPTHf10', None), ('DEPTHf20', None), ('DEPTHf30', None), ('DEPTHf40', None),
                            ('UHDICM00', None), ('UHDICM10', None), ('UHDICM20', None), ('UHDICM30', None), ('UHDICM40', None),
                            ('LHDICM00', None), ('LHDICM10', None), ('LHDICM20', None), ('LHDICM30', None), ('LHDICM40', None),
                            ('CRFVOL00', None), ('CRFVOL10', None), ('CRFVOL20', None), ('CRFVOL30', None), ('CRFVOL40', None),
                            ('SNDPPT00', None), ('SNDPPT10', None), ('SNDPPT20', None), ('SNDPPT30', None), ('SNDPPT40', None),
                            ('SLTPPT00', None), ('SLTPPT10', None), ('SLTPPT20', None), ('SLTPPT30', None), ('SLTPPT40', None),
                            ('CLYPPT00', None), ('CLYPPT10', None), ('CLYPPT20', None), ('CLYPPT30', None), ('CLYPPT40', None), 
                            ('BLD00', None), ('BLD10', None), ('BLD20', None), ('BLD30', None), ('BLD40', None),
                            ('PHIHOX00', None), ('PHIHOX10', None), ('PHIHOX20', None), ('PHIHOX30', None), ('PHIHOX40', None),
                            ('PHIKCL00', None), ('PHIKCL10', None), ('PHIKCL20', None), ('PHIKCL30', None), ('PHIKCL40', None),
                            ('ORCDRC00', None), ('ORCDRC10', None), ('ORCDRC20', None), ('ORCDRC30', None), ('ORCDRC40', None),
                            ('CECSUM00', None), ('CECSUM10', None), ('CECSUM20', None), ('CECSUM30', None), ('CECSUM40', None) ])
table_y = table_y['SOILCLASS'].value_counts()
print("Dataset features a total of " + repr(len(table_y)) + " soil classes.")
print(table_y)
print("Training and evaluating classifier through 10-fold cross-validation...")
classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
#classifier = GCForest(get_gcforest_config())
pipe = sklearn.pipeline.Pipeline( [ ('featurize', mapper), ('classify', classifier)] )
cross_val_score(pipe, X=table, y=table.SOILCLASS, scoring=make_scorer(classification_report_with_accuracy_score), cv=10)

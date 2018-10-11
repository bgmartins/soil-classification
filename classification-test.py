import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn_pandas import DataFrameMapper, cross_val_score

def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    return accuracy_score(y_true, y_pred)

table1 = pandas.read_csv("PROPS_selection.csv", header=True)
table2 = pandas.read_csv("TAXNWRB_selection.csv", header=True)
data = table1.merge(table2, how="inner", left_on='CLEAN_ID', right_on='LOC_ID')
mapper = DataFrameMapper( [ ('WRB_2006_NAMEf', sklearn.preprocessing.LabelEncoder()) ] ,
							(['LONWGS84'], sklearn.preprocessing.StandardScaler()), 
							(['LATWGS84'], sklearn.preprocessing.StandardScaler()), 
							('DEPTH', None) ] ,
							('UHDICM.f', None) ] ,
							('LHDICM.f', None) ] ,
							('DEPTH.f', None) ] ,
							('UHDICM', None) ] ,
							('LHDICM', None) ] ,
							('CRFVOL', None) ] ,
							('SNDPPT', None) ] ,
							('SLTPPT', None) ] ,
							('CLYPPT', None) ] ,
							('BLD', None) ] ,
							('PHIHOX', None) ] ,
							('PHIKCL', None) ] ,
							('ORCDRC', None) ] ,
							('CECSUM', None) ] , sparse=True )
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(classification_report_with_accuracy_score)}
pipe = sklearn.pipeline.Pipeline( [ ('featurize', mapper), ('rf', sklearn.ensemble.RandomForestClassifier())] )
cross_val_score(pipe, X=data, y=data.WRB_2006_NAMEf, scoring=scoring, cv=10)

import re
import numpy as np
import georasters as gr
import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
import inflect
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn_pandas import DataFrameMapper, cross_val_score
from fancyimpute import KNN
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
    pre_micro = precision_score(y_true, y_pred,average='micro')
    rec_micro = recall_score(y_true, y_pred,average='micro')
    f_1_micro = f1_score(y_true, y_pred,average='micro')
    pre_macro = precision_score(y_true, y_pred,average='macro')
    rec_macro = recall_score(y_true, y_pred,average='macro')
    f_1_macro = f1_score(y_true, y_pred,average='macro')
    pre_weighted = precision_score(y_true, y_pred,average='weighted')
    rec_weighted = recall_score(y_true, y_pred,average='weighted')
    f_1_weighted = f1_score(y_true, y_pred,average='weighted')
    aux = acc, pre_micro, rec_micro, f_1_micro, pre_macro, rec_macro, f_1_macro
    return acc

print("Reading information on soil classes...")
table_y = pandas.read_csv("TAXNWRB_selection.csv", header=0)
table_y['SOILCLASS'] = table_y['TAXNWRB.f'].apply(lambda x: x.split(" ")[1])

print("Reading information on land coverage...")
table_y["LANDCOV"] = "210"
NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
added = 0
table = gr.from_file("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
for index, row in table_y.iterrows():
    try: 
      val = table.map_pixel(row['LONWGS84'], row['LATWGS84'])
      added += 1
    except: val = None
    table_y.set_value(index,'LANDCOV',str(val))
print("Added land coverage information to " + repr(added) + " instances out of " + repr(len(table_y)) + " sample locations...")

print("Reading information on soil properties...")
table = pandas.read_csv("PROPS_selection.csv", header=0, dtype={col: np.float32 for col in list(['LATWGS84', 'LONWGS84', 'DEPTH', 'UHDICM.f', 'LHDICM.f', 'DEPTH.f', 'UHDICM', 'LHDICM', 'CRFVOL', 'SNDPPT', 'SLTPPT', 'CLYPPT', 'BLD', 'PHIHOX', 'PHIKCL', 'ORCDRC', 'CECSUM', 'PHICAL'])}, low_memory=False)
table = table.assign(CLEAN_ID = [ str(x).replace("ID_","") for x in table.LOC_ID ] )
table = table.merge(table_y, how="inner", left_on='CLEAN_ID', right_on='LOC_ID')

print("Grouping soil properties with basis on depth...")
#TODO: Check the validity of filling the missing values for DEPTH with basis on the average DEPTH value for the entire collection
table['DEPTH'].fillna((table['DEPTH'].mean()), inplace=True)
minvalue = table['DEPTH'].min() # Default value of 0
maxvalue = table['DEPTH'].max() # Default value of 200
print("Depth values ranging from " + repr(minvalue) + " to " + repr(maxvalue) + "...")
table['DEPTH'] = pandas.cut(table['DEPTH'], bins=[0 , 5, 15, 30, 60, 100, maxvalue + 0.01], right=True, labels=[0, 1, 2, 3, 4, 5])

mapper = DataFrameMapper( [ ('SOILCLASS', None), # Soil classification
                            ('LANDCOV', None), # Land coverage class from GlobCover
                            ('CLEAN_ID', None), # Unique identifier for each measurement point
                            ('LONWGS84_x', None), # Longitute coodinates for measurement points
                            ('LATWGS84_x', None), # Latitude coordinates for measurement points
                            ('DEPTH', None), # Depth of the measurement
                            ('UHDICM.f', None), #
                            ('LHDICM.f', None), #
                            ('DEPTH.f', None), # Depth of the measurement
                            ('UHDICM', None), # 
                            ('LHDICM', None), #
                            ('CRFVOL', None), # Coarse fragments volumetric in %
                            ('SNDPPT', None), # Sand content (50-2000 micro meter) mass fraction in %
                            ('SLTPPT', None), # Silt content (2-50 micro meter) mass fraction in %
                            ('CLYPPT', None), # Clay content (0-2 micro meter) mass fraction in %
                            ('BLD', None), # Bulk density (fine earth) in kg / cubic-meter
                            ('PHIHOX', None), # Soil pH x 10 in H2O
                            ('PHIKCL', None), # 
                            ('ORCDRC', None), # Soil organic carbon content (fine earth fraction) in permilles
                            ('CECSUM', None) ], df_out=True) # 
newtable = mapper.fit_transform(table)

print("Interpolating information for properties with missing values...")
for col in newtable.columns[newtable.isnull().any()]:
  print("Column " + col + " has missing values...")
  aux = newtable[['LONWGS84_x','LATWGS84_x','DEPTH.f',col]].values
  #aux = KNN(k=2).fit_transform(aux)
  #newtable[col] = aux[:3]
newtable = newtable.fillna(newtable.mean())

newtable = newtable.pivot_table(columns=['DEPTH'],index=['CLEAN_ID','DEPTH'])
newtable.columns = [ re.compile('[^a-zA-Z0-9_]').sub('',''.join(str(col))) for col in newtable.columns.values ]
table = table[['CLEAN_ID','SOILCLASS','LANDCOV']].drop_duplicates()
table = newtable.merge(table, how="inner", left_on='CLEAN_ID', right_on='CLEAN_ID')

for col in table.columns[table.isnull().any()]:
  newaux = table[['LONWGS84_x00','LATWGS84_x00','DEPTHf00',col]].values
  #newtable[col] = KNN(k=2).fit_transform(aux)[:3]
table = table.fillna(table.mean())

mapper = DataFrameMapper( [ ('SOILCLASS', sklearn.preprocessing.LabelEncoder()),
                            (['LANDCOV'], sklearn.preprocessing.OneHotEncoder()),
                            (['LONWGS84_x00'], sklearn.preprocessing.StandardScaler()), 
                            (['LATWGS84_x00'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf00'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf10'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf20'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf30'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf40'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf50'], sklearn.preprocessing.StandardScaler()),
                            (['LHDICMf00'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf10'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf20'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf30'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf40'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf50'], sklearn.preprocessing.StandardScaler()),
                            (['DEPTHf00'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf10'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf20'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf30'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf40'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf50'], sklearn.preprocessing.StandardScaler()),
                            (['UHDICM00'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM10'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM20'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM30'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM40'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM50'], sklearn.preprocessing.StandardScaler()),
                            (['LHDICM00'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM10'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM20'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM30'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM40'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM50'], sklearn.preprocessing.StandardScaler()),
                            (['CRFVOL00'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL10'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL20'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL30'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL40'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL50'], sklearn.preprocessing.StandardScaler()),
                            (['SNDPPT00'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT10'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT20'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT30'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT40'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT50'], sklearn.preprocessing.StandardScaler()),
                            (['SLTPPT00'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT10'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT20'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT30'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT40'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT50'], sklearn.preprocessing.StandardScaler()),
                            (['CLYPPT00'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT10'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT20'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT30'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT40'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT50'], sklearn.preprocessing.StandardScaler()),
                            (['BLD00'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD10'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD20'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD30'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD40'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD50'], sklearn.preprocessing.StandardScaler()),
                            (['PHIHOX00'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX10'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX20'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX30'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX40'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX50'], sklearn.preprocessing.StandardScaler()),
                            (['PHIKCL00'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL10'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL20'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL30'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL40'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL50'], sklearn.preprocessing.StandardScaler()),
                            (['ORCDRC00'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC10'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC20'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC30'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC40'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC50'], sklearn.preprocessing.StandardScaler()),
                            (['CECSUM00'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM10'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM20'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM30'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM40'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM50'], sklearn.preprocessing.StandardScaler()) ])
table_y = table_y['SOILCLASS'].value_counts()
print("Dataset features a total of " + repr(len(table_y)) + " soil classes.")
for i,v in table_y.iteritems(): print("\t" + i + " : " + repr(v))
print("Training and evaluating classifier through 10-fold cross-validation...")
#classifier = XGBClassifier(n_estimators=250)
#classifier = CatBoostClassifier()
classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=250)
#classifier = GCForest(get_gcforest_config())
pipe = sklearn.pipeline.Pipeline( [ ('featurize', mapper), ('classify', classifier)] )
aux = cross_val_score(pipe, X=table, y=table.SOILCLASS, scoring=make_scorer(classification_report_with_accuracy_score), cv=10)
print("Overall results...")
print(aux.mean())

print("Craining classification model on complete dataset...")
train_data = mapper.fit_transform(table)
classifier.fit(train_data, table.SOILCLASS)
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking within the classification model
print("Feature ranking:")
for f in range(X.shape[1]): print("\t %d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances within the classification model
plt.figure()
plt.title("Feature importances")
plt.bar(range(np.min( [ 10, X.shape[1] ] ), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(np.min( [ 10, X.shape[1] ] )), indices)
plt.xlim([-1, np.min( [ 10, X.shape[1] ] )])
plt.show()
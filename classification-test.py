import re
import numpy as np
import georasters as gr
import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
import inflect
from xgboost import XGBClassifier
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
table_y["LANDCOV"] = None 
NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
table = gr.from_file("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
for index, row in table_y.iterrows():
    try: val = table.map_pixel(row['LONWGS84'], row['LATWGS84'])
    except: val = None
    table_y.set_value(index,'LANDCOV',val)

print("Reading information on soil properties...")
table = pandas.read_csv("PROPS_selection.csv", header=0, dtype={col: np.float32 for col in list(['LATWGS84', 'LONWGS84', 'DEPTH', 'UHDICM.f', 'LHDICM.f', 'DEPTH.f', 'UHDICM', 'LHDICM', 'CRFVOL', 'SNDPPT', 'SLTPPT', 'CLYPPT', 'BLD', 'PHIHOX', 'PHIKCL', 'ORCDRC', 'CECSUM', 'PHICAL'])}, low_memory=False)
table = table.assign(CLEAN_ID = [ str(x).replace("ID_","") for x in table.LOC_ID ] )
table = table.merge(table_y, how="inner", left_on='CLEAN_ID', right_on='LOC_ID')

print("Grouping soil properties with basis on depth...")
#TODO: Check the validity of filling the missing values for DEPTH with basis on the average DEPTH value for the entire collection
table['DEPTH'].fillna((table['DEPTH'].mean()), inplace=True)
table['DEPTH'] = pandas.cut(table['DEPTH'], bins=[0, 5, 15, 30, 60, 100, 200], right=True, labels=[0, 1, 2, 3, 4, 5])

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
                            ('BLD', None), Bulk density (fine earth) in kg / cubic-meter
                            ('PHIHOX', None), # Soil pH x 10 in H2O
                            ('PHIKCL', None), # 
                            ('ORCDRC', None), # Soil organic carbon content (fine earth fraction) in permilles
                            ('CECSUM', None) ], df_out=True) # 
newtable = mapper.fit_transform(table)

print("Interpolating information for properties with missing values...")
for col in newtable.columns[newtable.isnull().any()]:
  aux = newtable[['LONWGS84_x','LATWGS84_x','DEPTH.f',col]].values
  #newtable[col] = KNN(k=2).fit_transform(aux)[:3]
newtable = newtable.fillna(newtable.mean())

newtable = newtable.pivot_table(columns=['DEPTH'],index=['CLEAN_ID','DEPTH'])
newtable.columns = [ re.compile('[^a-zA-Z0-9_]').sub('',''.join(str(col))) for col in newtable.columns.values ]
table = table[['CLEAN_ID','SOILCLASS']].drop_duplicates()
table = newtable.merge(table, how="inner", left_on='CLEAN_ID', right_on='CLEAN_ID')

for col in table.columns[table.isnull().any()]:
  newaux = table[['LONWGS84_x00','LATWGS84_x00','DEPTHf00',col]].values
  #newtable[col] = KNN(k=2).fit_transform(aux)[:3]
table = table.fillna(table.mean())

print(newtable.columns.values)

mapper = DataFrameMapper( [ ('SOILCLASS', sklearn.preprocessing.LabelEncoder()),
                            ('LANDCOV00', sklearn.preprocessing.OneHotEncoder()),
                            (['LONWGS84_x00'], sklearn.preprocessing.StandardScaler()), 
                            (['LATWGS84_x00'], sklearn.preprocessing.StandardScaler()), 
                            ('UHDICMf00', None), 
                            ('UHDICMf10', None), 
                            ('UHDICMf20', None), 
                            ('UHDICMf30', None), 
                            ('UHDICMf40', None), 
                            ('UHDICMf50', None),
                            ('LHDICMf00', None), 
                            ('LHDICMf10', None), 
                            ('LHDICMf20', None), 
                            ('LHDICMf30', None), 
                            ('LHDICMf40', None), 
                            ('LHDICMf50', None),
                            ('DEPTHf00', None), 
                            ('DEPTHf10', None), 
                            ('DEPTHf20', None), 
                            ('DEPTHf30', None), 
                            ('DEPTHf40', None), 
                            ('DEPTHf50', None),
                            ('UHDICM00', None), 
                            ('UHDICM10', None), 
                            ('UHDICM20', None), 
                            ('UHDICM30', None), 
                            ('UHDICM40', None), 
                            ('UHDICM50', None),
                            ('LHDICM00', None), 
                            ('LHDICM10', None), 
                            ('LHDICM20', None), 
                            ('LHDICM30', None), 
                            ('LHDICM40', None), 
                            ('LHDICM50', None),
                            ('CRFVOL00', None), 
                            ('CRFVOL10', None), 
                            ('CRFVOL20', None), 
                            ('CRFVOL30', None), 
                            ('CRFVOL40', None), 
                            ('CRFVOL50', None),
                            ('SNDPPT00', None), 
                            ('SNDPPT10', None), 
                            ('SNDPPT20', None), 
                            ('SNDPPT30', None), 
                            ('SNDPPT40', None), 
                            ('SNDPPT50', None),
                            ('SLTPPT00', None), 
                            ('SLTPPT10', None), 
                            ('SLTPPT20', None), 
                            ('SLTPPT30', None), 
                            ('SLTPPT40', None), 
                            ('SLTPPT50', None),
                            ('CLYPPT00', None), 
                            ('CLYPPT10', None), 
                            ('CLYPPT20', None), 
                            ('CLYPPT30', None), 
                            ('CLYPPT40', None), 
                            ('CLYPPT50', None),
                            ('BLD00', None), 
                            ('BLD10', None), 
                            ('BLD20', None), 
                            ('BLD30', None), 
                            ('BLD40', None), 
                            ('BLD50', None),
                            ('PHIHOX00', None), 
                            ('PHIHOX10', None), 
                            ('PHIHOX20', None), 
                            ('PHIHOX30', None), 
                            ('PHIHOX40', None), 
                            ('PHIHOX50', None),
                            ('PHIKCL00', None), 
                            ('PHIKCL10', None), 
                            ('PHIKCL20', None), 
                            ('PHIKCL30', None), 
                            ('PHIKCL40', None), 
                            ('PHIKCL50', None),
                            ('ORCDRC00', None), 
                            ('ORCDRC10', None), 
                            ('ORCDRC20', None), 
                            ('ORCDRC30', None), 
                            ('ORCDRC40', None), 
                            ('ORCDRC50', None),
                            ('CECSUM00', None), 
                            ('CECSUM10', None), 
                            ('CECSUM20', None), 
                            ('CECSUM30', None), 
                            ('CECSUM40', None), 
                            ('CECSUM50', None) ])
table_y = table_y['SOILCLASS'].value_counts()
print("Dataset features a total of " + repr(len(table_y)) + " soil classes.")
print(table_y)
print("Training and evaluating classifier through 10-fold cross-validation...")
classifier = XGBClassifier()
classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
#classifier = GCForest(get_gcforest_config())
pipe = sklearn.pipeline.Pipeline( [ ('featurize', mapper), ('classify', classifier)] )
aux = cross_val_score(pipe, X=table, y=table.SOILCLASS, scoring=make_scorer(classification_report_with_accuracy_score), cv=10)
print("Overall results...")
print(aux.mean())
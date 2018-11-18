import re
import numpy as np
import georasters as gr
import matplotlib.pyplot as plt
import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
import inflect
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.externals import joblib
#from gcforest.gcforest import GCForest

def get_gcforest_config(n_classes=30):
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = n_classes
    ca_config["estimators"] = []
    ca_config["estimators"].append({"n_folds": 5, "type": "XGBClassifier", "n_estimators": 250, "max_depth": 5, "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1} )
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 250, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 250, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config

test_results = None
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
    global test_results
    if test_results is None : test_results = [ ( acc, pre_micro, rec_micro, f_1_micro, pre_macro, rec_macro, f_1_macro ) ]
    else: test_results.append( ( acc, pre_micro, rec_micro, f_1_micro, pre_macro, rec_macro, f_1_macro ) )
    return acc

print("Reading information on soil classes...")
table_y = pandas.read_csv("TAXNWRB_selection.csv", header=0)
table_y['SOILCLASS'] = table_y['TAXNWRB.f'].apply(lambda x: x.split(" ")[1])
soil_class_encoder = sklearn.preprocessing.LabelEncoder()

print("Reading information on land coverage...")
table_y["LANDCOV"] = "210"
NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
raster_aux = dict()
table = gr.from_file("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
cnt = 0
for index, row in table_y.iterrows():
    try: 
      val = table.map_pixel(row['LONWGS84'], row['LATWGS84'])
      table_y.set_value(index,'LANDCOV',str(val))
      cnt += 1
    except: val = None
    if not(val is None):
      l = table.map_pixel_location(row['LONWGS84'], row['LATWGS84'])
      raster_aux[(l[0],l[1])] = table_y.loc[index,'LANDCOV']
land_coverage_encoder = sklearn.preprocessing.OneHotEncoder()
print("Added land coverage information to " + repr(cnt) + " instances out of " + repr(len(table_y)) + " sample locations...")
print("Land coverage information is available for " + repr(len(raster_aux)) + " unique raster cells...")

print("Reading information on soil properties...")
table = pandas.read_csv("PROPS_selection.csv", header=0, dtype={col: np.float32 for col in list(['LATWGS84', 'LONWGS84', 'DEPTH', 'UHDICM.f', 'LHDICM.f', 'DEPTH.f', 'UHDICM', 'LHDICM', 'CRFVOL', 'SNDPPT', 'SLTPPT', 'CLYPPT', 'BLD', 'PHIHOX', 'PHIKCL', 'ORCDRC', 'CECSUM', 'PHICAL'])}, low_memory=False)
table = table.assign(CLEAN_ID = [ str(x).replace("ID_","") for x in table.LOC_ID ] )
table = table.merge(table_y, how="inner", left_on='CLEAN_ID', right_on='LOC_ID')

print("Grouping soil properties with basis on depth...")
#TODO: Check the validity of filling the missing values for DEPTH with basis on the average DEPTH value for the entire collection
table['DEPTH'] = table['DEPTH'].fillna((table['DEPTH'].mean()))
minvalue = table['DEPTH'].min() # Default value of 0
maxvalue = table['DEPTH'].max() # Default value of 200
print("Depth values ranging from " + repr(minvalue) + " to " + repr(maxvalue) + "...")
#table['DEPTH'] = pandas.cut(table['DEPTH'], bins=[minvalue - 0.1 , 5, 15, 30, 60, 100, maxvalue + 0.01], right=True, labels=False)
table['DEPTH'] = pandas.qcut(table['DEPTH.f'], q=6, labels=False)

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

print("Interpolating information for properties with missing values over dataset with " + repr(newtable.shape[0]) + " instances...")
for col in newtable.columns[newtable.isnull().any()]:
  print("Column " + col + " has " + repr(table[col].isnull().sum()) + " missing values...")
  aux = newtable[['LONWGS84_x','LATWGS84_x','DEPTH.f',col]].dropna().values
  aux = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance').fit(aux[0:aux.shape[0],0:3],aux[0:aux.shape[0],3])
  for index, row in newtable.iterrows():
    if np.isnan(row[col]): newtable.loc[index,col] = aux.predict(row[['LONWGS84_x','LATWGS84_x','DEPTH.f']].values.reshape(1, -1))[0]
newtable = newtable.fillna(newtable.mean())

newtable = newtable.pivot_table(columns=['DEPTH'],index=['CLEAN_ID','DEPTH'])
newtable.columns = [ re.compile('[^a-zA-Z0-9_]').sub('',''.join(str(col))) for col in newtable.columns.values ]
table = table[['CLEAN_ID','SOILCLASS','LANDCOV']].drop_duplicates()
table = newtable.merge(table, how="inner", left_on='CLEAN_ID', right_on='CLEAN_ID')

for col in table.columns[table.isnull().any()]:
  print("Column " + col + " has " + repr(table[col].isnull().sum()) + " missing values in the " + repr(table.shape[0]) + " depth-aggregated records...")
  aux = table[['LONWGS84_x0','LATWGS84_x0',col]].dropna().values
#  aux = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance').fit(aux[0:aux.shape[0],0:2],aux[0:aux.shape[0],2])
#  for index, row in table.iterrows():
#    if np.isnan(row[col]): table.loc[index,col] = aux.predict(row[['LONWGS84_x0','LATWGS84_x0']].values.reshape(1, -1))[0]
table = table.fillna(table.mean())

mapper = DataFrameMapper( [ ('SOILCLASS', soil_class_encoder),
                            (['LANDCOV'], land_coverage_encoder),
                            (['LONWGS84_x0'], sklearn.preprocessing.StandardScaler()),
                            (['LATWGS84_x0'], sklearn.preprocessing.StandardScaler()),
                            (['UHDICMf0'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf1'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf2'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf3'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf4'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf5'], sklearn.preprocessing.StandardScaler()),
                            (['LHDICMf0'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf1'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf2'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf3'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf4'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf5'], sklearn.preprocessing.StandardScaler()),
                            (['DEPTHf0'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf1'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf2'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf3'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf4'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf5'], sklearn.preprocessing.StandardScaler()),
                            (['UHDICM0'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM1'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM2'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM3'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM4'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM5'], sklearn.preprocessing.StandardScaler()),
                            (['LHDICM0'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM1'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM2'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM3'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM4'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM5'], sklearn.preprocessing.StandardScaler()),
                            (['CRFVOL0'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL1'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL2'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL3'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL4'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL5'], sklearn.preprocessing.StandardScaler()),
                            (['SNDPPT0'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT1'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT2'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT3'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT4'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT5'], sklearn.preprocessing.StandardScaler()),
                            (['SLTPPT0'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT1'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT2'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT3'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT4'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT5'], sklearn.preprocessing.StandardScaler()),
                            (['CLYPPT0'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT1'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT2'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT3'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT4'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT5'], sklearn.preprocessing.StandardScaler()),
                            (['BLD0'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD1'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD2'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD3'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD4'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD5'], sklearn.preprocessing.StandardScaler()),
                            (['PHIHOX0'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX1'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX2'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX3'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX4'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX5'], sklearn.preprocessing.StandardScaler()),
                            (['PHIKCL0'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL1'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL2'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL3'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL4'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL5'], sklearn.preprocessing.StandardScaler()),
                            (['ORCDRC0'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC1'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC2'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC3'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC4'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC5'], sklearn.preprocessing.StandardScaler()),
                            (['CECSUM0'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM1'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM2'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM3'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM4'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM5'], sklearn.preprocessing.StandardScaler()) ])
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

print("Training classification model on complete dataset...")
train_data = mapper.fit_transform(table)
classifier.fit(train_data[0:train_data.shape[0],1:train_data.shape[1]], train_data[0:train_data.shape[0],0])
joblib.dump(classifier, 'classification-model.joblib') 

print("Infering the feature ranking within the classification model")
importances = classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
for f in range(train_data.shape[1] - 1): print("\t %d. feature %d (%s) (%f)" % (f + 1, indices[f], table.columns.values[indices[f] + 1], importances[indices[f]]))
# Plot the feature importances within the classification model
plt.figure()
plt.title("Feature importances")
plt.bar(range(np.min( [ 10, train_data.shape[1] - 1 ] ) ), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(np.min( [ 10, train_data.shape[1] - 1] ) ), indices)
plt.xlim([-1, np.min( [ 10, train_data.shape[1] - 1] )])
plt.show()

print("Generating a raster with the soil map...")
NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
raster = gr.from_file("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
for x in range(xsize):
  for y in range(ysize):
    if raster[x,y] == 210: 
      raster[x,y] = -1
      continue
    if (x,y) in raster_aux: raster[x,y] = land_coverage_encoder.transform( raster_aux[(x,y)] )
    else:
      land_cover = land_coverage_encoder.transform(raster[x,y])
      lon, lat = raster.map_pixel_inv(x, y, raster.x_cell_size, raster.y_cell_size, raster.xmin, raster.ymax)
      soil_class = -2
      raster[x,y] = soil_class
raster.to_tiff('soilmap.tif')

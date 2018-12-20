import os
import re
import numpy as np
import georasters as gr
import richdem as rd
import matplotlib.pyplot as plt
import pandas
import sklearn.preprocessing, sklearn.ensemble, sklearn.pipeline, sklearn.metrics
import inflect
from joblib import Parallel, delayed
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn.externals import joblib

test_results = [ ]
test_results_y_true = list()
test_results_y_pred = list()
def classification_report_with_accuracy_score(y_true, y_pred):
    print("==== REPORT OF RESULTS FOR ONE OF THE DATA FOLDS  ====")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
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
    global test_results_y_true
    global test_results_y_pred
    test_results_y_true += list(y_true)
    test_results_y_pred += list(y_pred)
    test_results.append( ( acc, pre_micro, rec_micro, f_1_micro, pre_macro, rec_macro, f_1_macro ) )
    return acc

print("Reading information on soil classes...")
table_y = pandas.read_csv("TAXNWRB_selection.csv", header=0).drop_duplicates()
table_y['SOILCLASS'] = table_y['TAXNWRB.f'].apply(lambda x: x.split(" ")[1])
table = table_y.drop(labels=['TIMESTRR'],axis=1).drop_duplicates()
table_y = table_y[['SOILCLASS','LOC_ID','LONWGS84','LATWGS84']].drop_duplicates()
soil_class_encoder = sklearn.preprocessing.LabelEncoder()
print("A total of " + repr(len(table_y)) + " instances with information on soil classes are available...")
print("The data instances refer to a total of " + repr(table_y['LOC_ID'].nunique()) + " unique locations...")
print("The data instances contain " + repr(table_y['LOC_ID'].nunique() - table['LOC_ID'].nunique()) + " locations with different classes at different time instants...")

print("Reading information on land coverage...")
table_y['LANDCOV'] = str("210_")
NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
raster_aux = dict()
table = gr.from_file("./globcover/GLOBCOVER_L4_200901_200912_V2.3.tif")
cnt = 0
for index, row in table_y.iterrows():
    try: 
      val = table.map_pixel(row['LONWGS84'], row['LATWGS84'])
      table_y.loc[index,'LANDCOV'] = str(val) + "_"
      cnt += 1
      l = table.map_pixel_location(row['LONWGS84'], row['LATWGS84'])
      raster_aux[(l[0],l[1])] = table_y.loc[index,'LANDCOV']
    except: table_y.loc[index,'LANDCOV'] = str("210_")
land_coverage_encoder = sklearn.preprocessing.OneHotEncoder()
print("Added land coverage information to " + repr(cnt) + " instances out of " + repr(len(table_y)) + " data instances...")
print("Land coverage information is available for " + repr(len(raster_aux)) + " unique raster cells...")

print("Reading information on terrain elevation...")
table_y['ELEVATION'] = 0
NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./globcover/digital-elevation-model.tif")
table = gr.from_file("./globcover/digital-elevation-model.tif")
cnt = 0
for index, row in table_y.iterrows():
    try: 
      val = table.map_pixel(row['LONWGS84'], row['LATWGS84'])
      table_y.loc[index,'ELEVATION'] = float(val)
      cnt += 1
    except: table_y.loc[index,'ELEVATION'] = 0
print("Added elevation information to " + repr(cnt) + " instances out of " + repr(len(table_y)) + " data instances...")

print("Computing information on terrain slope...")
for attr in ['slope_percentage' , 'aspect' , 'profile_curvature' ]:
  table_y[attr.upper()] = 0.0
  table = None
  table = rd.TerrainAttribute(rd.LoadGDAL("./globcover/digital-elevation-model.tif"), attrib=attr)
  rd.SaveGDAL("./slope.tif",table)
  table = None
  NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./slope.tif")
  table = gr.from_file("./slope.tif")
  cnt = 0
  for index, row in table_y.iterrows():
    try: 
      val = table.map_pixel(row['LONWGS84'], row['LATWGS84'])
      table_y.loc[index,attr.upper()] = float(val)
      cnt += 1
    except: table_y.loc[index,attr.upper()] = 0.0
  os.remove("./slope.tif")
  print("Added " + attr + " information to " + repr(cnt) + " instances out of " + repr(len(table_y)) + " data instances...")

print("Reading information on soil properties...")
table = pandas.read_csv("PROPS_selection.csv", header=0, dtype={col: np.float32 for col in list(['LATWGS84', 'LONWGS84', 'DEPTH', 'UHDICM.f', 'LHDICM.f', 'DEPTH.f', 'UHDICM', 'LHDICM', 'CRFVOL', 'SNDPPT', 'SLTPPT', 'CLYPPT', 'BLD', 'PHIHOX', 'PHIKCL', 'ORCDRC', 'CECSUM', 'PHICAL'])}, low_memory=False)
table = table.assign(CLEAN_ID = [ str(x).replace("ID_","") for x in table.LOC_ID ] )
tablefull = table.drop_duplicates()
table = table.merge(table_y, how="inner", left_on=['CLEAN_ID'], right_on=['LOC_ID']).drop_duplicates()

print("Grouping soil properties with basis on depth for " + repr(len(table)) + " instances...")
table['DEPTH'] = table['DEPTH'].fillna(table['DEPTH.f'])
table['DEPTH'] = table['DEPTH'].fillna((table['DEPTH'].mean()))
minvalue = table['DEPTH.f'].min() # Default value of 0
maxvalue = table['DEPTH.f'].max() # Default value of 200
print("Depth values ranging from " + repr(minvalue) + " to " + repr(maxvalue) + "...")
#table['DEPTH'] = pandas.cut(table['DEPTH.f'], bins=[minvalue , 5, 15, 30, 60, 100, 200, maxvalue], right=True, labels=False)
table['DEPTH'] = pandas.qcut(table['DEPTH.f'], q=3, labels=False)

mapper = DataFrameMapper( [ ('SOILCLASS', None), # Soil classification
                            ('LANDCOV', None), # Land coverage class from GlobCover
                            ('ELEVATION', None), # Terrain elevation from a DEM
                            ('SLOPE_PERCENTAGE', None), # Terrain slope from a DEM
                            ('ASPECT', None), # Terrain slope from a DEM
                            ('PROFILE_CURVATURE', None), # Terrain slope from a DEM
                            ('CLEAN_ID', None), # Unique identifier for each measurement point
                            ('TIMESTRR', None), # Date for the measurement
                            ('LONWGS84_x', None), # Longitute coodinates for measurement points
                            ('LATWGS84_x', None), # Latitude coordinates for measurement points
                            ('DEPTH', None), # Depth of the measurement
                            ('UHDICM.f', None), # Upper horizon depth
                            ('LHDICM.f', None), #  Lower horizon depth
                            ('DEPTH.f', None), # Depth of the measurement
                            ('UHDICM', None), # Upper horizon depth
                            ('LHDICM', None), # Lower horizon depth
                            ('CRFVOL', None), # Coarse fragments volumetric in %
                            ('SNDPPT', None), # Sand content (50-2000 micro meter) mass fraction in %
                            ('SLTPPT', None), # Silt content (2-50 micro meter) mass fraction in %
                            ('CLYPPT', None), # Clay content (0-2 micro meter) mass fraction in %
                            ('BLD', None), # Bulk density (fine earth) in kg / cubic-meter
                            ('PHIHOX', None), # Soil pH x 10 in H2O
                            ('PHIKCL', None), # pH medido numa solução de Potássio-Cloro (KCl)
                            ('ORCDRC', None), # Soil organic carbon content (fine earth fraction) in permilles
                            ('CECSUM', None) ], df_out=True) # 
newtable = mapper.fit_transform(table)

table = newtable[['CLEAN_ID','SOILCLASS','LANDCOV','ELEVATION','SLOPE_PERCENTAGE','ASPECT','PROFILE_CURVATURE','TIMESTRR','LONWGS84_x','LATWGS84_x']].drop_duplicates()
newtable = newtable.pivot_table(columns=['DEPTH'],index=['CLEAN_ID','TIMESTRR','LONWGS84_x','LATWGS84_x','ELEVATION','SLOPE_PERCENTAGE','ASPECT','PROFILE_CURVATURE'])
newtable.columns = [ re.compile('[^a-zA-Z0-9_]').sub('',''.join(str(col))) for col in newtable.columns.values ]
table = newtable.merge(table, how="inner", left_on=['CLEAN_ID','TIMESTRR','LONWGS84_x','LATWGS84_x'], right_on=['CLEAN_ID','TIMESTRR','LONWGS84_x','LATWGS84_x'])

tablefull['DEPTH'] = tablefull['DEPTH'].fillna(tablefull['DEPTH.f'])
tablefull['DEPTH'] = tablefull['DEPTH'].fillna((tablefull['DEPTH'].mean()))
tablefull['DEPTH'] = pandas.qcut(tablefull['DEPTH.f'], q=3, labels=False)
tablefull['LONWGS84_x'] = tablefull['LONWGS84']
tablefull['LATWGS84_x'] = tablefull['LATWGS84']
tablefull['SOILCLASS'] = 0
tablefull['LANDCOV'] = str("210_")
tablefull['ELEVATION'] = 0.0
tablefull['SLOPE_PERCENTAGE'] = 0.0
tablefull['ASPECT'] = 0.0
tablefull['PROFILE_CURVATURE'] = 0.0
newtable = mapper.transform(tablefull)
tablefull = newtable[['CLEAN_ID','TIMESTRR','LONWGS84_x','LATWGS84_x']].drop_duplicates()
newtable = newtable.pivot_table(columns=['DEPTH'],index=['CLEAN_ID','TIMESTRR','LONWGS84_x','LATWGS84_x'])
newtable.columns = [ re.compile('[^a-zA-Z0-9_]').sub('',''.join(str(col))) for col in newtable.columns.values ]
newtable = newtable.merge(tablefull, how="inner", left_on=['CLEAN_ID','TIMESTRR','LONWGS84_x','LATWGS84_x'], right_on=['CLEAN_ID','TIMESTRR','LONWGS84_x','LATWGS84_x'])

newtable = newtable.drop_duplicates()
table = table.drop_duplicates()
for col in table.columns[table.isnull().any()]:
  if col == 'TIMESTRR' : continue
  print("Column " + col + " has " + repr(table[col].isnull().sum()) + " missing values in the " + repr(len(table)) + " depth-aggregated records...")
  aux = newtable[['LONWGS84_x','LATWGS84_x',re.sub(r"([012])$", "\g<1>0", col)]].dropna().values
  aux = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance').fit(aux[0:aux.shape[0],0:2],aux[0:aux.shape[0],2])
  for index, row in table.iterrows():
    if np.isnan(row[col]):
      res = [ aux.predict(row[['LONWGS84_x','LATWGS84_x']].values.reshape(1, -1))[0] ]
      for i in [ '0' , '1' , '2' ]:
        if not(col.endswith(i)) and not(np.isnan(row[col[:-1] + i])): res.append(row[col[:-1] + i])
      table.at[index,col] = np.mean(res)
table = table.fillna(table.mean())
newtable = None

mapper = DataFrameMapper( [ ('SOILCLASS', soil_class_encoder),
                            (['ELEVATION'], sklearn.preprocessing.StandardScaler()),
                            (['SLOPE_PERCENTAGE'], sklearn.preprocessing.StandardScaler()),
                            (['ASPECT'], sklearn.preprocessing.StandardScaler()),
                            (['PROFILE_CURVATURE'], sklearn.preprocessing.StandardScaler()),
                            (['LONWGS84_x'], sklearn.preprocessing.StandardScaler()),
                            (['LATWGS84_x'], sklearn.preprocessing.StandardScaler()),
                            (['UHDICMf0'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICMf1'], sklearn.preprocessing.StandardScaler()),
                            (['UHDICMf2'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf0'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf1'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICMf2'], sklearn.preprocessing.StandardScaler()),
                            (['DEPTHf0'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf1'], sklearn.preprocessing.StandardScaler()), 
                            (['DEPTHf2'], sklearn.preprocessing.StandardScaler()),
                            (['UHDICM0'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM1'], sklearn.preprocessing.StandardScaler()), 
                            (['UHDICM2'], sklearn.preprocessing.StandardScaler()),
                            (['LHDICM0'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM1'], sklearn.preprocessing.StandardScaler()), 
                            (['LHDICM2'], sklearn.preprocessing.StandardScaler()),
                            (['CRFVOL0'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL1'], sklearn.preprocessing.StandardScaler()), 
                            (['CRFVOL2'], sklearn.preprocessing.StandardScaler()),
                            (['SNDPPT0'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT1'], sklearn.preprocessing.StandardScaler()), 
                            (['SNDPPT2'], sklearn.preprocessing.StandardScaler()),
                            (['SLTPPT0'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT1'], sklearn.preprocessing.StandardScaler()), 
                            (['SLTPPT2'], sklearn.preprocessing.StandardScaler()),
                            (['CLYPPT0'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT1'], sklearn.preprocessing.StandardScaler()), 
                            (['CLYPPT2'], sklearn.preprocessing.StandardScaler()),
                            (['BLD0'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD1'], sklearn.preprocessing.StandardScaler()), 
                            (['BLD2'], sklearn.preprocessing.StandardScaler()),
                            (['PHIHOX0'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX1'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIHOX2'], sklearn.preprocessing.StandardScaler()),
                            (['PHIKCL0'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL1'], sklearn.preprocessing.StandardScaler()), 
                            (['PHIKCL2'], sklearn.preprocessing.StandardScaler()),
                            (['ORCDRC0'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC1'], sklearn.preprocessing.StandardScaler()), 
                            (['ORCDRC2'], sklearn.preprocessing.StandardScaler()),
                            (['CECSUM0'], sklearn.preprocessing.StandardScaler()), 
                            (['CECSUM1'], sklearn.preprocessing.StandardScaler()),
                            (['CECSUM2'], sklearn.preprocessing.StandardScaler()),
                            (['LANDCOV'], land_coverage_encoder) ])
table_y = table['LANDCOV'].value_counts()
print("Dataset features a total of " + repr(len(table_y)) + " land cover classes.")
for i,v in table_y.iteritems(): print("\t" + i + " : " + repr(v))
table_y = table['SOILCLASS'].value_counts()
print("Dataset features a total of " + repr(len(table_y)) + " soil classes.")
for i,v in table_y.iteritems(): print("\t" + i + " : " + repr(v))

print("Training and evaluating classifier through 10-fold cross-validation...")
classifier = XGBClassifier(n_estimators=100, n_jobs=5)
classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=5)
pipe = sklearn.pipeline.Pipeline( [ ('featurize', mapper), ('classify', classifier)] )
aux = cross_val_score(pipe, X=table, y=table.SOILCLASS, scoring=make_scorer(classification_report_with_accuracy_score), cv=10)
print("Overall results...")
print("Accuracy : " + repr(aux.mean()))
classification_report_with_accuracy_score(test_results_y_true, test_results_y_pred)

print("Training classification model on complete dataset...")
train_data = mapper.fit_transform(table)
classifier.fit(train_data[0:train_data.shape[0],1:train_data.shape[1]], train_data[0:train_data.shape[0],0])
joblib.dump(classifier, 'classification-model.joblib') 

print("Infering the feature ranking within the classification model...")
if isinstance(classifier,XGBClassifier): 
  importances = classifier.get_booster().get_score(importance_type='weight') 
  importances = dict( [ (int(k.replace('f', '')), v) for k, v in importances.items() ] )
  importances = np.array( [ importances[k] for k in sorted(importances.keys(), reverse=False) ] )
  std = 0.0
else: 
  importances = classifier.feature_importances_
  std = np.std([tree.feature_importances_ for tree in classifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

for f in range(len(indices)):
  aux = indices[f] + 1
  if aux >= len(mapper.features) : aux = len(mapper.features) - 1
  print("\t %d. feature %d (%s) (%f)" % (f + 1, indices[f], mapper.features[aux][0][0], importances[indices[f]]))
print("Plotting the feature importances within the classification model...")
try:
  plt.figure()
  plt.title("Feature importances")
  plt.bar(range(np.min( [ 10, train_data.shape[1] - 1 ] ) ), importances[indices], color="r", yerr=std[indices], align="center")
  plt.xticks(range(np.min( [ 10, train_data.shape[1] - 1] ) ), indices)
  plt.xlim([-1, np.min( [ 10, train_data.shape[1] - 1] )])
  plt.show()
except: print("Error when plotting the feature importances...")

print("Generating a raster with the soil map...")
train_data = None
table = None
NDV, xsize, ysize, GeoT, Projection, DataType = gr.get_geo_info("./globcover/digital-elevation-model.tif")
raster = gr.from_file("./globcover/digital-elevation-model.tif")
def process_data( x ):
  for y in range(ysize - 1):    
    if raster.raster[x,y] == 210: 
      raster.raster[x,y] = -1
      continue
    if (x,y) in raster_aux: 
      aux = int(raster_aux[(x,y)])
      try: raster.raster[x,y] = soil_class_encoder.transform(np.array([aux]).reshape(1, -1))[0]
      except: raster.raster[x,y] = -1
    else:
      aux = str(int(raster.raster[x,y])) + "_"
      try: land_cover = land_coverage_encoder.transform([aux])[0]
      except: land_cover = land_coverage_encoder.transform(["210_"])[0]
      lon, lat = gr.map_pixel_inv(x, y, raster.x_cell_size, raster.y_cell_size, raster.xmin, raster.ymax)
      soil_class = -2 #TODO: Add classification result
      raster.raster[x,y] = soil_class
#Parallel(n_jobs=5)(delayed(process_data)(x,y) for x in range(xsize - 1))
#raster.to_tiff('soilmap.tif')

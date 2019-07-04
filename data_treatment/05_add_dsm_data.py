
import os
import sys
import getopt
import numpy as np
import pandas as pd
import richdem as rd
import georasters as gr
import matplotlib.pyplot as plt


def add_elevation(df, file, indexes):
    try:
        table = gr.from_file(file)

        for index in indexes:
            try:
                row = df.loc[index]
                val = table.map_pixel(row['lon'], row['lat'])
                df.loc[index, 'elevation'] = float(val)
            except:
                df.loc[index, 'elevation'] = -9999
    except:
        for index in indexes:
            df.loc[index, 'elevation'] = -9999
    return df


def add_slope_aspect_curvature(df, file, indexes):
    for attr in ['slope_percentage', 'aspect', 'profile_curvature']:
        table = None
        try:
            table = rd.TerrainAttribute(
                rd.LoadGDAL(file, no_data=-9999), attrib=attr)
            rd.SaveGDAL("./temp.tif", table)
            table = None
            table = gr.from_file("./temp.tif")
            for index in indexes:
                try:
                    row = df.loc[index]
                    val = table.map_pixel(row['lon'], row['lat'])
                    df.loc[index, attr] = float(val)
                except:
                    df.loc[index, attr] = np.nan
            os.remove("./temp.tif")
        except:
            for index in indexes:
                df.loc[index, attr] = np.nan
    return df


# Set variables and defaults
inputfile = '../data/test/mexico_k_1_layers_5.csv'
outputfile = '../data/test/mexico_k_1_layers_5_dsm.csv'
dsm_folder = '../data/rasters'
profile_file = '../data/profiles.csv'

h = '05_add_dsm_data.py -h <help> -i <inputfile> -o <outputfile> -d <dsm files folder>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:d:")
except getopt.GetoptError:
    print(h)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print(h)
        sys.exit()
    elif opt in ('-i'):
        inputfile = arg
    elif opt in ('-o'):
        outputfile = arg
    elif opt in ('-d'):
        dsm_folder = arg


df = pd.read_csv(inputfile)

profiles = pd.read_csv(profile_file)
profiles = profiles[['profile_id', 'latitude', 'longitude']]
profiles.columns = ['profile_id', 'lat', 'lon']

df_temp = profiles.merge(df, how="inner", left_on=[
    'profile_id'], right_on=['profile_id'])

needed_files = {}

for index, row in df_temp.iterrows():
    # if index>50:
    #    break
    key = '../data/rasters/test/N{:03}W{:03}_AVE_DSM.tif'.format(
        abs(int(row['lat'])), abs(int(row['lon'])))
    if key not in needed_files:
        needed_files[key] = [index]
    else:
        needed_files.get(key).append(index)

cnt = 0
for file in needed_files.keys():
    df_temp = add_elevation(df_temp, file, needed_files.get(file))
    df_temp = add_slope_aspect_curvature(df_temp, file, needed_files.get(file))
    cnt += len(needed_files.get(file))
    print("Added terrain information to " + repr(cnt) +
          " instances out of " + repr(len(df_temp)) + " data instances...")


# Merge back
df_temp = df_temp[['profile_id', 'elevation', 'slope_percentage',
                   'aspect', 'profile_curvature']]

df = df.merge(df_temp, how='inner', left_on=[
    'profile_id'], right_on=['profile_id'])

df.to_csv(outputfile, index=False)

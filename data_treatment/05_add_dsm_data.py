
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
                val = table.map_pixel(row['long'], row['lat'])
                df.loc[index, 'elevation'] = float(val)
            except:
                df.loc[index, 'elevation'] = 0
    except:
        for index in indexes:
            df.loc[index, 'elevation'] = 0
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
                    df.loc[index, attr] = 0
            os.remove("./temp.tif")
        except:
            for index in indexes:
                df.loc[index, attr] = 0
    return df


# Set variables and defaults
inputfile = '../data/test/mexico_k_1_layers_5.csv'
outputfile = '../data/mexico_k_1_layers_5_dsm.csv'
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
needed_files = {}

profiles = pd.read_csv(profile_file)
profiles = profiles[['profile_id', 'latitude', 'longitude']]
profiles.columns = ['profile_id', 'lat', 'lon']

df = profiles.merge(df, how="inner", left_on=[
    'profile_id'], right_on=['profile_id'])

# Fill the needed files for generating terrain data
for index, row in df.iterrows():
    key = '{}/N{:03}W{:03}_AVE_DSM.tif'.format(
        dsm_folder, abs(int(row['lat'])), abs(int(row['lon'])))
    if key not in needed_files:
        needed_files[key] = [index]
    else:
        needed_files.get(key).append(index)

# Fill the terrain data
cnt = 0
for file in needed_files.keys():
    df = add_elevation(df, file, needed_files.get(file))
    df = add_slope_aspect_curvature(df, file, needed_files.get(file))
    cnt += len(needed_files.get(file))
    print("Added terrain information to " + repr(cnt) +
          " instances out of " + repr(len(df)) + " data instances...")


df.to_csv(outputfile, index=False)

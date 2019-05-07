
import sys
import getopt
import pandas as pd
from multiprocessing import Pool
import time
import logging


# Receives an array of layers of a profile and returns a single row
def merge_profile(layers):
    default_depths = [2.5, 10, 22.5, 45, 80, 150]
    final_row = pd.DataFrame()
    for depth in default_depths:
        # Find the layer for current depth
        row = layers.loc[(layers['lower_depth'] > depth)].head(1)

        # Default to the last layer
        if row.empty:
            row = layers.iloc[[-1]]

        # Fix u/l_depth
        #row = row.assign(upper_depth=depth)
        #row = row.assign(lower_depth=depth+layer_depth)

        # Append columns
        if final_row.empty:
            final_row = row
        else:
            final_row = final_row.merge(
                row, how='inner', on='profile_id', suffixes=('', '_{}'.format(str(depth))))
    # Add a columns describing the total number of layers
    final_row['n_layers'] = len(layers)
    return final_row


# Set variables and defaults
inputfile = '../data/imputed_full_classified_data.csv'
outputfile = '../data/depth_merged_data.csv'

h = '03_merge_standard.py -h <help> -i <inputfile> -o <outputfile> '

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:")
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


d = pd.read_csv(inputfile)
rows = pd.DataFrame()
profile_ids = d.profile_id.unique()

start_time = time.time()

# Create an array for all the layers in each profile
profiles = []
for i, id in enumerate(profile_ids):
    # Find the layers for this profile and sort them
    layers = d[d['profile_id'] == id]
    layers = layers.sort_values(by=['lower_depth'])
    profiles.append(layers)


print('Merging {} layers of {} profiles'.format(
    d.shape[0], len(profiles)))

# Multiprocessing
with Pool() as pool:
    rows = pd.concat(
        pool.map(merge_profile, profiles), sort=False)

# Fix naming

rows = rows.drop(columns=list(
    rows.loc[:, rows.columns.str.contains('profile_id_')]))

duration = time.time() - start_time
print(f"Duration {duration} seconds")

rows.to_csv(outputfile, index=False)

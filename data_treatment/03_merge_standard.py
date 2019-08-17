
import sys
import getopt
import pandas as pd
from multiprocessing import Pool
import time
import logging


def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def merge_profile_by_most_present(layers):
    # Uses the most predominant layer for each depth
    default_depths = [0, 5, 15, 30, 60, 100, 200]
    final_row = pd.DataFrame()
    for i in range(1, len(default_depths)):
        overlap = 0.
        row = pd.DataFrame()

        j = 0
        for _, l in layers.iterrows():
            # Find the overlap layer to use as weight
            temp_overlap = getOverlap([l._ud, l._ld], [
                default_depths[i-1], default_depths[i]])
            if temp_overlap > overlap:
                overlap = temp_overlap
                row = layers.iloc[[j]]
            j += 1

        if row.empty:
            row = layers.iloc[[-1]]

        if final_row.empty:
            final_row = row
        else:
            final_row = final_row.merge(
                row, how='inner', on='profile_id', suffixes=('', '_{}'.format(str(default_depths[i-1]))))

    final_row['n_layers'] = len(layers)
    return final_row


def merge_profile(layers):
    # Receives an array of layers of a profile and returns a single row
    default_depths = [2.5, 10, 22.5, 45, 80, 150]
    final_row = pd.DataFrame()
    for depth in default_depths:
        # Find the layer for current depth
        row = layers.loc[(layers['_ld'] > depth)].head(1)

        # Default to the last layer
        if row.empty:
            row = layers.iloc[[-1]]

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
inputfile = '../data/mexico_k_1.csv'
outputfile = '../data/standard_merged_data.csv'

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
rows = rows.drop(columns=list(
    rows.loc[:, rows.columns.str.contains('_ud')]))
rows = rows.drop(columns=list(
    rows.loc[:, rows.columns.str.contains('_ld')]))

duration = time.time() - start_time
print(f"Duration {duration} seconds")

rows.to_csv(outputfile, index=False)

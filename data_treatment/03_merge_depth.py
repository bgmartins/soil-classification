
import sys
import getopt
import pandas as pd
from multiprocessing import Pool
import time
import logging


# Receives an array of layers of a profile and returns a single row
def merge_profile(layers):
    final_row = pd.DataFrame()
    for depth in range(0, max_depth+1, layer_depth):
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


def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def merge_profile_weighted(layers):
    u_depth = 0
    final_row = pd.DataFrame()
    for l_depth in range(layer_depth, max_depth+1, layer_depth):
        row = pd.DataFrame()
        total_overlap = 0

        # Find each layer in our profile that belongs to the final layer
        for _, l in layers.iterrows():
            # Find the overlap layer to use as weight
            overlap = getOverlap([l._ud, l._ld], [
                u_depth, l_depth]) / layer_depth
            if overlap != 0:
                # Multiply by the weight and add to the row
                total_overlap += overlap
                if(row.empty):
                    row = l.apply(lambda x: x * overlap)
                else:
                    row = row.add(l.apply(lambda x: x * overlap))

        if row.empty:
            # Default to the last layer in the profile
            row = layers.tail(1)
        else:
            if total_overlap != 1:
                # If we got to the end but there is still part of the layer to be described
                # we use the remaining weight and multiply by the last layer
                row = row.add(l.apply(
                    lambda x: x * (1-total_overlap)))

            # Transform back to a DataFrame for easier handling
            row = pd.DataFrame(data=[row], columns=list(layers.columns))

        # Fix u/l_depth
        #row = row.assign(upper_depth=u_depth)
        #row = row.assign(lower_depth=l_depth)

        # Append columns
        if final_row.empty:
            final_row = row
        else:
            # To avoid floating point errors, dont even ask
            row = row.assign(profile_id=list(final_row['profile_id'])[0])
            final_row = final_row.merge(
                row, how='inner', on='profile_id', suffixes=('', '_{}'.format(str(u_depth))))

        u_depth = l_depth

    final_row = final_row.assign(n_layers=len(layers))
    return final_row


# Set variables and defaults
inputfile = '../data/imputed_full_classified_data.csv'
outputfile = '../data/depth_merged_data.csv'
layer_depth = 50
max_depth = 200

h = '03_merge_depth.py -h <help> -i <inputfile> -o <outputfile> -d <depth for each layer> -m <max depth>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:d:m:")
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
        layer_depth = int(arg)
    elif opt in ('-m'):
        max_depth = int(arg)


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


print('Merging {} layers of {} profiles, each containing {} layers of depth {} in the end.'.format(
    d.shape[0], len(profiles), int(max_depth/layer_depth), layer_depth))

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

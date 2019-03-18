
import sys
import getopt
import pandas as pd
from multiprocessing import Pool
import time


# Receives an array of layers of a profile and returns a single row
def merge_profile(layers):
    final_row = pd.DataFrame()
    for i in range(0, n_layers):
        # Find the current layer
        row = layers.iloc[[i]] if len(layers) > i else layers.tail(1)

        # Rename the layer
        row = row.rename(columns=lambda x: x + "_" + str(i))

        # Append columns
        if final_row.empty:
            final_row = row
        else:
            final_row = final_row.merge(row, how="inner", left_on=[
                final_row.columns.values[0]], right_on=[row.columns.values[0]])
    # Add a columns describing the total number of layers
    final_row['n_layers'] = len(layers)
    return final_row


# Set variables and defaults
inputfile = '../data/imputed_full_classified_data.csv'
outputfile = '../data/layer_merged_data.csv'
n_layers = 4

h = '03_merge_layers.py -h <help> -i <inputfile> -o <outputfile> -l <number of layers>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:l:")
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
    elif opt in ('-l'):
        n_layers = int(arg)

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

print('Merging {} layers of {} profiles, each containing {} layers in the end.'.format(
    d.shape[0], len(profiles), n_layers))

# Multiprocessing
with Pool() as pool:
    rows = pd.concat(pool.map(merge_profile, profiles))

# Fix naming
rows['profile_id'] = rows['profile_id_0']
rows = rows.drop(columns=list(
    rows.loc[:, rows.columns.str.contains('profile_id_')]))

duration = time.time() - start_time
print(f"Duration {duration} seconds")

rows.to_csv(outputfile, index=False)

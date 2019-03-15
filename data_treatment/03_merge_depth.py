
import sys
import getopt
import pandas as pd


# Set variables and defaults
inputfile = '../data/imputed_full_classified_data.csv'
outputfile = '../data/depth_merged_data.csv'
layer_depth = 50
max_depth = 150

# TODO
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

for i, id in enumerate(profile_ids):
    if i % 100 == 0:
        print('Merging profile {}/{}'.format(i, len(profile_ids)))
    # Find the layers for this profile and sort them
    layers = d[d['profile_id'] == id]
    layers = layers.sort_values(by=['lower_depth'])

    final_row = pd.DataFrame()
    for depth in range(0, max_depth+1, layer_depth):
        # Find the layer for current depth
        row = layers.loc[(layers['lower_depth'] > depth)].head(1)

        # Default to the last layer
        if row.empty:
            row = layers.iloc[[-1]]

        # Rename the layer
        row = row.rename(columns=lambda x: x + "_" + str(depth))

        # Append columns
        if final_row.empty:
            final_row = row
        else:
            final_row = final_row.merge(row, how="inner", left_on=[
                final_row.columns.values[0]], right_on=[row.columns.values[0]])
    final_row['n_layers'] = len(layers)
    rows = rows.append(final_row, sort=False)

# Fix naming
rows['profile_id'] = rows['profile_id_0']
rows = rows.drop(columns=list(
    rows.loc[:, rows.columns.str.contains('profile_id_')]))


rows.to_csv(outputfile, index=False)

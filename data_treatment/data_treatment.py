import sys
import getopt
import pandas as pd
from fancyimpute import KNN
from multiprocessing import Pool


# Receives an array of layers of a profile and returns a single row
def merge_profile_depth(layers):
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
    # Add a columns describing the total number of layers
    final_row['n_layers'] = len(layers)
    return final_row


# Receives an array of layers of a profile and returns a single row
def merge_profile_layers(layers):
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
input_profiles = '../data/profiles.csv'
input_layers = '../data/layers.csv'
outputfile = '../data/classified_data.csv'
threshold = 0.01
h = 'data_treatment.py -h <help> -p <profile file> -l <layers file> -t <threshold to remove columns(default 0.01)> -c <filter country> -k <impute with k neighbours> -d <merge with depth d> -m <merge by m layers> -o <outputfile>'
country_filter = ''
knn = 0
layer_depth = 0
max_depth = 150
n_layers = 0

# Read arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:l:t:o:c:k:d:m:")
except getopt.GetoptError:
    print(h)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print(h)
        sys.exit()
    elif opt in ('-p'):
        input_profiles = arg
    elif opt in ('-l'):
        input_layers = arg
    elif opt in ('-t'):
        threshold = float(arg)
    elif opt in ('-o'):
        outputfile = arg
    elif opt in ('-c'):
        country_filter = arg
    elif opt in ('-k'):
        knn = int(arg)
    elif opt in ('-d'):
        layer_depth = int(arg)
    elif opt in ('-m'):
        n_layers = int(arg)


profiles = pd.read_csv(input_profiles)
layers = pd.read_csv(input_layers)

print('Merging and cleaning data.')

# Selecting only those classified using WRB
classified_profiles = profiles[~profiles['cwrb_reference_soil_group'].isnull()]

# Merging of the files
classified_data = layers.merge(classified_profiles, how="inner", left_on=[
                               'profile_id'], right_on=['profile_id'])

# Filter by country
if country_filter != '':
    classified_data = classified_data.loc[classified_data['country_id']
                                          == country_filter]

# Replace Albeluvisols with Retisols as was made in the latest version of WRB
classified_data.loc[classified_data['cwrb_reference_soil_group']
                    == 'Albeluvisols', 'cwrb_reference_soil_group'] = 'Retisols'

# Drop columns that are not numeric / not necessary (licenses)
classified_data = classified_data.drop(columns=['dataset_id', 'country_id', 'cfao_version', 'cfao_major_group',
                                                'cwrb_version', 'cwrb_reference_soil_group', 'cstx_version', 'cstx_order_name', 'translated'])
classified_data = classified_data.drop(columns=list(
    classified_data.loc[:, classified_data.columns.str.contains('license')]))

# Remove the columns that have only missing values
classified_data = classified_data.dropna(axis=1, how='all')

# Remove all columns with more null values than allowed by the threshold
c = classified_data.count().apply(lambda x: x/classified_data.shape[0])
classified_data = classified_data.drop(columns=c[c < threshold].index.tolist())

print('Cleaned data has {} rows and {} columns'.format(
    classified_data.shape[0], classified_data.shape[1]) + (', filtered by {}'.format(country_filter) if country_filter != '' else ''))

# Perform imputation
if knn != 0:
    print('\n\nImputing values with knn: {}, with {} rows and {} columns.\n\n'.format(
        knn, classified_data.shape[0], classified_data.shape[1]))

    # Impute the missing values
    classified_data = pd.DataFrame(data=KNN(k=knn).fit_transform(
        classified_data), columns=classified_data.columns, index=classified_data.index)


# Merge
if n_layers != 0 or layer_depth != 0:
    rows = pd.DataFrame()
    profile_ids = classified_data.profile_id.unique()

    # Create an array for all the layers in each profile
    profiles = []
    for i, id in enumerate(profile_ids):
        # Find the layers for this profile and sort them
        layers = classified_data[classified_data['profile_id'] == id]
        layers = layers.sort_values(by=['lower_depth'])
        profiles.append(layers)

    # Multiprocessing
    with Pool() as pool:
        if n_layers != 0:
            print('Merging {} layers of {} profiles, each containing {} layers in the end.'.format(
                classified_data.shape[0], len(profiles), n_layers))
            classified_data = pd.concat(
                pool.map(merge_profile_layers, profiles))
        elif layer_depth != 0:
            print('Merging {} layers of {} profiles, each containing {} layers of depth {} in the end.'.format(
                classified_data.shape[0], len(profiles), int(max_depth/layer_depth), layer_depth))
            classified_data = pd.concat(
                pool.map(merge_profile_depth, profiles))

    # Fix naming
    classified_data['profile_id'] = classified_data['profile_id_0']
    classified_data = classified_data.drop(columns=list(
        classified_data.loc[:, classified_data.columns.str.contains('profile_id_')]))


print('Final data has {} rows and {} columns.'.format(
    classified_data.shape[0], classified_data.shape[1]))
classified_data.to_csv(outputfile, index=False)

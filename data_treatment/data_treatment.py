import sys
import getopt
import pandas as pd
from fancyimpute import KNN
from multiprocessing import Pool
from sklearn.preprocessing import scale


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


# Receives an array of layers of a profile and returns a single row
def merge_profile_layers(layers):
    final_row = pd.DataFrame()
    for i in range(0, n_layers):
        # Find the current layer
        row = layers.iloc[[i]].copy() if len(
            layers) > i else layers.tail(1).copy()

        """ # PARA FAZER SEM A ULTIMA CAMADA E COM -9999 EM TUDO
        if not(len(layers) > i):
            for col in row.columns:
                if col != 'profile_id':
                    row[col] = -9999
        """
        # Append columns
        if final_row.empty:
            final_row = row
        else:
            final_row = final_row.merge(
                row, how='inner', on='profile_id', suffixes=('', '_{}'.format(str(i))))
    # Add a columns describing the total number of layers
    final_row = final_row.assign(n_layers=len(layers))
    return final_row


# Set variables and defaults
input_profiles = '../data/profiles.csv'
input_layers = '../data/layers.csv'
input_file = ''
outputfile = '../data/classified_data.csv'
threshold = 0.01
h = 'data_treatment.py -h <help> -i <input file> -p <profile file> -l <layers file> -t <threshold to remove columns(default 0.01)> -c <filter country> -k <impute with k neighbours> -d <merge with depth d> -m <merge by m layers> -o <outputfile>'
country_filter = ''
knn = 0
layer_depth = 0
max_depth = 200
n_layers = 0

# Read arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:p:l:t:o:c:k:d:m:")
except getopt.GetoptError:
    print(h)
    sys.exit(2)

for opt, arg in opts:
    if opt == '-h':
        print(h)
        sys.exit()
    elif opt in ('-p'):
        input_profiles = arg
    elif opt in ('-i'):
        input_file = arg
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

if input_file == '':
    profiles = pd.read_csv(input_profiles)
    layers = pd.read_csv(input_layers)

    print('Merging and cleaning data.')

    # Selecting only those classified using WRB
    classified_profiles = profiles[~profiles['cwrb_reference_soil_group'].isnull(
    )]

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
                                                    'cwrb_version', 'cwrb_reference_soil_group', 'cstx_version',
                                                    'cstx_order_name', 'translated', 'profile_layer_id'])
    classified_data = classified_data.drop(columns=list(
        classified_data.loc[:, classified_data.columns.str.contains('license')]))

    # Remove the columns that have only missing values
    classified_data = classified_data.dropna(axis=1, how='all')

    # Remove all columns with more null values than allowed by the threshold
    c = classified_data.count().apply(lambda x: x/classified_data.shape[0])
    classified_data = classified_data.drop(
        columns=c[c < threshold].index.tolist())

    # Remove rows where lower depth is clearly outlier
    classified_data = classified_data[classified_data.lower_depth < 999]

    # Add Thickness
    classified_data['thickness'] = classified_data.apply(
        lambda row: row['lower_depth'] - row['upper_depth'], axis=1)

    print('Cleaned data has {} rows and {} columns'.format(
        classified_data.shape[0], classified_data.shape[1]) + (', filtered by {}'.format(country_filter) if country_filter != '' else ''))
else:
    classified_data = pd.read_csv(input_file)


# Perform imputation
if knn != 0:
    ids = classified_data.profile_id
    ud = classified_data.upper_depth
    ld = classified_data.lower_depth

    classified_data.drop(['profile_id'], axis=1)

    # Normalizes and standardizes data
    classified_data = pd.DataFrame(data=scale(
        classified_data), columns=classified_data.columns, index=classified_data.index)

    print('\n\nImputing values with knn: {}, with {} rows and {} columns.\n\n'.format(
        knn, classified_data.shape[0], classified_data.shape[1]))

    # Impute the missing values
    classified_data = pd.DataFrame(data=KNN(k=knn).fit_transform(
        classified_data), columns=classified_data.columns, index=classified_data.index)

    classified_data['profile_id'] = ids
    classified_data['_ud'] = ud
    classified_data['_ld'] = ld


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
                pool.map(merge_profile_weighted, profiles))

    # Fix some columns
    classified_data = classified_data.drop(columns=list(
        classified_data.loc[:, classified_data.columns.str.contains('profile_id_')]))
    classified_data = classified_data.drop(columns=list(
        classified_data.loc[:, classified_data.columns.str.contains('latitude_')]))
    classified_data = classified_data.drop(columns=list(
        classified_data.loc[:, classified_data.columns.str.contains('longitude_')]))
    classified_data = classified_data.drop(columns=list(
        classified_data.loc[:, classified_data.columns.str.contains('_ud')]))
    classified_data = classified_data.drop(columns=list(
        classified_data.loc[:, classified_data.columns.str.contains('_ld')]))


print('Final data has {} rows and {} columns.'.format(
    classified_data.shape[0], classified_data.shape[1]))
classified_data.to_csv(outputfile, index=False)

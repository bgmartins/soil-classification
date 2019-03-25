'''
Import data from the csv files provided and obtain only the data classified using WRB
We also perform a small fix to the data where Albeluvisols are transformed to Retisols as is described in the 
latest version of WRB
'''
import sys
import getopt
import pandas as pd

# Set variables and defaults
input_profiles = '../data/profiles.csv'
input_layers = '../data/layers.csv'
outputfile = '../data/classified_data.csv'
threshold = 0.01
h = '01_create_classified_set.py -h <help> -p <profile file> -l <layers file> -t <threshold to remove columns(default 0.01)> -c <filter country> -o <outputfile> '
country_filter = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:l:t:o:c:")
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


profiles = pd.read_csv(input_profiles)
layers = pd.read_csv(input_layers)

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

print('Final data has {} rows and {} columns'.format(
    classified_data.shape[0], classified_data.shape[1]))

classified_data.to_csv(outputfile, index=False)

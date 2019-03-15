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
h = '01_create_classified_set.py -h <help> -p <profile file> -l <layers file> -o <outputfile> '

try:
    opts, args = getopt.getopt(sys.argv[1:], "hp:l:o:")
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
    elif opt in ('-o'):
        outputfile = arg

profiles = pd.read_csv(input_profiles)
layers = pd.read_csv(input_layers)
classified_profiles = profiles[~profiles['cwrb_reference_soil_group'].isnull()]
classified_data = layers.merge(classified_profiles, how="inner", left_on=[
                               'profile_id'], right_on=['profile_id'])
classified_data.loc[classified_data['cwrb_reference_soil_group']
                    == 'Albeluvisols', 'cwrb_reference_soil_group'] = 'Retisols'

classified_data.to_csv(outputfile, index=False)

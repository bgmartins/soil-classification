'''
Import data from the csv files provided and obtain only the data classified using WRB
We also perform a small fix to the data where Albeluvisols are transformed to Retisols as is described in the 
latest version of WRB
'''

import pandas as pd

profiles = pd.read_csv('../data/profiles.csv')
layers = pd.read_csv('../data/layers.csv')
classified_profiles = profiles[~profiles['cwrb_reference_soil_group'].isnull()]
classified_data = layers.merge(classified_profiles, how="inner", left_on=[
                               'profile_id'], right_on=['profile_id'])
classified_data.loc[classified_data['cwrb_reference_soil_group']
                    == 'Albeluvisols', 'cwrb_reference_soil_group'] = 'Retisols'

classified_data.to_csv('../data/classified_data.csv')

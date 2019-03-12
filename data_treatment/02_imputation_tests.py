import pandas as pd
from fancyimpute import KNN


data = pd.read_csv('../data/classified_data.csv',
                   dtype={"cstx_order_name": object})

data = data.drop(columns=['dataset_id', 'country_id', 'cfao_version', 'cfao_major_group',
                          'cwrb_version', 'cwrb_reference_soil_group', 'cstx_version',
                          'cstx_order_name', 'translated'])

data = data.drop(columns=list(
    data.loc[:, data.columns.str.contains('license')]))

data_filled = pd.DataFrame(data=KNN(k=3).fit_transform(
    data), columns=data.columns, index=data.index)
data_filled.to_csv('../data/imputed_full_classified_data.csv')

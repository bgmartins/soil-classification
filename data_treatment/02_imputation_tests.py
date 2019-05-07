
'''
Performs an imputation using KNN.
Can filter the data by country by using an argument :
    python3 02_imputation_tests.py MX   -> Mexico only
'''

import sys
import getopt
import pandas as pd
from fancyimpute import KNN
from sklearn.preprocessing import scale


def impute_data_with_default(df, default_columns=['latitude', 'longitude']):
    # Remove columns to use for default
    columns_to_impute = [
        x for x in df.columns if x not in set(default_columns)]

    # Remove columns that do not have null values
    columns_to_impute = [x for x in columns_to_impute if x in set(
        df.columns[df.isna().any()])]

    for i, column in enumerate(columns_to_impute):
        print('Imputing', column, i+1, 'out of', len(columns_to_impute))
        temp_df = df[([column] + default_columns)]
        data_filled = pd.DataFrame(data=KNN(k=knn).fit_transform(
            temp_df), columns=temp_df.columns, index=temp_df.index)
        df[column] = data_filled[column]

    return df


# Set variables and defaults
inputfile = '../data/classified_data.csv'
outputfile = '../data/imputed_classified_data.csv'
knn = 3
h = '02_imputation_tests.py -h <help> -i <inputfile> -o <outputfile> -k <knn neighbours> '

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:k:")
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
    elif opt in ('-k'):
        knn = int(arg)


# Read data
data = pd.read_csv(inputfile)

ids = data.profile_id

print('\n\nImputing values with knn: {}, with {} rows and {} columns.\n\n'.format(
    knn, data.shape[0], data.shape[1]))

# Scale and center data
#data = pd.DataFrame(data=scale(data), columns=data.columns, index=data.index)


'''
# Impute the missing values
data_filled = pd.DataFrame(data=KNN(k=knn).fit_transform(
    data), columns=data.columns, index=data.index)
'''

data_filled = impute_data_with_default(data)

data.profile_id = ids


# Save the data
data_filled.to_csv(outputfile, index=False)

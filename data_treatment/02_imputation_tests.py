
'''
Performs an imputation using KNN.
Can filter the data by country by using an argument :
    python3 02_imputation_tests.py MX   -> Mexico only
'''

import sys
import getopt
import pandas as pd
from fancyimpute import KNN

# Set variables and defaults
inputfile = '../data/classified_data.csv'
outputfile = '../data/imputed_full_classified_data.csv'
knn = 3
country_filter = ''
h = '02_imputation_tests.py -h <help> -i <inputfile> -o <outputfile> -k <knn neighbours> -c <filter country>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:k:c:")
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
    elif opt in ('-c'):
        country_filter = arg


# Read data
data = pd.read_csv(inputfile)

# Filter by country
if country_filter != '':
    data = data.loc[data['country_id'] == country_filter]

# Drop remaining columns that will not help on the classification
data = data.drop(columns=['country_id'])

# Remove the columns that have only missing values and therefore cannot be imputed
data = data.dropna(axis=1, how='all')

print('\n\nImputing values with knn: {}, filtering by {}, with {} rows and {} columns.\n\n'.format(
    knn, country_filter, data.shape[0], data.shape[1]))

# Impute the missing values
data_filled = pd.DataFrame(data=KNN(k=knn).fit_transform(
    data), columns=data.columns, index=data.index)

# Save the data
data_filled.to_csv(outputfile, index=False)

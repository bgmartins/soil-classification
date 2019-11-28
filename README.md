# soil-classification
Scripts associated to experiments related to soil classification


## The following software is required to run the project:
* python 3
* The following packages:
```
pip install pandas matplotlib sklearn xgboost seaborn numpy
```

## To run the project
To create the various representations of the data, first run :
```
data_treatment/create_sets.sh	
```
After that simply run the algorithm and the data representaion required:
```
python3 04_test_random_forests.py -i ../data/test/mexico_k_2_standard.csv
```

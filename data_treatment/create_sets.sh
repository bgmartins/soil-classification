#!/bin/bash  

echo "Creating classified dataset"
python3 data_treatment.py -c MX -k 1 -o ../data/mexico_k_1.csv 
python3 data_treatment.py -c MX -k 2 -o ../data/mexico_k_2.csv 
python3 data_treatment.py -c MX -k 3 -o ../data/mexico_k_3.csv 



# echo "Adding dsm data"
# python3 05_add_dsm_data.py -i ../data/mexico.csv -o ../data/mexico.csv -d ../data/rasters/test

#echo "Imputing missing values"
#python3 02_imputation_tests.py -i ../data/mexico.csv -o ../data/mexico.csv

echo "Creating the sets"
python3 data_treatment.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_layers_1.csv -m 1
python3 data_treatment.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_layers_3.csv -m 3
python3 data_treatment.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_layers_5.csv -m 5
python3 data_treatment.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_layers_7.csv -m 7

python3 data_treatment.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_depth_10.csv -d 10
python3 data_treatment.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_depth_30.csv -d 30
python3 data_treatment.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_depth_60.csv -d 60

python3 03_merge_depth.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_depth_not_weighted_10.csv -d 10
python3 03_merge_depth.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_depth_not_weighted_30.csv -d 30
python3 03_merge_depth.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_depth_not_weighted_60.csv -d 60

python3 03_merge_standard.py -i ../data/mexico_k_1.csv -o ../data/test/mexico_k_1_standard.csv

echo "Creating the sets k = 2"
python3 data_treatment.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_layers_1.csv -m 1
python3 data_treatment.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_layers_3.csv -m 3
python3 data_treatment.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_layers_5.csv -m 5
python3 data_treatment.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_layers_7.csv -m 7

python3 data_treatment.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_depth_10.csv -d 10
python3 data_treatment.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_depth_30.csv -d 30
python3 data_treatment.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_depth_60.csv -d 60

python3 03_merge_depth.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_depth_not_weighted_10.csv -d 10
python3 03_merge_depth.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_depth_not_weighted_30.csv -d 30
python3 03_merge_depth.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_depth_not_weighted_60.csv -d 60

python3 03_merge_standard.py -i ../data/mexico_k_2.csv -o ../data/test/mexico_k_2_standard.csv


echo "Creating the sets k = 3"
python3 data_treatment.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_layers_1.csv -m 1
python3 data_treatment.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_layers_3.csv -m 3
python3 data_treatment.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_layers_5.csv -m 5
python3 data_treatment.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_layers_7.csv -m 7

python3 data_treatment.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_depth_10.csv -d 10
python3 data_treatment.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_depth_30.csv -d 30
python3 data_treatment.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_depth_60.csv -d 60

python3 03_merge_depth.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_depth_not_weighted_10.csv -d 10
python3 03_merge_depth.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_depth_not_weighted_30.csv -d 30
python3 03_merge_depth.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_depth_not_weighted_60.csv -d 60

python3 03_merge_standard.py -i ../data/mexico_k_3.csv -o ../data/test/mexico_k_3_standard.csv
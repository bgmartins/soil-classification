# -*- coding: utf-8 -*-
#
# Extracts clean WRB soil groups from the original classification dataset.
#
# Author: Luís de Sousa luis (dot) de (dot) sousa (at) protonmail (dot) ch
# Date: 29-10-2018
###############################################################################

import pandas

# -- Extract Soil Groups --
wrb = pandas.read_csv("TAXNWRB_selection.csv", header=0)

wrb['WRB_GROUP'] = wrb['TAXNWRB.f'].apply(lambda x: x.split(" ")[1])

count = wrb['WRB_GROUP'].value_counts()
print(count)
print("Total classified profiles: " + str(count.sum()))
print("Total profiles in dataset: " + str(len(wrb.index)))

wrb_clean = wrb[['LOC_ID', 'WRB_GROUP']]

# -- Clean profile IDs --
profiles = pandas.read_csv("PROPS_selection.csv", header=0)

def clean_id(row):
    splt = str(row['LOC_ID']).split("_", 1)
    if (len(splt) > 1):
        return splt[1]
    else:
        return ""

profiles['CLEAN_ID'] = profiles.apply(clean_id, axis=1)
print("\nIDs in the head():")
print(profiles.head()['CLEAN_ID'])

# -- Join and save result --
classed_profs = profiles.join(wrb_clean.set_index('LOC_ID'), how='inner', on='CLEAN_ID')
print("\nNumber of rows in final DataFrame: " + str(len(classed_profs.index)))

classed_profs.to_csv("WRB_Groups.pnts.csv")
print("All done.")



WoSIS
=====

The datasets used in this study are a selection from the [Word Soil Information System](https://www.isric.org/explore/wosis) (WoSIS). The selection corresponds to all profiles in WoSIS classified either according to the [World Reference Base](http://www.fao.org/soils-portal/soil-survey/soil-classification/world-reference-base/en/) (WRB) or the [Soil Taxonomy](https://www.nrcs.usda.gov/wps/portal/nrcs/main/soils/survey/class/taxonomy/) of the United States Department of Agriculture.

Datasets
========

Profiles
--------
A profile is a column of soil that has been surveyed, usually with the collection of samples at different depths for posterior chemical analysis. In this dataset are found the following relevant fields:

- **latitude/longitude** - coordinates of the soil profile.
- **cwrb_version** - version of the WRB system used to classify the soil.
- **cwrb_reference_soil_group** - main soil class in the WRB system.
- **cstx_version** - version of the USDA soil taxonomy system used to classify the soil.		
- **cstx_order_name** - main soil class in the USDA soil taxonomy.

Layers
------
A layer is a section of a profile within which the soil properties are uniform. Each profile is composed by the a series of sequential layers. Relevant fields in this dataset:

- **profile_id** - the profile of which the layer is part.
- **upper_depth/lower_depth** - the extent of the layer in the profile.
- **..._value_average** - value of a particular variable determined according to a specific method. The first characters in this field are the key to the *Variables* dataset.

Variables
---------
Description of the variables and methods present in the *Layers* dataset. The **code**  field corresponds to the first characters in the *Layers* dataset. The *Variables* dataset is not to be used in training, it serves as a reference to the data used in this study.

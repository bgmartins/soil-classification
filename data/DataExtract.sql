SET search_path TO sg250m,public;

SELECT DISTINCT(attribute)
  FROM wosis_201901_attributes;
 
SELECT COUNT(*) 
  FROM wosis_201901_profiles
 WHERE cwrb_reference_soil_group IS NOT NULL
    OR cstx_great_group IS NOT NULL
    OR cfao_major_group IS NOt NULL;
  
-- Profiles   
DROP VIEW v_tmp;
CREATE VIEW v_tmp AS  
SELECT  profile_id, 
		dataset_id, 
		country_id, 
		latitude, 
		longitude,
       	cfao_version, 
       	cfao_major_group, 
       	cwrb_version, 
       	cwrb_reference_soil_group,
       	cstx_version,
       	cstx_order_name
  FROM  wosis_201901_profiles
 WHERE 	cwrb_reference_soil_group IS NOT NULL
    OR 	cstx_great_group IS NOT NULL
    OR 	cfao_major_group IS NOt NULL;
   
-- Layers
SELECT  profile_id,
		profile_layer_id,
		upper_depth,
		lower_depth,
		bdfi33_value_avg,
		bdfiad_value_avg,
		bdfifm_value_avg,
		bdfins_value_avg,
		bdfiod_value_avg,
		bdws33_value_avg,
		bdwsad_value_avg,
		bdwsfm_value_avg,
		bdwsns_value_avg,
		bdwsod_value_avg,
		tceq_value_avg,
		cecph7_value_avg,
		cecph8_value_avg,
		clay_value_avg,
		cfgr_value_avg,
		cfvo_value_avg,
		ecec_value_avg,
		elco1x_value_avg,
		elcons_value_avg,
		elcosp_value_avg,
		orgc_value_avg,
		phca_value_avg,
		phaq_value_avg,
		phkc_value_avg,
		phnf_value_avg,
		phpbyi_value_avg,
		phpmh3_value_avg,
		phpols_value_avg,
		phprtn_value_avg,
		phptot_value_avg,
		phpwsl_value_avg,
		sand_value_avg,
		silt_value_avg,
		totc_value_avg,
		nitkjd_value_avg,
		wg0100_value_avg,
		wg0010_value_avg,
		wg1500_value_avg,
		wg0200_value_avg,
		wg0033_value_avg,
		wg0500_value_avg,
		wg0006_value_avg,
		wv0100_value_avg,
		wv0010_value_avg,
		wv1500_value_avg,
		wv0200_value_avg,
		wv0033_value_avg,
		wv0500_value_avg,
		wv0006_value_avg
  FROM  sg250m.wosis_201901_layers
 WHERE  profile_id IN (SELECT profile_id 
                         FROM sg250m.wosis_201901_profiles); 

SELECT * FROM sg250m.v_tmp;

-- Variables		
SELECT * FROM sg250m.wosis_201901_attributes;		

-- Profiles with FAO but missing WRB
SELECT  COUNT(*)
  FROM  sg250m.v_tmp
 WHERE  cwrb_reference_soil_group IS NULL
   AND  cfao_major_group IS NOT NULL;
  
SELECT  DISTINCT(cfao_major_group)
  FROM  sg250m.v_tmp
 WHERE  cwrb_reference_soil_group IS NULL;

-- Profiles with USDA but missing WRB
SELECT  COUNT(*)
  FROM  sg250m.v_tmp
 WHERE  cwrb_reference_soil_group IS NULL
   AND  cstx_order_name IS NOT NULL;
  
SELECT  DISTINCT(cstx_order_name)
  FROM  sg250m.v_tmp
 WHERE  cwrb_reference_soil_group IS NULL;
  

-- Other queries
SELECT DISTINCT(cstx_order_name)
  FROM wosis_201901_profiles;
  
 
 
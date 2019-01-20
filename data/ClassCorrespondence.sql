SELECT DISTINCT(system_source)
  FROM wosis.class_taxa_correlation;
  
SELECT DISTINCT(system_destination)
  FROM wosis.class_taxa_correlation
 WHERE system_source LIKE 'USDA';
 
SELECT DISTINCT(system_destination)
  FROM wosis.class_taxa_correlation
 WHERE system_source LIKE 'FAO';
 
SELECT DISTINCT(taxon_source, level_source)
  FROM wosis.class_taxa_correlation
 WHERE system_source LIKE 'USDA';
 
SELECT DISTINCT(taxon_source, taxon_destination)
  FROM wosis.class_taxa_correlation
 WHERE system_source LIKE 'FAO'
   AND system_destination LIKE 'WRB'
   AND level_source LIKE '1';
   
SELECT DISTINCT(taxon_source, taxon_destination)
  FROM wosis.class_taxa_correlation
 WHERE system_source LIKE 'USDA'
   AND system_destination LIKE 'FAO'
   AND level_destination LIKE '1';
   
-- Temporary tables
CREATE TABLE sg250m.tmp_profiles AS 
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
  FROM  sg250m.wosis_201901_profiles
 WHERE 	cwrb_reference_soil_group IS NOT NULL
    OR 	cstx_great_group IS NOT NULL
    OR 	cfao_major_group IS NOT NULL;

ALTER TABLE sg250m.tmp_profiles
ADD COLUMN translated BOOLEAN;

UPDATE sg250m.tmp_profiles 
   SET translated = TRUE;

-- ################## 
-- Classes from FAO
SELECT  DISTINCT(cfao_major_group)
  FROM  sg250m.wosis_201901_profiles
 WHERE  cwrb_reference_soil_group IS NULL
    OR 	cfao_major_group IS NOT NULL; 

UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Acrisols', translated = TRUE WHERE cfao_major_group LIKE 'Acrisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Alisols', translated = TRUE WHERE cfao_major_group LIKE 'Alisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Andosols', translated = TRUE WHERE cfao_major_group LIKE 'Andosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Anthrosols', translated = TRUE WHERE cfao_major_group LIKE 'Anthrosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Arenosols', translated = TRUE WHERE cfao_major_group LIKE 'Arenosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Calcisols', translated = TRUE WHERE cfao_major_group LIKE 'Calcisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Cambisols', translated = TRUE WHERE cfao_major_group LIKE 'Cambisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Chernozems', translated = TRUE WHERE cfao_major_group LIKE 'Chernozems' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Ferralsols', translated = TRUE WHERE cfao_major_group LIKE 'Ferralsols' AND cwrb_reference_soil_group IS NULL;  
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Fluvisols', translated = TRUE WHERE cfao_major_group LIKE 'Fluvisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Gleysols', translated = TRUE WHERE cfao_major_group LIKE 'Gleysols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Kastanozems', translated = TRUE WHERE cfao_major_group LIKE 'Greyzems' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Gypsisols', translated = TRUE WHERE cfao_major_group LIKE 'Gypsisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Histosols', translated = TRUE WHERE cfao_major_group LIKE 'Histosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Kastanozems', translated = TRUE WHERE cfao_major_group LIKE 'Kastanozems' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Leptosols', translated = TRUE WHERE cfao_major_group LIKE 'Lithosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Lixisols', translated = TRUE WHERE cfao_major_group LIKE 'Lixisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Luvisols', translated = TRUE WHERE cfao_major_group LIKE 'Luvisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Nitisols', translated = TRUE WHERE cfao_major_group LIKE 'Nitosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Phaeozems', translated = TRUE WHERE cfao_major_group LIKE 'Phaeozems' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Planosols', translated = TRUE WHERE cfao_major_group LIKE 'Planosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Plinthosols', translated = TRUE WHERE cfao_major_group LIKE 'Plinthosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Podzols', translated = TRUE WHERE cfao_major_group LIKE 'Podzols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Albeluvisols', translated = TRUE WHERE cfao_major_group LIKE 'Podzoluvisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Retisols', translated = TRUE WHERE cfao_major_group LIKE 'Podzoluvisols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Regosols', translated = TRUE WHERE cfao_major_group LIKE 'Regosols' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Umbrisols', translated = TRUE WHERE cfao_major_group LIKE 'Rendzinas' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Solonchaks', translated = TRUE WHERE cfao_major_group LIKE 'Solonchaks' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Solonetz', translated = TRUE WHERE cfao_major_group LIKE 'Solonetz' AND cwrb_reference_soil_group IS NULL;
UPDATE sg250m.tmp_profiles SET cwrb_reference_soil_group = 'Vertisols', translated = TRUE WHERE cfao_major_group LIKE 'Vertisols' AND cwrb_reference_soil_group IS NULL;

SELECT  COUNT(*)
  FROM  sg250m.tmp_profiles
 WHERE  cwrb_reference_soil_group IS NULL
   AND  cfao_major_group IS NOT NULL;
  
-- ################## 
-- Classes from USDA
SELECT DISTINCT(taxon_source, taxon_destination)
  FROM wosis.class_taxa_correlation
 WHERE system_source LIKE 'USDA'
   AND system_destination LIKE 'FAO'
   AND level_destination LIKE '1';
  
UPDATE  sg250m.tmp_profiles 
   SET  cwrb_reference_soil_group = 'Histosols', 
        translated = TRUE 
 WHERE  cwrb_reference_soil_group IS NULL
   AND  profile_id IN(
 		SELECT  profile_id
          FROM  sg250m.wosis_201901_profiles
         WHERE  cstx_order_name LIKE '%Borosaprist%'
            OR  cstx_suborder LIKE '%Borosaprist%'
            OR  cstx_great_group LIKE '%Borosaprist%'
            OR  cstx_subgroup LIKE '%Borosaprist%'
        );

UPDATE  sg250m.tmp_profiles 
   SET  cwrb_reference_soil_group = 'Vertisols', 
        translated = TRUE 
 WHERE  cwrb_reference_soil_group IS NULL
   AND  profile_id IN(
 		SELECT  profile_id
          FROM  sg250m.wosis_201901_profiles
         WHERE  cstx_order_name LIKE '%Chromoxerert%'
            OR  cstx_suborder LIKE '%Chromoxerert%'
            OR  cstx_great_group LIKE '%Chromoxerert%'
            OR  cstx_subgroup LIKE '%Chromoxerert%'
        );

UPDATE  sg250m.tmp_profiles 
   SET  cwrb_reference_soil_group = 'Histosols', 
        translated = TRUE 
 WHERE  cwrb_reference_soil_group IS NULL
   AND  profile_id IN(
 		SELECT  profile_id
          FROM  sg250m.wosis_201901_profiles
         WHERE  cstx_order_name LIKE '%Fibrist%'
            OR  cstx_suborder LIKE '%Fibrist%'
            OR  cstx_great_group LIKE '%Fibrist%'
            OR  cstx_subgroup LIKE '%Fibrist%'
        );

UPDATE  sg250m.tmp_profiles 
   SET  cwrb_reference_soil_group = 'Gypsisols', 
        translated = TRUE 
 WHERE  cwrb_reference_soil_group IS NULL
   AND  profile_id IN(
 		SELECT  profile_id
          FROM  sg250m.wosis_201901_profiles
         WHERE  cstx_order_name LIKE '%Gypsiorthid%'
            OR  cstx_suborder LIKE '%Gypsiorthid%'
            OR  cstx_great_group LIKE '%Gypsiorthid%'
            OR  cstx_subgroup LIKE '%Gypsiorthid%'
        );
       
UPDATE  sg250m.tmp_profiles 
   SET  cwrb_reference_soil_group = 'Plinthosols', 
        translated = TRUE 
 WHERE  cwrb_reference_soil_group IS NULL
   AND  profile_id IN(
 		SELECT  profile_id
          FROM  sg250m.wosis_201901_profiles
         WHERE  cstx_order_name LIKE '%Haplorthox%'
            OR  cstx_suborder LIKE '%Haplorthox%'
            OR  cstx_great_group LIKE '%Haplorthox%'
            OR  cstx_subgroup LIKE '%Haplorthox%'
        );
      
UPDATE  sg250m.tmp_profiles 
   SET  cwrb_reference_soil_group = 'Vertisols', 
        translated = TRUE 
 WHERE  cwrb_reference_soil_group IS NULL
   AND  profile_id IN(
 		SELECT  profile_id
          FROM  sg250m.wosis_201901_profiles
         WHERE  cstx_order_name LIKE '%Pelloxerert%'
            OR  cstx_suborder LIKE '%Pelloxerert%'
            OR  cstx_great_group LIKE '%Pelloxerert%'
            OR  cstx_subgroup LIKE '%Pelloxerert%'
        );
      
UPDATE  sg250m.tmp_profiles 
   SET  cwrb_reference_soil_group = 'Plinthosols', 
        translated = TRUE 
 WHERE  cwrb_reference_soil_group IS NULL
   AND  profile_id IN(
 		SELECT  profile_id
          FROM  sg250m.wosis_201901_profiles
         WHERE  cstx_order_name LIKE '%Plinthaquept%'
            OR  cstx_suborder LIKE '%Plinthaquept%'
            OR  cstx_great_group LIKE '%Plinthaquept%'
            OR  cstx_subgroup LIKE '%Plinthaquept%'
        );

-- ################  
-- Save output and clean up
SELECT * FROM sg250m.tmp_profiles;
DROP TABLE sg250m.tmp_profiles;




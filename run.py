from masking_cloud_shadows import save_mask
from inundated_areas import inundated_area_calculation, croping_display


outer_path = 'S2A_MSIL1C_20220102T001111_N0301_R073_T55JGG_20220102T013405.SAFE'
shapefile_path = "../namoi_dams2.shp"
dam = 's242-1'


save_mask(outer_path)
dam, date, inundated_area, unmasked_ratio = inundated_area_calculation(outer_path, shapefile_path, dam)

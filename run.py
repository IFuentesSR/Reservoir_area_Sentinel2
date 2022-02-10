import sys

from masking_cloud_shadows import save_mask
from inundated_areas import inundated_area_calculation, croping_display, raster_intersects
# print(sys.version)
# !pip install pyshp

if len(sys.argv) != 4:
    print('run.py requires three arguments:\n\
           1 : the path to Sentinel 2 folder;\n\
           2 : the path to dams shapefile;\n\
           3 : the id/name of the dam')
else:
    outer_path = sys.argv[1]
    shapefile_path = sys.argv[2]
    dam = sys.argv[3]

    # outer_path = 'S2A_MSIL1C_20220102T001111_N0301_R073_T55JGG_20220102T013405.SAFE'
    # shapefile_path = "../namoi_dams2.shp"
    # dam = 's242-1'


    save_mask(outer_path)
    if raster_intersects(outer_path, shapefile_path, dam):
        dam, date, inundated_area, unmasked_ratio = inundated_area_calculation(outer_path, shapefile_path, dam)
        print('reservoir {} has {} m² inundated on date {}\
        with an unmasked ratio of {}'.format(dam, inundated_area, date, unmasked_ratio))
        croping_display(outer_path, shapefile_path, dam)
    else:
        print("polygon doesn't intercept rater")


"""ToDo import sys and pass arguments from terminal to run.py, maybe create
class for inundated_areas"""

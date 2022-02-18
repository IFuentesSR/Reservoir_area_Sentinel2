# Reservoir_area_Sentinel2
Cloud and cloud shadow masking of Sentinel 2 Images, and inundated area estiimations for reservoirs.

The code uses the s2cloudless algorithm to mask clouds, and the solar azimuthal angle to project shadows.
The run.py file can be run from the teerminal passing three arguments, the path to Sentinel 2 level 1C folders, the path to reservoir shapefile and the name of the dam.
For instance, it can be executed running:

```
python run.py "path_to_S2_folder" "path_to_shapefile" "dam_name"
```

The code creates a *tmp* folder where the cloud mask and resampled bands are stored.
It also stores a raster window surrounding the dam containing the water index calculated for specific Sentinel 2 images. These are stored with the name of the dam followed by the date of the Sentinel 2 images.

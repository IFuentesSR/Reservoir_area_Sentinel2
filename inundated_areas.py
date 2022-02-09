import os
import rasterio
import rasterio.mask
from rasterio.warp import reproject, Resampling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import shapefile
from shapely.geometry import Polygon, MultiPolygon, box


def get_feature(shape_path, dam):
    """Retrieves feature from shapefile given the name of dam.

    Parameters
    ----------
    shape_path : str
        path to shapefile.
    dam : str
        name of dam feature.

    Returns
    -------
    feature : shapefile.ShapeRecord
        feature associated with dam.

    """
    shape = shapefile.Reader(shape_path)
    feature = [n for n in shape.shapeRecords() if n.record.DAM_LOTPLA == dam][0]
    return feature


def single_band(path):
    """Retrieves the path to high resolution B02 S2 band.

    Parameters
    ----------
    path : str
        path of Sentinel 2 image folder.

    Returns
    -------
    single_band_path : str
        path to 10 m resolution B02 Sentinel 2 band.

    """
    inner_path = os.path.join(path,
                              'GRANULE',
                              os.listdir(os.path.join(path, 'GRANULE'))[0])
    band_paths = [os.path.join(os.path.join(inner_path, 'IMG_DATA'), n)
                  for n in os.listdir(os.path.join(inner_path, 'IMG_DATA'))]
    bands = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    resampled_bands = [1, 2, 3, 4, 5, 8, 9]
    sorted_bands = [[m for m in band_paths if n in m] for n in bands]
    sorted_bands = [n for l in sorted_bands for n in l]
    single_band_path = sorted_bands[1]
    return single_band_path


def raster_intersects(raster_path, shape_path, dam):
    """Tests if dam intersects the raster image.

    Parameters
    ----------
    raster_path : str
        path of Sentinel 2 raster folder.
    shape_path : str
        path to shapefile.
    dam : str
        name of dam.

    Returns
    -------
    Boolean
        True if dam feature intersects raster image.

    """
    single_band_path = single_band(raster_path)
    fil = rasterio.open(single_band_path)
    box_raster = box(*fil.bounds)
    fea = get_feature(shape_path, dam)
    poly = Polygon(fea.shape.__geo_interface__['coordinates'][0])
    return box_raster.intersects(poly)


def fea_area(feature):
    """Creates polygon from feature and calculates its area.

    Parameters
    ----------
    feature : shapefile.ShapeRecord
        feature associated with dam.

    Returns
    -------
    poly : shapely polygon
        polygon geometry of feature
    area_poly : float
        area of polygon.

    """
    geo_poly = feature.shape.__geo_interface__
    if geo_poly['type'] == 'Polygon':
        poly = Polygon(geo_poly['coordinates'][0])
    area_poly = poly.area
    return poly, area_poly


def buffered_area(feature):
    """Creates a buffered polygon and calculates its area.

    Parameters
    ----------
    feature : shapefile.ShapeRecord
        feature associated with dam.

    Returns
    -------
    buffered : shapely Polygon
        polygon geometry buffered by 10 m.
    buffered.area : float
        area of buffered polygon.

    """
    geo_poly = feature.shape.__geo_interface__
    if geo_poly['type'] == 'Polygon':
        poly = Polygon(geo_poly['coordinates'][0])
    buffered = poly.buffer(10)
    return buffered, buffered.area


def croping(img_band, feature):
    """Crops raster based on polygon.

    Parameters
    ----------
    img_band : str
        path to img_band.
    feature : shapely Polygon
        Polygon geometry from feature.

    Returns
    -------
    out_img : np.array
        array of the image for the size of the polygon used
    out_meta : dictionary
        metadata for the cropped image

    """
    with rasterio.open(img_band) as src:
        out_image, out_transform = rasterio.mask.mask(src,
                                                      [feature],
                                                      crop=True,
                                                      nodata=0)
        out_meta = src.meta
        out_meta.update({'transform':out_transform,
                         'width':out_image.shape[2],
                         'height':out_image.shape[1]})
        src.close()
    return out_image, out_meta


def get_windows(shape, reservoir, img):
    """Retrieves small window from raster surrounding dam.

    Parameters
    ----------
    shape : str
        path to shapefile.
    reservoir : str
        Name of dam.
    img : str
        path to img.

    Returns
    -------
    out_img : np.array
        array of the image for the size of the window used
    out_meta : dictionary
        metadata for the cropped image

    """
    fea = get_feature(shape, reservoir)
    buffer_poly, buffer_area = buffered_area(fea)
    out_img, out_meta = croping(img, buffer_poly)
    return out_img, out_meta


def read_band(img_band):
    """Reads img_band.

    Parameters
    ----------
    img_band : str
        path to S2 band.

    Returns
    -------
    band : np.array
        array of values for img_band.
    meta : dictionary
        metadata of img_band raster.
    aff : dictionary
        affine of img_band.

    """
    with rasterio.open(img_band) as scl:
        band = scl.read()
        meta = scl.meta
        aff = scl.transform
        scl.close()
    return band, meta, aff


def resample_bands(img_band, sub_band):
    """resamples sub_band to target resolution of img_band.

    Parameters
    ----------
    img_band : str
        path to Sentinel 2 band (target scale).
    sub_band : str
        path to Sentinel 2 band (old_scale).

    Returns
    -------
    sb_band : np.array
        resampled array for sub_band
    lead_meta : dictionary
        metadata of target scale image (img_band).

    """
    lead_band, lead_meta, aff = read_band(img_band)
    tmparr = np.empty_like(lead_band)
    sb_band, sb_meta, sb_aff = read_band(sub_band)
    reproject(sb_band, tmparr,
              src_transform = sb_aff,
              dst_transform = aff,
              src_crs = sb_meta['crs'],
              dst_crs = sb_meta['crs'],
              resampling = Resampling.nearest)
    sb_band = tmparr.copy()
    return sb_band, lead_meta


def save_resampled(img_band, sub_band, name):
    """Saves resampled band as tif file in tmp folder.

    Parameters
    ----------
    img_band : str
        path to Sentinel 2 band (target scale).
    sub_band : str
        path to Sentinel 2 band (old_scale).
    name : str
        name for resampled image.

    Returns
    -------
    None
        Saves resampled bands in tmp folder.

    """
    if 'tmp' not in os.listdir():
        os.mkdir('tmp')
    sb_band, meta = resample_bands(img_band, sub_band)
    with rasterio.open(os.path.join('tmp', name), "w", **meta) as dest:
        dest.write(sb_band)


def band_lists(outer_path):
    """Retrieves the list of paths for S2 bands.

    Parameters
    ----------
    outer_path : str
        path of Sentinel 2 raster folder.

    Returns
    -------
    bands10 : list
        list with 10 m resolution bands.
    bands20 : list
        list with 20 m resolution bands.

    """
    inner_path = os.path.join(outer_path,
                              'GRANULE',
                              os.listdir(os.path.join(outer_path, 'GRANULE'))[0])
    band_paths = [os.path.join(os.path.join(inner_path, 'IMG_DATA'), n)
                  for n in os.listdir(os.path.join(inner_path, 'IMG_DATA'))]
    band_sufix10 = ['B02', 'B03', 'B04', 'B08']
    bands10 = [[m for m in band_paths if n in m] for n in band_sufix10]
    bands10 = [n for s in bands10 for n in s]
    band_sufix20 = ['B11', 'B12']
    bands20 = [[m for m in band_paths if n in m] for n in band_sufix20]
    bands20 = [n for s in bands20 for n in s]
    return bands10, bands20


def resample_bands20(bands10, bands20):
    """Saves resampled bands in tmp folder.

    Parameters
    ----------
    bands10 : list
        list with 10 m resolution bands.
    bands20 : list
        list with 20 m resolution bands.

    Returns
    -------
    None
        Saves resampled 20 m bands to 10 m bands in tmp file.

    """
    [save_resampled(bands10[0],  n, 'resampled{}.tif'.format(n[-30:-4]))
           for n in bands20 if 'resampled{}.tif'.format(n[-30:-4]) not in os.listdir('tmp')]


def get_band_paths(outer_path):
    """creates list containing band paths required.

    Parameters
    ----------
    outer_path : str
        path of Sentinel 2 raster folder.

    Returns
    -------
    path_bands2 : list
        list containing resampled bands and mask for further processing.

    """
    bands10, bands20 = band_lists(outer_path)
    resample_bands20(bands10, bands20)
    band_sufix20 = ['resampled{}'.format(n[-30:-4]) for n in bands20]
    bands20 = [['tmp/{}'.format(m) for m in os.listdir('tmp') if (n in m) & (m.endswith('tif'))]
               for n in band_sufix20]
    bands20 = [n for s in bands20 for n in s]
    path_bands2 = bands10 + bands20
    path_bands2.append("tmp/mask{}.tif".format(bands10[0][-30:-8]))
    return path_bands2


def water_index(array):
    """Creates Fisher water index.

    Parameters
    ----------
    array : np.array
        array with stacked bands required for water index.

    Returns
    -------
    water_ix : np.array
        Fisher's water index for S2 image.

    """
    agg_array = array / 10000
    blue = agg_array[0,:,:,0]
    green = agg_array[0,:,:,1]
    red = agg_array[0,:,:,2]
    nir = agg_array[0,:,:,3]
    swir1 = agg_array[0,:,:,4]
    swir2 = agg_array[0,:,:,5]
    water_ix = 1.7204 + 171 * green + 3 * red - 70 * nir - 45 * swir1 - 71 * swir2
    return water_ix


def inundated_area_calculation(outer_path, shapefile, dam):
    """Allows the calculation of inundated areas using features and rasters.

    Parameters
    ----------
    outer_path : str
        path of Sentinel 2 raster folder.
    shapefile : str
        path to shapefile.
    dam : str
        name of dam.

    Returns
    -------
    dam : str
        name of dam.
    date : datetime
        monitoring date.
    inundated_area : float
        calculation of inundated area within dam.
    unmasked_ratio : float
        ratio of non-masked pixels.

    """
    paths = get_band_paths(outer_path)
    cloud_mask = get_windows(shapefile, dam, paths[-1])
    band_list = [get_windows(shapefile, dam, n) for n in paths[:-1]]
    agg_array = np.stack([n[0] for n in band_list]).transpose(1, 2, 3, 0)
    mask = np.where(agg_array[0,:,:,0] == 0, 0, 1)
    water_ix = water_index(agg_array)
    water_thresh = 0
    masked_water = np.where(mask == 0, np.nan, water_ix)
    masked_water = np.where(cloud_mask[0][0, :, :] > 0, np.nan, masked_water)
    pixel_area = band_list[0][1]['transform'][0]**2
    unmasked_area = np.sum(np.where(masked_water > -999, 1, 0)) * pixel_area
    buff_fea_area = buffered_area(get_feature(shapefile, dam))[1]
    unmasked_ratio = unmasked_area / buff_fea_area
    water_arr = np.where(masked_water > water_thresh, 1, 0)
    inundated_area = np.sum(water_arr) * pixel_area
    return dam, pd.to_datetime(outer_path[11:19], format='%Y%m%d'), inundated_area, unmasked_ratio


def croping_display(outer_path, shape, dam):
    """Stores water index small images based on dam and raster images.

    Parameters
    ----------
    outer_path : str
        path of Sentinel 2 raster folder.
    shapefile : str
        path to shapefile.
    dam : str
        name of dam.

    Returns
    -------
    None
        Saves small windows with the water index calculated for S2 scenes in tmp.

    """
    fea = get_feature(shape, dam)
    poly = Polygon(fea.shape.__geo_interface__['coordinates'][0])
    buffer = poly.buffer(300)
    coords = buffer.bounds
    coords = [(coords[0], coords[1]), (coords[2], coords[1]),
              (coords[2], coords[3]), (coords[0], coords[3])]
    outer_poly = Polygon(coords)
    paths = get_band_paths(outer_path)
    date = outer_path[11:19]
    cloud_mask, mask_meta = croping(paths[-1], outer_poly)
    band_list = [croping(n, outer_poly) for n in paths[:-1]]
    agg_array = np.stack([n[0] for n in band_list]).transpose(1, 2, 3, 0)
    mask = np.where(agg_array[0,:,:,0] == 0, 0, 1)
    water_ix = water_index(agg_array)
    masked_water = np.where(mask == 0, np.nan, water_ix)
    masked_water = np.where(cloud_mask[0, :, :] > 0, np.nan, masked_water)
    mask_meta.update({'dtype':'float32', 'driver':'GTiff', 'count':1})
    with rasterio.open(os.path.join('tmp', '{}_{}.tif'.format(dam, date)), "w", **mask_meta) as dest:
        dest.write(masked_water, indexes=1)

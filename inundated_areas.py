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
    shape = shapefile.Reader(shape_path)
    feature = [n for n in shape.shapeRecords() if n.record.DAM_LOTPLA == dam][0]
    return feature


def single_band(path):
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
    single_band_path = single_band(raster_path)
    fil = rasterio.open(single_band_path)
    box_raster = box(*fil.bounds)
    fea = get_feature(shape_path, dam)
    poly = Polygon(fea.shape.__geo_interface__['coordinates'][0])
    return box_raster.intersects(poly)


def fea_area(feature):
    geo_poly = feature.shape.__geo_interface__
    if geo_poly['type'] == 'Polygon':
        poly = Polygon(geo_poly['coordinates'][0])
    area_poly = poly.area
    return poly, area


def buffered_area(feature):
    geo_poly = feature.shape.__geo_interface__
    if geo_poly['type'] == 'Polygon':
        poly = Polygon(geo_poly['coordinates'][0])
    buffered = poly.buffer(10)
    return buffered, buffered.area


def croping(img_band, feature):
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
    fea = get_feature(shape, reservoir)
    buffer_poly, buffer_area = buffered_area(fea)
    out_img, out_meta = croping(img, buffer_poly)
    return out_img, out_meta


def read_band(img_band):
    with rasterio.open(img_band) as scl:
        band = scl.read()
        meta = scl.meta
        aff = scl.transform
        scl.close()
    return band, meta, aff


def resample_bands(img_band, sub_band):
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
    if 'tmp' not in os.listdir():
        os.mkdir('tmp')
    sb_band, meta = resample_bands(img_band, sub_band)
    with rasterio.open(os.path.join('tmp', name), "w", **meta) as dest:
        dest.write(sb_band)


def band_lists(outer_path):
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
    [save_resampled(bands10[0],  n, 'resampled{}.tif'.format(n[-30:-4]))
           for n in bands20 if 'resampled{}.tif'.format(n[-30:-4]) not in os.listdir('tmp')]


def get_band_paths(outer_path):
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

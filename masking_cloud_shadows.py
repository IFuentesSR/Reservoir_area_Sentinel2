import os
from bs4 import BeautifulSoup
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest, get_s2_evalscript
from scipy.ndimage.interpolation import shift


def unproj_array(img_band):
    """Reads image as array.

    Parameters
    ----------
    img_band : str
        path to Sentinel 2 band.

    Returns
    -------
    band : np.array
        array with pixel values of image.

    """
    with rasterio.open(img_band) as scl:
        band = scl.read()
        scl.close()
    return band


def read_band(img_band):
    """Reads Sentinel 2 image.

    Parameters
    ----------
    img_band : str
        path to Sentinel 2 band.

    Returns
    -------
    band : np.array
        array for img_band.
    meta: dictionary
        metadata of img_band.
    aff : dictionary
        affine for img_band.

    """
    with rasterio.open(img_band) as scl:
        band = scl.read()
        meta = scl.meta
        meta.update({'dtype':'uint8'})
        aff = scl.transform
        scl.close()
    return band, meta, aff


def get_meta(img_band):
    """Retrieves metadata of image.

    Parameters
    ----------
    img_band : str
        path to Sentinel 2 band.

    Returns
    -------
    meta : dictionary
        metadata of img_band updated to uint8 dtype.

    """
    with rasterio.open(img_band) as scl:
        meta = scl.meta
        meta.update({'dtype':'uint8'})
        scl.close()
    return meta


def resample_bands(img_band, sub_band):
    """Resamples sub_band to img_band.

    Parameters
    ----------
    img_band : str
        path to Sentinel 2 band (target scale).
    sub_band : str
        path to Sentinel 2 band (old_scale).

    Returns
    -------
    sb_band : np.array
        resampled array for sub_band.

    """
    lead_band, lead_meta, aff = read_band(img_band)
    tmparr = np.empty_like(lead_band)
    sb_band, sb_meta, sb_aff = read_band(sub_band)
    reproject(sb_band, tmparr,
              src_transform = sb_aff,
              dst_transform = aff,
              src_crs = sb_meta['crs'],
              dst_crs = sb_meta['crs'],
              resampling = Resampling.cubic)
    sb_band = tmparr.copy()
    return sb_band


def resample_img(inner_path):
    """resample high resolution to lower resolution (60 m of B01).

    Parameters
    ----------
    inner_path : str
        path to granule subfolder in Sentinel 2.

    Returns
    -------
    arrays_cat : np.array
        array of resampled bands in image.
    meta : dictionary
        metadata of resampled bands.
    meta_higher : dictionary
        metadata for higher resolution bands.
    sorted_bands : list
        paths of sorted bands.

    """
    band_paths = [os.path.join(os.path.join(inner_path, 'IMG_DATA'), n)
                  for n in os.listdir(os.path.join(inner_path, 'IMG_DATA'))]
    bands = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    resampled_bands = [1, 2, 3, 4, 5, 8, 9]
    sorted_bands = [[m for m in band_paths if n in m] for n in bands]
    sorted_bands = [n for l in sorted_bands for n in l]
    arrays = [unproj_array(n) if i not in resampled_bands
              else resample_bands(sorted_bands[0], n)
              for i,n in enumerate(sorted_bands)]
    arrays = [n/10000 for n in arrays]
    arrays_cat = np.stack(arrays)
    arrays_cat = arrays_cat.transpose(1, 2, 3, 0)
    meta =  get_meta(sorted_bands[0])
    meta_higher = get_meta(sorted_bands[1])
    return arrays_cat, meta, meta_higher, sorted_bands


def get_azizen(path):
    """Retrieves azimuth and zenith angles from metadata.

    Parameters
    ----------
    path : str
        path to metadata file.

    Returns
    -------
    mean_azimuth : float
        mean solar azimuth angle.
    zenith_angle : float
        mean zenith viewing angle.
    """
    with open(path, 'r') as f:
        data = f.read()
        f.close()
    metadata = BeautifulSoup(data, 'xml')
    mean_angles = metadata.find_all('Mean_Sun_Angle')
    az_str = mean_angles[0].find('AZIMUTH_ANGLE').decode()
    mean_azimuth = float(az_str[az_str.find('>')+1: az_str.find('</')])
    zen_incidence = [n for n in metadata.find_all('Mean_Viewing_Incidence_Angle')
                     if n.get('bandId') == '10']
    zen_str = zen_incidence[0].decode()
    ix_beg = zen_str.find('ZENITH_ANGLE unit="deg">')+24
    ix_end = zen_str.find('</ZENITH_ANGLE')
    zenith_angle = float(zen_str[ix_beg:ix_end])
    return mean_azimuth, zenith_angle


def cloud_shadow_mask(path, threshold=0.4, average_over=4, dilation_size=2):
    """Creates mask array, metadata, metadata for higher resolution bands, and
    list of sorted band paths.

    Parameters
    ----------
    path : str
        path to Sentinel 2 image folder.
    threshold : float
        cloud probability threshold for mask.
    average_over : integer
        Size of the disk in pixels for performing convolution
        (averaging probability over pixels).
    dilation_size : integer
        Size of the disk in pixels for performing dilation.

    Returns
    -------
    masked : np.array
        cloud-shadows mask array.
    meta : dictionary
        metadata of cloud band (60 m resolution).
    meta_higher : dictionary
        metadata of higher resolution bands (10 m resolution).
    sorted : list
        list of sorted band paths.

    """
    inner_path = os.path.join(path,
                              'GRANULE',
                              os.listdir(os.path.join(path, 'GRANULE'))[0])
    resampled, meta, meta_higher, sorted = resample_img(inner_path)
    cloud_detector = S2PixelCloudDetector(threshold=threshold,
                                          average_over=average_over,
                                          dilation_size=dilation_size)
    cloud = cloud_detector.get_cloud_masks(resampled).astype(rasterio.uint8)
    metadata_path = os.path.join(inner_path,
                                 [n for n in os.listdir(inner_path)
                                  if n.endswith('.xml')][0])
    # projecting clouds for shadow masking
    azi, zen = get_azizen(metadata_path)
    azimuth = (azi * np.pi /180) + (0.5 * np.pi)
    x = np.round(np.cos(azimuth) * 15)
    y = np.round(np.sin(azimuth) * 15)
    shadows = shift(cloud[0, :, :], shift=[y,x], cval=np.NaN)
    shadows = shadows*2
    masked = np.where(cloud != 1, shadows, cloud)
    return masked, meta, meta_higher, sorted


def save_mask(path, threshold=0.4, average_over=4, dilation_size=2):
    """Creates tmp folder and saves cloud-shadows mask for S2 image in path.

    Parameters
    ----------
    path : str
        path to Sentinel 2 image folder.
    threshold : float
        cloud probability threshold for mask.
    average_over : integer
        Size of the disk in pixels for performing convolution
        (averaging probability over pixels).
    dilation_size : integer
        Size of the disk in pixels for performing dilation.

    Returns
    -------
    None
        Saves mask as a tif file in tmp folder.

    """
    if 'tmp' not in os.listdir():
        os.mkdir('tmp')
    array, meta, meta_higher, sorted_bands = cloud_shadow_mask(path,
                                                               threshold,
                                                               average_over,
                                                               dilation_size)
    lead_meta, aff = meta_higher, meta_higher['transform']
    tmparr = np.empty_like(rasterio.open(sorted_bands[1]).read())
    sb_meta, sb_aff = meta, meta['transform']
    reproject(array, tmparr,
              src_transform = sb_aff,
              dst_transform = aff,
              src_crs = sb_meta['crs'],
              dst_crs = sb_meta['crs'],
              resampling = Resampling.cubic)
    sb_band = tmparr.copy()
    name = sorted_bands[0][-30:-8]
    if 'mask{}.tif'.format(name) not in os.listdir('tmp'):
        with rasterio.open('tmp/mask{}.tif'.format(name), "w", **meta_higher) as dest:
            dest.write(sb_band)

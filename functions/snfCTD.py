'''
    Set of functions to process RBR CTD data
    Created by Eric Oliver 
'''

import numpy as np
from scipy import interpolate as interp
#from scipy import ndimage
#from datetime import datetime
from matplotlib import pyplot as plt
import gsw

def trimProfiles(ctd, zTrimTop, zTrimBot):
    '''
    Trim profiles by removing all data for depths shallower
    than 'zTrimTop' and all data for depths deeper than
    'zTrimBot' off the bottom (defined by deepest value in profile)
    '''
    for site in ctd.keys():
        kk = (ctd[site]['Depth'] > zTrimTop) * (ctd[site]['Depth'] < ctd[site]['Depth'].max()-zTrimBot)
        for key in ctd[site].keys():
            ctd[site][key] = ctd[site][key][kk]
    #
    return ctd

def binAverage(ctd, z, dz):
    '''
    Bin-average CTD data, using bin centres 'z' and bin width 'dz'

    Takes ctd variable in the format ctd[site]
    '''
    ctdBinned = {}
    for site in ctd.keys():
        # Bin-average raw CTD data onto specified vertical grid
        ctdBinned[site] = {}
        for key in ctd[site].keys():
            if key == 'Depth':
                ctdBinned[site]['Depth'] = z
            else:
                ctdBinned[site][key] = np.nan*np.ones(z.shape)
                for kk in range(len(z)):
                    kkk = (ctd[site]['Depth'] > z[kk]-0.5*dz) * (ctd[site]['Depth'] < z[kk]+0.5*dz)
                    if kkk.sum() >= 1:
                        ctdBinned[site][key][kk] = np.nanmean(ctd[site][key][kkk])
    #
    return ctdBinned

def smoothProfiles(ctd, L, fields):
    '''
    Smooth profiles using a flat window of length L.
    Smooth only the fields indicated.
    '''
    ctd_sm = {}
    for site in ctd.keys():
        ctd_sm[site] = {}
        for field in ctd[site].keys():
            if np.in1d(field, fields):
                ctd_sm[site][field] = runavg(ctd[site][field], L, mode='valid')
            else:
                ctd_sm[site][field] = ctd[site][field]
    #
    return ctd_sm

def interpSection(x_sec, depth_sec, sites, ctd, coord, field):
    '''
    Interpolate the ctd data at locations indicated by sites
    over the 2D grid given by x_sec, depth_sec. The x coordinate
    can either be latitude or longitude, given by coord ('lon'/'lat')
    '''
    # Interpolate over a grid
    xx_sec, ddepth_sec = np.meshgrid(x_sec, depth_sec)
    # Data points
    x_sparse = np.array([])
    depth_sparse = np.array([])
    field_sparse = np.array([])
    for site in sites:
        valid = ~np.isnan(ctd[site][field])
        k1 = np.where(valid)[0][0] # Only pass the main valid chunk of data
        k2 = np.where(valid)[0][-1]
        x_sparse = np.append(x_sparse, coord[site]*(ctd[site]['Depth'][k1:k2+1]*0+1))
        depth_sparse = np.append(depth_sparse, ctd[site]['Depth'][k1:k2+1])
        field_sparse = np.append(field_sparse, ctd[site][field][k1:k2+1])
    #
    field_sec = interp.griddata(np.array([x_sparse, depth_sparse]).T, field_sparse, (xx_sec, ddepth_sec), method='linear')
    return field_sec, xx_sec, depth_sec

def interpSectionStitch(sites, ctd, coord, field, dx):
    '''
    Stitch together a series of station-to-station interpolations
    to build up an interpolated section
     dx     - step in section axis
     sites   - which locations/sites to use for section
     ctd    - dictionary of ctd sites
     coord  - coordinates of ctd sites
     field  - name of oceanographic field to interpolate
    '''
    # Initialize some variables
    Nsites = len(sites)
    x1 = 999
    x2 = -999
    d2 = -9999
    for site in sites:
        x1 = np.min([coord[site], x1])
        x2 = np.max([coord[site], x2])
        d2 = np.max([np.nanmax(ctd[site]['Depth']), d2])
    # Interpolation coordinates
    dd = 0.05
    depth_sec = np.arange(0, d2+dd, dd)
    #depth_sec = ctd[site]['Depth'] #np.arange(d1, d2+dd, dd)
    x_sec = np.zeros((0,))
    field_sec = np.zeros((len(depth_sec), 0))
    # Build up interpolated section
    for i in range(Nsites-1):
        site1 = sites[i]
        site2 = sites[i+1]
        # Interpolate over a grid
        x_sec0 = np.arange(coord[site1], coord[site2], dx)
        xx_sec0, ddepth_sec0 = np.meshgrid(x_sec0, depth_sec)
        field_sec0, tmp, tmp = interpSection(x_sec0, depth_sec, sites[i:i+1+1], ctd, coord, field)
        # Add to full section arrays
        x_sec = np.append(x_sec, x_sec0, axis=0)
        field_sec = np.append(field_sec, field_sec0, axis=1)
    #
    xx_sec, ddepth_sec = np.meshgrid(x_sec, depth_sec)
    #
    return field_sec, xx_sec, ddepth_sec

def runavg(ts, w, mode='same'):
    '''
    Perform running average of ts (1D numpy array) using uniform window
    of width w (w must be odd). Option 'mode' specifies if output should
    be defined over only the 'valid' range of of the data/window convolution
    (in which case the output is padded with NaNs outside the valid range)
    or over the 'same' range as the original data.
    '''
    if mode == 'same':
        ts_smooth = np.convolve(ts, np.ones(w)/w, mode=mode)
    elif mode == 'valid':
        ts_smooth = np.append(np.append(np.nan*np.ones(int((w-1)/2)), np.convolve(ts, np.ones(w)/w, mode=mode)), np.nan*np.ones(int((w-1)/2)))
    return ts_smooth


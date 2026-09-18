'''
Custom plotting functions
'''
import numpy as np
import colormaps as cmaps
from matplotlib.colors import ListedColormap

def custom_cmap_trimmed(orig_cmap, skew, vmin, vmax):
	# Parameters
	# vmin, vmax - cut off the first vmin–vmax% of the colormap
	# Increase skew to push more of the colormap toward the beginning of the colorbar
	# Skewed linspace in the trimmed domain
	n_colors = 256
	skewed_positions = vmin + (vmax - vmin) * (np.linspace(0, 1, n_colors) ** skew)
	trimmed_skewed_colors = orig_cmap(skewed_positions)
	custom_cmap = ListedColormap(trimmed_skewed_colors)
	return custom_cmap

def shifted_cmap(orig_cmap, vmin, vcenter, vmax, ncolors=256):

    # Where should the center fall in data space?
    center_frac = (vcenter - vmin) / (vmax - vmin)

    # First half of colormap
    lower = orig_cmap(np.linspace(0, 0.5, int(ncolors*center_frac)))

    # Second half of colormap
    upper = orig_cmap(np.linspace(0.5, 1, ncolors-int(ncolors*center_frac)))

    colors = np.vstack([lower, upper])

    return ListedColormap(colors)

# Nonlinear transform to bias toward lower values
def skewed_linspace(n=256, skew=3):
    """Generate skewed linspace (more samples near 0)"""
    return np.linspace(0, 1, n)**skew

def custom_cmap_skew(orig_cmap,skew):
	# Increase skew to push more of the colormap toward the beginning of the colorbar
	skewed_colors = orig(skewed_linspace(n=256, skew=skew))
	custom_cmap = ListedColormap(skewed_colors)
	return custom_cmap

def plot_cartopy(ax):
    '''
    Set up the cartopy map with projection rotated so that Labrador coast is vertical. Input is the ax handle. 
    '''
    #Declare the land and ocean parameters
    LAND_highres = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m',
    edgecolor='black',
    facecolor=('lightgrey'),
    linewidth=1)
    OCEAN_highres = cartopy.feature.NaturalEarthFeature('physical', 'ocean', '10m',
    facecolor='white')
    # Set opacity to 0.5
    ax.background_patch.set_alpha(0.5)

    #Declare the lat and lon boundaries for the map and data
    domain = [99, 99, -99, -99]
    # domain[0] = np.min(lat) # South
    # domain[1] = np.min(lon) # West
    # domain[2] = np.max(lat) # North
    # domain[3] = np.max(lon) # East
#     domain = list(np.array(domain) + np.array([+1, +6, 0, -4]))
#     domain = [54.5, -61.30979545701268, 62, -54.47452933956656]
    domain = [50, -51, 72, -63]

    aoi_map = [domain[0], domain[2], domain[1], domain[3]]
    print(domain)
#     # Plot results
#     transform = rot.transform_points(rot,lon,lat)
#     x_n = transform[...,0]
#     y_n = transform[...,1]

    ax.add_feature(LAND_highres)
    ax.add_feature(OCEAN_highres)
    ax.set_extent([aoi_map[2], aoi_map[3], aoi_map[0], aoi_map[1]])
#     ax.fill_between([0,100000],[0,100000], color="none", hatch="..", edgecolor="k", linewidth=0.0,transform=ccrs.PlateCarree(),zorder=4)
    
    # Fill land and coastline
#     plt.pcolormesh(lon,lat,land,transform=ccrs.PlateCarree(),zorder=4,cmap = matplotlib.colors.ListedColormap(['lightgrey', 'silver']))
#     plt.contour(lon,lat,land,colors='black',transform=ccrs.PlateCarree(),linewidth=0.2,zorder=4)

    ax.coastlines(resolution='10m',linewidth=0.5)
    
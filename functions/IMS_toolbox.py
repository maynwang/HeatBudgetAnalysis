import glob
import numpy as np
import xarray as xr
import pandas as pd
import os
import scipy
from pyrsktools import RSK
# import sympy as sp
import inspect
from matplotlib.colors import LinearSegmentedColormap



# Ice Monitoring Site toolbox


def find_unique_filenames(path):
    '''Search in subdirectories for unique filenames'''
    unique_filenames = []

    # Use glob to get all files in the directory and its subdirectories
    all_files = glob.glob(path, recursive=True)

    for file in all_files:
        if os.path.isfile(file):  # Check if it's a file (not a directory)
            filename = os.path.basename(file)

            if filename not in unique_filenames:
                unique_filenames.append(file) # List of filepaths for each unique file
                
    return unique_filenames

    # if __name__ == "__main__":
    #     files = find_unique_filenames(path)


def load_CTD_cast(csv_filename):
    '''Load a single CTD cast from a CSV and convert the time to datetime'''
    # Load CTD data
    data = pd.read_csv(filename)
    matlab_datenums = data['Time'].astype(float)

    # Convert MATLAB serial dates to datetime
    data['Time'] = pd.to_datetime(matlab_datenums-719529, unit='D') 

    return data


def load_RSK(RSK_filename, t1, t2):
    '''
    Load RSK file for IMS data
    Convert conductivity to salinity and pressure to depth. Subset times to match IMS timeseries
    '''

    rsk = RSK(RSK_filename)
    rsk.open()
    # Start on the 26th to match weather station data (the shortest time series), even though data starts on 23rd
    rsk.readdata(t1, t2)
    if 'conductivity' in rsk.channelNames:
        rsk.derivesalinity() # add salinity to channels
    rsk.deriveseapressure()
    rsk.derivedepth()
    return rsk

def DoodsonX0(ts):
    '''
    Running Avgerage on data of using Doodson X0 filter.
    Filter weights are designed to remove the main tidal
    consituents of hourly data. Filter is 39 points long
    and so both the first and last 19 points of 'avg' are
    made into NaNs.

    Filter found here: http://www.pol.ac.uk/ntslf/acclaimdata/gloup/doodson_X0.html

    Adapted from MATLAB code by Eric Oliver, 14 Aug 2019
    '''
    X0 = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 1, 0, 2, 1, 1, 2, 0, 2, 1, 1, 2, 0, 1, 1, 0, 2, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1])/30.
    ts_smooth = np.convolve(ts, X0, mode='same')
    ts_smooth[:19] = np.nan 
    ts_smooth[-19:] = np.nan
    return ts_smooth

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def stretch1d(y,arr_size):
    '''
    stretch numpy array "y" to match another array's ("arr_size") size through linear interpolation 
    '''
    # interpolate across nans first
    nans,x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans]) 

    # Interpolate updated arrays to match new array size
    y_interp = scipy.interpolate.interp1d(np.arange(y.size),y)
    y_stretch = y_interp(np.linspace(0,y.size-1,arr_size.size))

    return y_stretch


def load_Rway_station_data(file, var_str):
    '''
    Load Robert Way's weather station data into an xarray datarray (one variable only)
    '''
    # Make a new xarray with only time as datetime and air temp as floats

    file_xarr = file.to_xarray()
    time = file_xarr['time'][1:]
    var_ = file_xarr[var_str][1:]
    ds = xr.DataArray(
        data=var_.astype(float), # convert from object to float
        dims=["time"],
        coords=dict(
            time=pd.to_datetime(time).values, # convert to datetime
        ),
        # attrs=dict(
        #     description="Average temperature",
        #     units="degC",
        # ),
    )

    return ds


def load_postville_weather():
    '''
    Load Robert Way's weather station data into an xarray datarray
    '''

    filepath = './Weather/data/postville_weather_2020-2024.csv'
    df = pd.read_csv(filepath,low_memory=False)

    # remove unecessary stuff
    df_ = df.drop(index=0) # row of units
    df_ = df_.drop(['longitude','latitude','station_name','stat_num','wsc_num'],axis=1)
    # Convert time to datetime (not sure why this is necessary, but must be done)
    df_['time'] = pd.to_datetime(df_['time'])
    df_ = df_.set_index('time')
    # Convert to xarray
    ds = df_.to_xarray()
    ds =  ds.astype(float)
    # Convert ot datetime64 
    ds['time'] = pd.DatetimeIndex(ds['time'].values) 

    return ds



def fit_linear_2d(x_2d, y_2d):
    '''
    Fit linear regression function through a set of 2-d data arrays
    '''
    # Flatten the 2D datasets
    x_flat = x_2d.values.flatten()
    y_flat = y_2d.values.flatten()
    
    # Create a mask where both x and y are not NaN
    mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)
    
    # Apply the mask to filter valid values
    x_valid = x_flat[mask]
    y_valid = y_flat[mask]
    
    # Perform linear regression on the valid data
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_valid, y_valid)
    
    return slope, intercept, r_value, p_value, std_err



def cross_corr(ts1, ts2):
    """
    Compute the cross-correlation between two time series.

    Parameters:
    ts1 (array-like): First time series.
    ts2 (array-like): Second time series (must have the same length as ts1).
    max_lag (int, optional): Maximum lag for cross-correlation. Default is length of the series - 1.

    Returns:
    lags (np.ndarray): Array of lag values.
    correlation (np.ndarray): Cross-correlation values at each lag.
    """

    # Calculate mean and standardize the series (zero mean for normalization)
    ts1_mean = np.nanmean(ts1)
    ts2_mean = np.nanmean(ts2)
    ts1_std = np.std(ts1)
    ts2_std = np.std(ts2)

    ts1_norm = (ts1 - ts1_mean) / ts1_std
    ts2_norm = (ts2 - ts2_mean) / ts2_std

    # Compute cross-correlation using numpy.correlate
    # MAsk out nans
    valid = ~np.isnan(ts1*ts2)
    corr = np.correlate(ts1_norm[valid], ts2_norm[valid], mode='full')
    
    # Normalize by length
    corr /= len(ts1)
    
    # Define lags
    # the valid mask is only good if the nans are at the edges...
    lags = np.arange(-len(ts1[valid]) + 1, len(ts1[valid]))
    

    return lags, corr




# General error propagation function
def error_propagation_func(func, variables, uncertainties=None):
    '''
    Calculate the uncertainty of a function output using generic error propagation

    sig_y = sqrt[(dy/da * sig_a)^2 + (dy/db * sig_b)^2 + ...]

    Parameters: 
    - a function (func), 
    - an array of variables that are the inputs to func (variables), 
    - the uncertainties associated with each variable (uncertainties)
    
    Returns: the propagated uncertainty
    '''
    # Get the names of the variables from the function signature
    signature = inspect.signature(func)
    variable_names = list(signature.parameters.keys())  # Extract parameter names from function signature
    
    # Define symbolic variables (same as input function's variables)
    symbols = sp.symbols(variable_names)

    # Define the symbolic function 
    f_sym = func(*symbols)
    
    # Compute the partial derivatives of the function with respect to each variable
    partial_derivatives = [sp.diff(f_sym, symbol) for symbol in symbols]
    
    # Compute the uncertainties if provided
    uncertainty = 0
    if uncertainties:
        for i, partial in enumerate(partial_derivatives):
            # Evaluate the partial derivative at the input values (substitute actual variable values)
            partial_value = np.array([float(partial.subs(dict(zip(symbols, variables_vals)))) 
                                     for variables_vals in zip(*variables)])
            uncertainty += (partial_value * uncertainties[i])**2
        
        uncertainty = np.sqrt(uncertainty)
    
    return uncertainty


def error_propagation(partial_derivatives,uncertainties):
    """
    Calculate the total uncertainty from partial derivatives and uncertainties.

    Parameters:
    partial_derivatives (list): List of partial derivatives of the function with respect to each variable.
    uncertainties (list): List of uncertainties corresponding to the variables.

    Returns:
    float: The total propagated uncertainty.
    """

    # Initialize total uncertainty
    total_uncertainty = 0
    
    # Sum the squared contributions of each term
    for i in range(len(partial_derivatives)):
        total_uncertainty += (partial_derivatives[i] * uncertainties[i]) ** 2
    
    # Return the square root of the total sum of squared uncertainties
    return np.sqrt(total_uncertainty)




def sensible_heat_flux(U_wd, T_s, T_a, sigma_Uwd=None, sigma_Ts=None, sigma_Ta=None):
    '''
    Calculate Fsens with optional error calculation
    '''
    # fsh = 1.22 * 1005 * 1.75 * 10**(-3)
    c_H = 2e-3 # Konda & Yamazawa, 1986; Andreas 2002
    rho_a = 1.22 #kg/m3
    c_p = 1005 # Specific heat capacity of air
    fsh = c_H * rho_a * c_p
    # If T_a > T_s, flux points down (negative)
    F_sens = fsh * U_wd * (T_s - T_a)

    if sigma_Uwd is not None:
        dFs_dUwd = fsh*(T_s - T_a)
        dFs_dTs = fsh
        dFs_dTa = -fsh

        # Total uncertainty propagation
        sigma_Fsens = np.sqrt(
            (dFs_dUwd * sigma_Uwd)**2 +
            (dFs_dTs * sigma_Ts)**2 +
            (dFs_dTa * sigma_Ta)**2
        )

        return F_sens, sigma_Fsens
    else:
        return F_sens



def latent_heat_flux(U_wd,RH,P,T_s,T_a):

    '''
    Calculate latent heat flux using the bulk aerodynamic formula
    '''
    rho_a = 1.22 # kg/m3 - atmospheric density
    L_s = 2.83e6 # J/kg - latent heat of sublimation (or vaporization)
    c_E = 1.6e-3 # Latent heat transfer coefficient (value from Vihma et al. 2009)


    e_s_air = calculate_es_ice(T_a) # In air
    e_s_snow = calculate_es_ice(T_s) # Snow surface

    # Specific humidity of air calculated from relative humidity
    e_ = RH/100 * e_s_air

    q_a = calculate_qs(P,e_) # Actual specific humidity, g/kg
    q_s = calculate_qs(P,e_s_snow) # Snow surface specific humidity, Assume saturated so RH = 1\

    return rho_a*L_s*c_E*U_wd*(q_s - q_a)



def calculate_es_ice(T):
    # Calculate the saturated vapour pressure over ice using Arden-Buck equations (1996)
    # units: Pa
    # T must be in deg. C
    e_s_ice = 6.1115*np.exp((23.036 - T/333.7)*(T/(279.82 + T))) * 100 #hPa to Pa
    return e_s_ice


def calculate_es_water(T):
    # Calculate the saturated vapour pressure over water using Arden-Buck equations (1996)
    # units: Pa
    # T must be in deg. C
    e_s_water = 6.1121*np.exp((18.678 - T/234.5)*(T/(257.14 + T))) * 100 # hPa to Pa
    return e_s_water

# Calculate specific humidity or air (q_a) and snow surface (q_s)

def calculate_qs(P,e_s):
    # Calculate the specific humidity given pressure (P) and the saturated vapour pressure (e_s)
    # Units: kg/kg
    # P is given in mbar, then converted here to Pa (1mbar = 100Pa)
    P_Pa = P*100
    # 0.622 is the ratio of water vapour:air mass
    q_s = 0.622*e_s/(P_Pa-0.378*e_s) 
    return q_s



def add_to_dataset(ds, name, values, time=None, attrs=None):
    """
    Add a 1D timeseries to an xarray.Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        The original dataset to which the timeseries will be added.
    name : str
        Name of the new variable to add.
    values : array-like
        Data values for the timeseries (must align with `time`).
    time : array-like or None
        Time coordinate for the timeseries. If None, will use ds.time.
    attrs : dict or None
        Optional attributes to assign to the new variable.

    Returns:
    --------
    xr.Dataset
        A new dataset with the timeseries added.
    """

    time = ds['time'].values

    da = xr.DataArray(values, coords={'time': time}, dims='time', name=name)
    if attrs:
        da.attrs.update(attrs)

    return ds.assign({name: da})


def RMSE(y_true,y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return(rmse)

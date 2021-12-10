import numpy as np
import h5py
import os
from scipy.io import loadmat
from tike.ptycho.io import position_units_to_pixels

def determine_Niter(path2search):
    folder1 = os.listdir(path2search)
    path2search_2 = f'{path2search}{folder1[0]}'
    folder2 = os.listdir(path2search_2)
    folder2 = [int(x[5:-4]) for x in folder2 if 'Niter' in x]
    file = f'{path2search_2}/Niter{sorted(folder2)[0]}.mat'
    return file


def import_diffraction_data(file):
    """
    Import diffraction data from Lami beamline.
    
    Parameters
    ----------
    file : str
        path to the hdf5 file to import
    
    Returns
    -------
    nd.array
        diffraction data from the experiment
    """
    return h5grab_data(file, 'entry/data/eiger_4')

def center_crop_with_fft_shift(data, center_x, center_y, crop_size):
    """
    Crops central diffraction pattern and fft shifts data
    
    Parameters
    ----------
    data : nd.array
        the diffraction data to crop and fftshift
    center_x : int
        the x position on where to center
    center_y : int
        the y position on where to center
    crop_size : int
        one side length of a square to crop to
        
    Returns
    -------
    nd.array
        cropped and fftshifted data diffraction data
    """
    size = int(crop_size / 2)
    data = data[:, center_x-size:center_x+size, center_y-size:center_y+size]
    return data, np.fft.ifftshift(np.array(data), axes=(1, 2))
    
def import_probe(mat_file, useH5=False):
    """
    Obtain the first probe value from the matlab file generated from the beamline
    
    Parameters
    ----------
    mat_file : str
        path to the matlab file to read
    
    Returns
    -------
    nd.array
        the first probe in the array
    """
    
    if not useH5:
        probe_all = loadmat(mat_file)
        probe = probe_all['probe'][:,:,0,0]
    
    elif useH5:
        with h5py.File(mat_file, 'r') as f:
            data = f.get('probe')
            loc = data[0][0]
            probe = np.array(f[loc])
    probe = probe['real'] + probe['imag']*1j
    probe = probe.T
    return probe[None, None, None]

def import_scan_paths(file, dat_file, skiprows=2, x_idx=2, y_idx=5, meter_convert=1e-6,
                      dist2det=2.97, ev=8800, detector_pixel_count=256, path_corr=[-1, -1]):
    
    """
    Import correct scan positions in pixels
    
    Parameters
    ----------
    file : str
        path to the hdf5 file to import
    dat_file : str
        the location of the scan positions file
    skiprows : int
        amount of rows to skip in the file
    x_idx : int
        the index on where to find the x coordinates
    y_idx : int
        the index on where to find the y coordinates
    meter_convert : float
        multiplier for the scan positions to turn them into meters
    dist2det : float
        distance to the detector
    ev : float
        the ev used durrring the experiment
    detector_pixel_count : int
        the number of pixels across one edge of the detector
        
    Returns
    -------
    nd.array
        scan positions in terms of pixles on the detector.
    """

    scan = np.loadtxt(dat_file, unpack=False, skiprows=skiprows)[:, (y_idx, x_idx)] * meter_convert 
    scan = scan * np.array([path_corr])
    pix_size = h5read_attr(file, 'entry/data/eiger_4', "Pixel_size")
    scan = position_units_to_pixels(scan, dist2det, detector_pixel_count, pix_size[0][0], ev)
    return scan



    
def setup_diffraction_data():
    """
    Obtain the 2D diffraction data for a single scan.
    Dimensions should be x_pos, y_pos, x_diff_dim, y_diff_dim.
    """
    pass

def setup_realspace_data():
    """
    Obtrain 2D real space data for each scan point of a ptcyh experiment.
    Dimensions should be x_pos, y_pos, x_realspace_dim, y_realspace_dim.
    """
    pass


def h5group_list(file, group_name='base'):
    """Displays all group members for a user defined group
    Parameters
    ==========
    file (str)
        the path to the hdf5 file
    group_name (str)
        the path to the group the user wants the Keys for. Set to 'base' if you want the top most group
    Returns
    =======
    a list of all the subgroups inside the user defined group
    """
    # Parenthesis are needed - keep them
    with h5py.File(file, 'r') as hdf:
        if group_name == 'base':
            return (list(hdf.items()))
        else:
            g1=hdf.get(group_name)
            return (list(g1.items()))
        
        
def h5grab_data(file, data_loc):
    """Returns the data stored in the user defined group
    Parameters
    ==========
    file (str):
        the user defined hdf5 file
    data_loc (str):
        the group the user would like to pull data from
    Returns
    =======
    the data stored int the user defined location
    """
    with h5py.File(file, 'r') as hdf:
        data = hdf.get(data_loc)
        data = np.array(data)

    return data

def h5read_attr(file, loc, attribute_name):
    """Read an attribute from a user selected group and attribute name
    Parameters
    ==========
    file (str)
        the path to the hdf5 file
    loc (str)
        the location to the group inside the hdf5 file
    attribute_name (str)
        the name of the attribute
    Returns
    =======
    Attribute value
    """
    with h5py.File(file, 'a') as hdf:
        data = hdf.get(loc)
        return data.attrs[attribute_name]
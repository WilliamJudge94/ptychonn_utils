def tike_saving_for_ptychonn_postrecon(save_dir, scan_num, diffraction_data, tike_result, patch_width=256):
    """
    Save TIKE data (after recon) in the proper format for PtychoNN.
    
    Parameters
    ----------
    save_dir : str
        path to the main dir that holds all the scan save dirs
    scan_num : int
        the scan number to save the data to
    diffraction_data : nd.array
        the array holding the cropped diffraction data
    tike_result : dict
        the output from the tike reconstruction
    patch_width : int
        assumes square patch, patchifiies tike recon for training
    
    Returns
    -------
    None
        Saves tike data to npy files
    """
    from tike.ptycho.learn import extract_patches
    import numpy as np
    
    patches = extract_patches(tike_result['psi'], tike_result['scan'], patch_width=patch_width)
    # needed for ptychoNN
    np.save(f'{save_dir}{scan_num}/patched_psi.npy', patches)
    np.save(f'{save_dir}{scan_num}/scan_pixel_positions.npy', result['scan'])
    np.save(f'{save_dir}{scan_num}/cropped_exp_diffr_data.npy', diffraction_data)
    
def tike_saving_for_ptychonn_prerecon(save_dir, scan_num, diffraction_data, scan_positions):
    """
    Save TIKE data (pre recon) in the proper format for PtychoNN.
    
    Parameters
    ----------
    save_dir : str
        path to the main dir that holds all the scan save dirs
    scan_num : int
        the scan number to save the data to
    diffraction_data : nd.array
        the array holding the cropped diffraction data
    scan_positions : tike dict_entry
        the output from the tike reconstruction
    
    Returns
    -------
    None
        Saves tike data to npy files
    """
    import numpy as np
    
    np.save(f'{save_dir}{scan_num}/scan_pixel_positions.npy', scan_positions)
    np.save(f'{save_dir}{scan_num}/cropped_exp_diffr_data.npy', diffraction_data)

    
def setup_prediction(save_dir, scan_num, h5data_file, position_dat_file,
                     dist2det, ev, crop_size, center_x, center_y, 
                     skiprows=2, x_idx=2, y_idx=5,
                     meter_convert=1e-6, path_corr=[-1, -1], ):
    """
    Predicting using PtychoNN requires scan positions and raw data.
    This functions sets the correct format for PtychoNN.
    
    Parameters
    ----------
    save_dir : str
        the path to the directory where the scan data is held
    scan_num : int
        the scan number to predict
    h5data_file : str
        the full path to the scan_num data file
    position_dat_file : str
        the full path to the positional .dat file for the scan
    dist2det : float
        the distance to the detector in meters
    ev : float
        the eV the scan was taken at
    crop_size : int
        the amount of square pixles to cropp the diffraction pattern to
    center_x : int
        x pixel position to center around
    center_y : int
        the y pixel position to center around
    skiprows : int
        the amount of rows to skip when reading the .dat positional file
    x_idx : int
        the column index in the .dat positional file to find the x scan positions
    y_idx : int
        the column index in the .dat positional file to find the y scan positions
    meter_convert : float
        value multiplied to the scan positions to get them into meters
    path_corr : array
        multiplier values for the scan positions of [x, y]
        
    Returns
    -------
    None
        save scan position and cropped diffraction data to the appropriate
        locations to be used during PtychoNN predictions.
    """
    
    # Scan Positions
    scan = np.loadtxt(position_dat_file,
                      unpack=False,
                      skiprows=skiprows)[:, (y_idx, x_idx)] * meter_convert
    scan = scan * np.array([path_corr])
    
    with h5py.File(h5data_file, 'a') as hdf:
        data = hdf.get('entry/data/eiger_4')
        pix_size = data.attrs["Pixel_size"]

    scan_positions = position_units_to_pixels(scan, dist2det,
                                              crop_size,
                                              pix_size[0][0], ev)

    # Diffraction Data
    with h5py.File(h5data_file, 'r') as hdf:
        data = hdf.get('entry/data/eiger_4')
        data = np.array(data)
        
    # Cropping
    size = int(crop_size / 2)
    if crop_size % 2 != 0:
        raise ValueError('Crop size is not an even number.')
    diffraction_data = data[:, center_x-size:center_x+size, center_y-size:center_y+size]
    
    # Saving
    np.save(f'{save_dir}{scan_num}/scan_pixel_positions.npy', scan_positions)
    np.save(f'{save_dir}{scan_num}/cropped_exp_diffr_data.npy', diffraction_data)
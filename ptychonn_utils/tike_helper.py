def tike_saving_for_ptychonn(save_dir, scan_num, diffraction_data, tike_result, patch_width=256):
    """
    Save TIKE data in the proper format for PtychoNN.
    
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
import typer
import pathlib

import logging

import matplotlib.pyplot as plt
import numpy as np

import tike
import tike.ptycho
import tike.view
from tike.ptycho.learn import extract_patches

import sys
import tifffile as tif
from scipy.io import loadmat
from skimage.restoration import unwrap_phase
from tike.ptycho.io import position_units_to_pixels

base_path = '/projects/hp-ptycho/wjudge/'
sys.path.append(f"{base_path}ptychonn/")
from ptychnn_theta import *


app = typer.Typer()

def full_function(base_path, data_num):
    
    data_path = f'{base_path}ptychonn/bicer_data/28idc_2021-10-12/eiger_4/S00000-00999/S00{data_num}/run_00{data_num}_000000000000.h5'

    data = import_diffraction_data(data_path)

    sx = 450 + 112 - 3
    sy = 725 + 138 - 2

    og, data = center_crop_with_fft_shift(data,
                                          center_x=sx,
                                          center_y=sy,
                                          crop_size=256)
    print('Extracted Diff')

    mat_loc = f'{base_path}ptychonn/bicer_data/28idc_2021-10-12/ptycho_reconstruct/catalyst/S00{data_num}/roi0_Ndp256/'
    mat_loc = determine_Niter(mat_loc)
    probe = import_probe(mat_file=mat_loc, useH5=True)
    print('Extracted Probe')
    
    scan_path = f'{base_path}ptychonn/bicer_data/28idc_2021-10-12/scan_positions/scan_00{data_num}.dat'
    scan = import_scan_paths(file=data_path, dat_file=scan_path, skiprows=2,
                     x_idx=2, y_idx=5, meter_convert=1e-6,
                      dist2det=2.97, ev=8800, detector_pixel_count=256)
    print('Extracted Scan Pos')
    
    probe_options = tike.ptycho.ProbeOptions()  # uses default settings for probe recovery
    object_options = tike.ptycho.ObjectOptions(
        # The object will be updated.
        positivity_constraint=0.03,  # smoothness constraint will use our provided setting
        # other object options will be default values
    )

    position_options = None # indicates that positions will not be updated


    logging.basicConfig(filename=f'{base_path}ptychonn/scan_dataml/{data_num}/tike_{data_num}.log',
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)

    # Provide initial guesses for parameters that are updated
    result = {
        # initial guess for psi can be automatically generated
        'probe': probe.astype('complex64'),
        'scan': scan.astype('float32'),
    }

    #logging.basicConfig(level=logging.INFO)

    result = tike.ptycho.reconstruct(
        data=data.astype('float32'),
        **result,
        algorithm='cgrad',
        num_iter=128,
        batch_size=data.shape[-3]//3,
        probe_options=probe_options,
        object_options=object_options,
        #position_options=position_options,
    )

    print(f'{data_num} - Finised!')

    phase = np.angle(result['psi'])
    ampli = np.abs(result['psi'])
    phase = np.asarray(phase)
    ampli = np.asarray(ampli)
    
    def tike_saving_for_ptychonn(save_dir, scan_num, diffraction_data, tike_result, patch_width=256):
        from tike.ptycho.learn import extract_patches
        patches = extract_patches(tike_result['psi'], tike_result['scan'], patch_width=patch_width)
        # needed for ptychoNN
        np.save(f'{save_dir}{scan_num}/patched_psi.npy', patches)
        np.save(f'{save_dir}{scan_num}/scan_pixel_positions.npy', result['scan'])
        np.save(f'{save_dir}{scan_num}/cropped_exp_diffr_data.npy', diffraction_data)
        
    tike_saving_for_ptychonn('/projects/hp-ptycho/wjudge/ptychonn/scan_dataml/', scan_num=data_num,
                             diffraction_data=og, tike_result=result, patch_width=256)
    
    # Not needed for ptycho
    np.save(f'/projects/hp-ptycho/wjudge/ptychonn/scan_dataml/{data_num}/tike_recon_phase.npy', phase)
    np.save(f'/projects/hp-ptycho/wjudge/ptychonn/scan_dataml/{data_num}/tike_recon_ampli.npy', ampli)
    
    print('Saved')

@app.command()
def tike_recon(start_scan: int = typer.Argument(..., help="beginning scan"),
               end_scan: int = typer.Argument(..., help="ending scan")):
    base_path = '/projects/hp-ptycho/wjudge/'
    for data_num in range(start_scan, end_scan+1):
        full_function(base_path, data_num)
         
        
if __name__ == "__main__":
    app()



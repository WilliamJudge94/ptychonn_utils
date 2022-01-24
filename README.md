# ptychonn_utils

The following repository lays the groundwork for PtychoNN to be integrated into a larger workflow. Below is a step by step guide on how to properly get it running.


## Conda Env Setup

There is a conda env placed in the "envs" directory. Use this to install all dependecies needed to run the workflow.


## Data Setup

The following inplementation requires a specific data setup. This set up includes the following:

1. a main data path to store all scan data to - {save_dir}
2. a corresponding scan number directory inside this main path to all scan numbers collected. They must be integers! - {scan_num}
3. please view ptychonn_utils.tike_helper for the following functions tike_saving_for_ptychonn_postrecon(), tike_saving_for_ptychonn_prerecon()
4. These functions are to be guides on how to properly save the required data for PtychoNN


## Training

from network_utils import *
from ptychnn_jupyter import train_ptychonn, predict_ptychonn

data_path = '/projects/hp-ptycho/wjudge/ptychonn/scan_dataml/'

train_ptychonn(data_path, start_scan=204, end_scan=238,
                   train_lines=0.85, val_lines=0.15, epochs=60, gpu_select='2', batch_size=64,
                   load_model_scan=-1, save_model_scan=-1, ngpus=1, lr=1e-3, height=192, width=192,
                   mean_phsqr=0.02, verbose=False)
                   
                   
## Prediction

predict_ptychonn(data_path,
                pred_scan=457,
                network_scan=238, 
                ngpus=1, gpu_select='1', batch_size=64, height=192,
                width=192, verbose=False)
                
                
# Obtaining Predictions

import numpy as np
p_amp = np.load(f'{data_path}{scan_num}/prediction_amp.npy')
p_phase = np.load(f'{data_path}{scan_num}/prediction_phi.npy')


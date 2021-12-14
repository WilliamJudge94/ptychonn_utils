import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

import os
import matplotlib
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from numpy.fft import fftn, fftshift
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark=False


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


def setup_training_params(EPOCHS,
                          NGPUS=torch.cuda.device_count(),
                          BATCH_SIZE=64,
                          LR=1e-3,
                          verbose=False):
    """
    Setup a portion of PtychoNN's training parameters.
    
    
    Parameters
    ----------
    EPOCHS : int
        the amount of epochs to train for.
        
    NGPUS : int
        the amount of gpu's to train with.
        
    BATCH_SIZE : int
        the batch size during training.
        
    LR : float
        the learning rate during training.
        
    verbose : bool
        Display useful information.
        
    Returns
    -------
    list
        a list of the following items (EPOCHS, NGPUS, BATCH_SIZE, LR)

    """

    EPOCHS = EPOCHS
    BATCH_SIZE = NGPUS * BATCH_SIZE
    LR = NGPUS * LR
    
    if verbose:
        print("GPUs:", NGPUS, "Batch size:", BATCH_SIZE, "Learning rate:", LR)
        
    return EPOCHS, NGPUS, BATCH_SIZE, LR


def setup_model_path(data_path, save_model_scan):
    """
    Setup the trained model path.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    str
        the path to the trained model save location
    """

    MODEL_SAVE_PATH = f'{data_path}{save_model_scan}/trainded_model/'
    if (not os.path.isdir(MODEL_SAVE_PATH)):
        os.mkdir(MODEL_SAVE_PATH)
        
    return MODEL_SAVE_PATH


def load_data(data_diffr_loc, real_space_loc):

    data_diffr = np.load(data_diffr_loc).astype('float32')
    real_space = np.load(real_space_loc)
    amp = np.abs(real_space).astype('float32')
    ph = np.angle(real_space).astype('float32')
    
    return data_diffr, amp, ph

def load_dataV2(main_path, start_scan, end_scan, mean_phsqr_val=0.02, im_shape=(256, 256), torp='train'):
    """
    Load in the saved training data from TIKE
    
    Parameters
    ----------
    main_path : str
        path to a folder housing scan number folders housing the data to load
    start_scan : int
        the first scan number to load data from
    end_scan : int
        the final scan number to load data from
    mean_phsqr_val : float
        the square of the phase image must be above this value to be placed into training
    im_shape : list
        reshape all training images to this dimension
        
    Returns
    -------
    list of nd.array
        the diffraction patterns,
        the amplidudes of the real-space images,
        the phase of the real-space image.
    """
    
    r_space = []
    dif_data = []
    amp_data = []
    ph_data = []

    for scan_num in tqdm(range(start_scan, end_scan+1), position=0, leave=False, desc='Loading Scans'):
        if torp == 'train':
            real_space = np.load(f"{main_path}/{scan_num}/recon_data.npy")
            print(np.shape(real_space))
            r_space.append(real_space)
            phase = np.angle(real_space)
            ampli = np.abs(real_space)
            amp_data.append(ampli)
            ph_data.append(phase)
            
        data_diffr = np.load(f"{main_path}/{scan_num}/diff_data.npy")
        print(np.shape(data_diffr))
        dif_data.append(data_diffr)
        
    if len(dif_data) != 1:
        total_data_diff = np.concatenate(dif_data)
    else:
        _ = np.asarray(dif_data, dtype='float32')
        _ = _[0, :, :, :]
        total_data_diff = _[:,np.newaxis,:,:]
        
    if torp == 'train':
        tqdm_total = 3
    else:
        tqdm_total = 1
        
    with tqdm(total=tqdm_total, leave=False, desc='Resizeing Data') as pbar:
    
        if torp == 'train':
            total_data_amp = np.concatenate(amp_data)
            total_data_phase = np.concatenate(ph_data)
            av_vals = np.mean(total_data_phase**2, axis=(1, 2))
            idx = np.argwhere(av_vals >= mean_phsqr_val)
            total_data_diff = total_data_diff[idx].astype('float32')
            total_data_amp = total_data_amp[idx].astype('float32')
            total_data_phase = total_data_phase[idx].astype('float32')
            print(np.shape(total_data_phase))
            _ = input()
            total_data_amp = resize(total_data_amp, (total_data_amp.shape[0], 1, im_shape[0], im_shape[1]))
            pbar.update(1)
            total_data_phase = resize(total_data_phase, (total_data_phase.shape[0], 1, im_shape[0], im_shape[1]))
            pbar.update(1)
        else:
            total_data_amp = []
            total_data_phase = []

        total_data_diff = resize(total_data_diff, (total_data_diff.shape[0], 1, im_shape[0], im_shape[1]))
        pbar.update(1)
    
    return total_data_diff, total_data_amp, total_data_phase


def load_dataV3(main_path, start_scan, end_scan, mean_phsqr_val=0.0, im_shape=(256, 256), torp='train'):
    """
    Load in the saved training data from TIKE
    
    Parameters
    ----------
    main_path : str
        path to a folder housing scan number folders housing the data to load
    start_scan : int
        the first scan number to load data from
    end_scan : int
        the final scan number to load data from
    mean_phsqr_val : float
        the square of the phase image must be above this value to be placed into training
    im_shape : list
        reshape all training images to this dimension
        
    Returns
    -------
    list of nd.array
        the diffraction patterns,
        the amplidudes of the real-space images,
        the phase of the real-space image.
    """
    
    r_space = []
    dif_data = []
    amp_data = []
    ph_data = []

    for scan_num in tqdm(range(start_scan, end_scan+1), position=0, leave=False, desc='Loading Scans'):
        if torp == 'train':
            real_space = np.load(f"{main_path}/{scan_num}/patched_psi.npy")
            r_space.append(real_space)
            phase = np.angle(real_space)
            ampli = np.abs(real_space)
            amp_data.append(ampli)
            ph_data.append(phase)
            
        data_diffr = np.load(f"{main_path}/{scan_num}/cropped_exp_diffr_data.npy")
        dif_data.append(data_diffr)
    
        
    if len(dif_data) != 1:
        total_data_diff = np.concatenate(dif_data)
    else:
        _ = np.asarray(dif_data, dtype='float32')
        _ = _[0, :, :, :]
        total_data_diff = _[:,np.newaxis,:,:]
        
    if torp == 'train':
        tqdm_total = 3
    else:
        tqdm_total = 1
        
    with tqdm(total=tqdm_total, leave=False, desc='Resizeing Data') as pbar:
    
        if torp == 'train':
            total_data_amp = np.concatenate(amp_data)
            total_data_phase = np.concatenate(ph_data)
            av_vals = np.mean(total_data_phase**2, axis=(1, 2))
            idx = np.argwhere(av_vals >= mean_phsqr_val)
            total_data_diff = total_data_diff[idx].astype('float32')
            total_data_amp = total_data_amp[idx].astype('float32')
            total_data_phase = total_data_phase[idx].astype('float32')
            if total_data_amp.shape[0] == 0:
                raise ValueError(f"Please reduce mean_phsqr_val value. It is set to high.")
            total_data_amp = resize(total_data_amp, (total_data_amp.shape[0], 1, im_shape[0], im_shape[1]))
            pbar.update(1)
            total_data_phase = resize(total_data_phase, (total_data_phase.shape[0], 1, im_shape[0], im_shape[1]))
            pbar.update(1)
        else:
            total_data_amp = []
            total_data_phase = []

        total_data_diff = resize(total_data_diff, (total_data_diff.shape[0], 1, im_shape[0], im_shape[1]))
        pbar.update(1)
    
    return total_data_diff, total_data_amp, total_data_phase


def plot3(data,titles):
    if(len(titles)<3):
        titles=["Plot1", "Plot2", "Plot3"]
    fig,ax = plt.subplots(1,3, figsize=(20,12))
    im=ax[0].imshow(data[0])
    ax[0].set_title(titles[0])
    ax[0].axis('off')
    plt.colorbar(im,ax=ax[0], fraction=0.046, pad=0.04)
    im=ax[1].imshow(data[1])
    ax[1].set_title(titles[1])
    ax[1].axis('off')
    plt.colorbar(im,ax=ax[1], fraction=0.046, pad=0.04)
    im=ax[2].imshow(data[2])
    ax[2].set_title(titles[2])
    ax[2].axis('off')
    plt.colorbar(im,ax=ax[2], fraction=0.046, pad=0.04)
  

def update_saved_model(model, path, NGPUS):
    """
    Update the saved model.
    
    Parameters
    ----------
    model : torch.model
        the model to save
    path : str
        path to save the model to
    NGPUS : int
        the amount of gpu's used during training
    
    Returns
    -------
    None
        only saves the model to the desired location
    """
    
    if not os.path.isdir(path):
        os.mkdir(path)
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))
    if (NGPUS>1):    
        
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), path+'best_model.pth')
        else:
            torch.save(model.state_dict(), path+'best_model.pth')
            
        #model = torch.nn.DataParallel(model).cuda()
        #torch.save(model.module.state_dict(), path+'best_model.pth') #Have to save the underlying model else will always need 4 GPUs
    else:
        torch.save(model, path+'best_model.pth')
    
    
def train_nn(trainloader, metrics, device, model, criterion, optimizer, scheduler):
    """
    Train the created model
    
    Parameters
    ---------
    trainloader : torch.dataloader
        the trainind data dataloader
    metrics : dic
        the metrics dictionary
    device : torch.device
        the pytorch device to use
    model : torch.model
        the pytorch model to train
    criterion : unknown
        unknown
    optimizer : torch.optimizer
        the optimizer to use during training
    scheduler : unknown
        unknown
    
    Returns
    -------
    None
    """
    
    tot_loss = 0.0
    loss_amp = 0.0
    loss_ph = 0.0
    
    for i, (ft_images,amps,phs) in tqdm(enumerate(trainloader), position=1, leave=False, total=len(trainloader), desc='Batches'):
        ft_images = ft_images.to(device) #Move everything to device
        amps = amps.to(device)
        phs = phs.to(device)

        pred_amps, pred_phs = model(ft_images) #Forward pass

        #Compute losses
        loss_a = criterion(pred_amps,amps) #Monitor amplitude loss
        loss_p = criterion(pred_phs,phs) #Monitor phase loss but only within support (which may not be same as true amp)
        loss = loss_a + loss_p #Use equiweighted amps and phase

        #Zero current grads and do backprop
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        tot_loss += loss.detach().item()
        loss_amp += loss_a.detach().item()
        loss_ph += loss_p.detach().item()

        #Update the LR according to the schedule -- CyclicLR updates each batch
        scheduler.step() 
        metrics['lrs'].append(scheduler.get_last_lr())
        
        
    #Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
    metrics['losses'].append([tot_loss/i,loss_amp/i,loss_ph/i]) 
    

def validate_nn(validloader, metrics, device, model, criterion, scheduler, MODEL_SAVE_PATH, NGPUS, verbose):
    """
    Validate the trained network
    
    Parameters
    ----------
    vaildloader : torch.dataloader
        the pytorch data loader holding the validation data
    device : torch.device
        the pytorch device to run the model on
    model : torch.model
        the pytorch model to use
    criterion : unknown
        unknown
    scheduler : unknown
        unknown
    MODEL_SAVE_PATH : str
        the path on where to save the model
    NGPUS : int
        the amount of gpu's to use
        
    Returns
    -------
    None
    """
    
    tot_val_loss = 0.0
    val_loss_amp = 0.0
    val_loss_ph = 0.0
    for j, (ft_images,amps,phs) in enumerate(validloader):
        ft_images = ft_images.to(device)
        amps = amps.to(device)
        phs = phs.to(device)
        pred_amps, pred_phs = model(ft_images) #Forward pass
    
        val_loss_a = criterion(pred_amps,amps) 
        val_loss_p = criterion(pred_phs,phs)
        val_loss = val_loss_a + val_loss_p
    
        tot_val_loss += val_loss.detach().item()
        val_loss_amp += val_loss_a.detach().item()
        val_loss_ph += val_loss_p.detach().item()  
    metrics['val_losses'].append([tot_val_loss/j,val_loss_amp/j,val_loss_ph/j])
  
  #Update saved model if val loss is lower
    if(tot_val_loss/j<metrics['best_val_loss']):
        if verbose:
            print("Saving improved model after Val Loss improved from %.5f to %.5f" %(metrics['best_val_loss'],tot_val_loss/j))
        metrics['best_val_loss'] = tot_val_loss/j
        update_saved_model(model, MODEL_SAVE_PATH, NGPUS
                          )
        
nconv = 32

class recon_model(nn.Module):

    def __init__(self):
        super(recon_model, self).__init__()


        self.encoder = nn.Sequential( # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
          nn.Conv2d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv, nconv, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),

          nn.Conv2d(nconv*2, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),          
          nn.ReLU(),
          nn.MaxPool2d((2,2)),
          )

        self.decoder1 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Sigmoid() #Amplitude model
          )

        self.decoder2 = nn.Sequential(

          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*4, nconv*4, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*4, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),
            
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Conv2d(nconv*2, nconv*2, 3, stride=1, padding=(1,1)),
          nn.ReLU(),
          nn.Upsample(scale_factor=2, mode='bilinear'),

          nn.Conv2d(nconv*2, 1, 3, stride=1, padding=(1,1)),
          nn.Tanh() #Phase model
          )
    
    def forward(self,x):
        x1 = self.encoder(x)
        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        #Restore -pi to pi range
        ph = ph*np.pi #Using tanh activation (-1 to 1) for phase so multiply by pi

        return amp,ph
    
    
def step1_setup(data_diffr_red, amp, ph, NLINES, NLTEST, N_VALID, BATCH_SIZE, H, W, verbose):
    """
    Setting up the dataloaders
    
    Parameters
    ----------
    data_diffr_red : nd.array
        array holding the diffraction data
    amp : nd.array
        array holding the amplitude data
    ph : nd.array
        array holding the phase data
    NLINES : int
        the amount of data to use for training
    NLTEST : int
        the amount of data to use for test data
    N_VALID : int
        the amount of data to use for validation
    BATCH_SIZE: int
        the batch size of the training
    H : int
        height of the images
    W : int
        width of the images
        
    Returns
    -------
    list of nd.array
        train_dataloader, validation_dataloader, test_dataloader, N_TRAIN, X_test_data, Y_amp_data, Y_phase_data
    """
    
    tst_strt = amp.shape[0]-NLTEST #Where to index from
    if verbose:
        print(tst_strt)

    X_train = data_diffr_red[:NLINES].reshape(-1,H,W)[:,np.newaxis,:,:]
    X_test = data_diffr_red[tst_strt:].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_I_train = amp[:NLINES].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_I_test = amp[tst_strt:].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_phi_train = ph[:NLINES].reshape(-1,H,W)[:,np.newaxis,:,:]
    Y_phi_test = ph[tst_strt:].reshape(-1,H,W)[:,np.newaxis,:,:]

    ntrain = X_train.shape[0]*X_train.shape[1]
    ntest = X_test.shape[0]*X_test.shape[1]

    if verbose:
        print(X_train.shape, X_test.shape)

    X_train, Y_I_train, Y_phi_train = shuffle(X_train, Y_I_train, Y_phi_train, random_state=0)
    
    #Training data
    X_train_tensor = torch.Tensor(X_train) 
    Y_I_train_tensor = torch.Tensor(Y_I_train) 
    Y_phi_train_tensor = torch.Tensor(Y_phi_train)

    #Test data
    X_test_tensor = torch.Tensor(X_test) 
    Y_I_test_tensor = torch.Tensor(Y_I_test) 
    Y_phi_test_tensor = torch.Tensor(Y_phi_test)

    if verbose:
        print(Y_phi_train.max(), Y_phi_train.min())

        print(X_train_tensor.shape, Y_I_train_tensor.shape, Y_phi_train_tensor.shape)

    train_data = TensorDataset(X_train_tensor,Y_I_train_tensor,Y_phi_train_tensor)
    test_data = TensorDataset(X_test_tensor)
    
    N_TRAIN = X_train_tensor.shape[0]
    train_data2, valid_data = torch.utils.data.random_split(train_data,[N_TRAIN-N_VALID,N_VALID])
    if verbose:
        print(len(train_data2),len(valid_data),len(test_data))
    
    #download and load training data
    trainloader = DataLoader(train_data2, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    validloader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    #same for test
    #download and load training data
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, validloader, testloader, N_TRAIN, X_test, Y_I_test, Y_phi_test


def gpu_opti_settings(model, H, W, N_TRAIN, N_VALID, BATCH_SIZE, LR, verbose=False):
    """
    Setup/get parameters for the GPU
    
    Parameters
    ----------
    model : torch.model
        pytorch model
    H : int
        height of the images
    W : int
        width of the images
    N_TRAIN : int
        amount of lines to train
    N_VALID : int
        amount of lines for validation
    BATCH_SIZE : int
        the batch size of training
    LR : float
        the learning rate
    verbose : bool
        prints useful data
    
    Returns
    -------
    list of nd.array
        model, criterion, optimizer, scheduler, device, iterations_per_epoch
    """
    
    if verbose:
        summary(model,(1,H,W),device="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 10:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model) #Default all devices

    model = model.to(device)
    
    #Optimizer details
    iterations_per_epoch = np.floor((N_TRAIN-N_VALID)/BATCH_SIZE)+1 #Final batch will be less than batch size
    step_size = 6*iterations_per_epoch #Paper recommends 2-10 number of iterations, step_size is half cycle
    if verbose:
        print("LR step size is:", step_size, "which is every %d epochs" %(step_size/iterations_per_epoch))

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR/10, max_lr=LR, step_size_up=step_size,
                                                  cycle_momentum=False, mode='triangular2')
    
    return model, criterion, optimizer, scheduler, device, iterations_per_epoch


def stitch_predictions(data_dir, scan_num, predicted_data, pixel_div=1, H=128, W=128):
    main_path = f"{data_dir}{scan_num}/"
    scan_pos = np.load(f"{main_path}scan_pixel_positions.npy")
    scan_pos = np.asarray(scan_pos, dtype=object)
    pos = [scan_pos[:, 1], scan_pos[:, 0]]
    pos = np.asarray(pos, dtype=object)
    pos = pos / pixel_div
    
    pos_row = (pos[1]-np.min(pos[1]))
    pos_col = (pos[0]-np.min(pos[0]))

    print(pos_row.shape)

    # integer position
    pos_int_row = pos_row.astype(np.int32)
    pos_int_col = pos_col.astype(np.int32)

    pos_subpixel_row = pos_row - pos_int_row
    pos_subpixel_col = pos_col - pos_int_col
    
    preds_amp = predicted_data.copy()
    hH = int(H/2)
    hW = int(W/2)

    composite_amp = np.zeros((np.max(pos_int_row)+H,np.max(pos_int_col)+W),float)

    print(composite_amp.shape)
    ctr = np.zeros_like(composite_amp)

    data_reshaped = preds_amp.reshape(preds_amp.shape[0], H, W)
    print(data_reshaped.shape)

    for i in range(pos_row.shape[0]):
    #     data_tmp = np.real(sub_shift.subpixel_shift(data_reshaped[i]*pb_weight,pos_subpixel_row[i],pos_subpixel_col[i]))
    #     weight_tmp = np.real(sub_shift.subpixel_shift(pb_weight,pos_subpixel_row[i],pos_subpixel_col[i]))
        composite_amp[pos_int_row[i]:pos_int_row[i]+H,pos_int_col[i]:pos_int_col[i]+H] += data_reshaped[i] * 1 #* pb_weights
        ctr[pos_int_row[i]:pos_int_row[i]+W,pos_int_col[i]:pos_int_col[i]+W] += pb_weights

    composite_amp = composite_amp[hH:-hH,hW:-hW]
    ctr = ctr[hH:-hH,hW:-hW]
    
    return composite_amp
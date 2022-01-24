import sys, os
from network_utils import *
from tqdm import tqdm
import torch
import datetime


def train_ptychonn(data_path, start_scan, end_scan,
                   train_lines=0.85, val_lines=0.15, epochs=60, gpu_select='1', batch_size=64,
                   load_model_scan=-1, save_model_scan=-1, ngpus=1, lr=1e-3, height=128, width=128,
                   mean_phsqr=0.02, verbose=False):
    
    """
    Train PtychoNN network on the selected data
    
    Parameters
    ----------
    data_path : str
        the path to the folder which houses the saved tike data
    
    start_scan : int
        the staring scan to import tike data from
    
    end_scan : int
        the ending scan to import tike data from
    
    train_lines : float
        the percentage of data to use during training
        
    val_lines : float
        the percentage of data to use during validation
    
    epochs : int
        the number of epochs to train
        
    gpu_select : str
        the string ID of the gpu the user would like to train with
    
    batch_size : int
        the batch size during training
        
    load_model_scan : int
        the scan number to pull saved model from. Value of -1 loads blank network
        
    save_model_scan : int
        the scan number to save the model to. Value of -1 saves to end_scan.
    
    ngpus : int
        the amount of gpu's to train with (IN BETA)
        
    height : int
        the x dim to resize to for training
    
    width : int
        the y dim to resize to for training
    
    mean_phsqr : float
        all training images with a mean squared phase below this value will not be used during training.
        this avoids blank FOV's
    
    verbose : bool
        if True it will output useful metrics.
    
    Returns
    -------
    None
        saves trained model and training metrics to User defined scan number
    """
    t1 = datetime.datetime.now()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_select

    im_shape = (height, width)

    EPOCHS, NGPUS, BATCH_SIZE, LR = setup_training_params(epochs, ngpus, batch_size, lr, verbose)

    if save_model_scan == -1:
        save_model_scan = end_scan

    MODEL_SAVE_PATH = setup_model_path(data_path, save_model_scan)

    data_diffr_red, amp, ph = load_dataV3(data_path, start_scan, end_scan, im_shape=im_shape, mean_phsqr_val=mean_phsqr)


    total_lines = len(data_diffr_red)

    H,W = height, width
    NLINES = int(total_lines * train_lines)
    NLTEST = int(total_lines * (1-train_lines))
    N_VALID = int(total_lines * val_lines)

    trainloader, validloader, testloader, N_TRAIN, X_test, Y_I_test, Y_phi_test = step1_setup(data_diffr_red,
                                                                                              amp, ph,
                                                                                              NLINES,
                                                                                              NLTEST,
                                                                                              N_VALID,
                                                                                              BATCH_SIZE,
                                                                                              H, W, verbose)

    model = recon_model()

    if load_model_scan != -1:
        model = torch.load(f'{data_path}{load_model_scan}/trainded_model/best_model.pth')

    model, criterion, optimizer, scheduler, device, iterations_per_epoch = gpu_opti_settings(model,
                                                                                             H, W,
                                                                                             N_TRAIN,
                                                                                             N_VALID,
                                                                                             BATCH_SIZE,
                                                                                             LR)
    
    t2 = datetime.datetime.now()
    metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
    for epoch in tqdm(range(EPOCHS), position=0, leave=True, desc='Epoch'):

        #Set model to train mode
        model.train() 

        #Training loop
        train_nn(trainloader, metrics, device, model, criterion, optimizer, scheduler)

        #Switch model to eval mode
        model.eval()

        #Validation loop
        validate_nn(validloader, metrics, device, model, criterion, scheduler, MODEL_SAVE_PATH, NGPUS, verbose)

        if verbose:
            print('Epoch: %d | FT  | Train Loss: %.5f | Val Loss: %.5f' %(epoch, metrics['losses'][-1][0], metrics['val_losses'][-1][0]))
            print('Epoch: %d | Amp | Train Loss: %.4f | Val Loss: %.4f' %(epoch, metrics['losses'][-1][1], metrics['val_losses'][-1][1]))
            print('Epoch: %d | Ph  | Train Loss: %.3f | Val Loss: %.3f' %(epoch, metrics['losses'][-1][2], metrics['val_losses'][-1][2]))
            print('Epoch: %d | Ending LR: %.6f ' %(epoch, metrics['lrs'][-1][0]))

    t3 = datetime.datetime.now()
    np.savez(f'{data_path}{save_model_scan}/trainded_model/training_metrics.npz',
             losses=metrics['losses'], val_losses=metrics['val_losses'],
             lrs=metrics['lrs'], best_val_loss=metrics['best_val_loss'],
             total_load_time=t2-t1, total_train_time=t3-t2)
    
    
def predict_ptychonn(data_path, pred_scan, network_scan, ngpus=1, gpu_select='1', batch_size=64, height=128, width=128, verbose=False):
    """
    Predict a ptycho recon using a trained ptychoNN network.
    
    Parameters
    ----------
    data_path : str
        the path to the folder which houses the saved tike data
        
    pred_scan : int
        the scan number to predict
    
    network_scan : int
        the scan number to load a trained model from
        
    ngpus : int
        the number of gpus to train with (IN BETA)
        
    gpu_select : str
        the gpu ID's to train with
    
    batch_size : int
        the batch size to use during prediction
    
    height : int
        the x dim to resize to for training
    
    width : int
        the y dim to resize to for training
        
    verbose : bool
        if True it will output useful metrics. 
    
    Returns
    -------
    None
        saves the predicted amplitude and phase to the data_path/pred_scan/prediction_amp.npy or
        data_path/pred_scan/prediction_phi.npy
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_select
    im_shape = (height, width)
    data_diffr_red, amp, ph = load_dataV3(data_path, pred_scan, pred_scan, im_shape=im_shape, torp='predict')
    X_pred = data_diffr_red.reshape(-1, height, width)[:,np.newaxis,:,:]
    X_pred_tensor = torch.Tensor(X_pred) 
    pred_data = TensorDataset(X_pred_tensor)
    
    predloader = DataLoader(pred_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
    MODEL_SAVE_PATH = setup_model_path(data_path, network_scan)
    model = torch.load(MODEL_SAVE_PATH + 'best_model.pth')
    model, criterion, optimizer, scheduler, device, iterations_per_epoch = gpu_opti_settings(model,
                                                                                             height, width,
                                                                                             1,
                                                                                             1,
                                                                                             batch_size,
                                                                                             1e-3)
    
    model.eval()
    amps = []
    phs = []
    for i, ft_images in enumerate(predloader):
        ft_images = ft_images[0].to(device)
        amp, ph = model(ft_images)
        for j in range(ft_images.shape[0]):
            amps.append(amp[j].detach().to("cpu").numpy())
            phs.append(ph[j].detach().to("cpu").numpy())

    amps = np.array(amps).squeeze()
    phs = np.array(phs).squeeze()
    
    org_shape = np.load(f"{data_path}{network_scan}/scan_shape_metadata.npy")
    pidxel_div = org_shape[-1]/width
    
    amp = stitch_predictions(data_path, pred_scan, amps, pixel_div=pidxel_div, H=height, W=width)
    ph = stitch_predictions(data_path, pred_scan, phs, pixel_div=pidxel_div, H=height, W=width)
    
    np.save(f'{data_path}{pred_scan}/prediction_amp.npy', amp)
    np.save(f'{data_path}{pred_scan}/prediction_phi.npy', ph)
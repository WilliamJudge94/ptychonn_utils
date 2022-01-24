import sys, os
#sys.path.append('/projects/hp-ptycho/wjudge/ptychonn/ptychonn/')
from network_utils import *
from tqdm import tqdm
import torch
import typer

import warnings
warnings.filterwarnings("ignore")

app = typer.Typer()

# '/projects/hp-ptycho/wjudge/ptychonn/scan_dataml/'

@app.command()
def train(data_path: str = typer.Argument(..., help="path to the folder containing folders of scans"),
          start_scan: int = typer.Argument(..., help="start scan number"),
          end_scan: int = typer.Argument(..., help="end scan number"),
          
          train_lines: float = typer.Option(0.85, help="percent of lines to train with"),
          val_lines: float = typer.Option(0.15, help="percent of training data to validate with"),
          
          epochs: int = typer.Option(60, help="number of epochs to train"),
          gpu_select: str = typer.Option('1', help="select which gpu to use"),
          batch_size: int = typer.Option(64, help="batch size during training"),
          load_model_scan: int = typer.Option(-1, help="dir where to load model from - default load blank network"),
          save_model_scan: int = typer.Option(-1, help="dir where to save model to - default save to end_scan"),
          
          ngpus: int = typer.Option(1, help="number of gpu's to use"),
          lr: float = typer.Option(1e-3, help="learning rate"),
          height: int = typer.Option(128, help="height of image"),
          width: int = typer.Option(128, help="width of image"),
          mean_phsqr: float = typer.Option(0.02, help="sqrt of phase to allow during training"),
          verbose: bool = typer.Option(False, help="print useful data")):
    
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
    
    metrics = {'losses':[],'val_losses':[], 'lrs':[], 'best_val_loss' : np.inf}
    for epoch in tqdm(range(EPOCHS), position=0, desc='Epoch'):

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
    
    np.savez(f'{data_path}{save_model_scan}/trainded_model/training_metrics.npz',
             losses=metrics['losses'], val_losses=metrics['val_losses'],
             lrs=metrics['lrs'], best_val_loss=metrics['best_val_loss'])

@app.command()
def predict(data_path: str = typer.Argument(..., help="path to the folder containing folders of scans"),
          pred_scan: int = typer.Argument(..., help="pred scan number"),
          network_scan: int = typer.Argument(..., help="scan to load trained network from"),
          ngpus: int = typer.Option(1, help="number of gpu's to use"),
          gpu_select: str = typer.Option('1', help="select which gpu to use"),
          batch_size: int = typer.Option(64, help="batch size during training"),
          height: int = typer.Option(128, help="height of image"),
          width: int = typer.Option(128, help="width of image"),
          verbose: bool = typer.Option(False, help="print useful data")):
    
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
    
        
if __name__ == "__main__":
    app()
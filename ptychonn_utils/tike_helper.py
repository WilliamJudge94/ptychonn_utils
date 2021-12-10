def save_ptychonn_scan_data(location, tike_result, diffr_data, probe_size=256):
    patches = extract_patches(tike_result['psi'], tike_result['scan'], probe_size)
    
    phase = np.angle(tike_result['psi'])
    ampli = np.abs(tike_result['psi'])
    phase = np.asarray(phase)
    ampli = np.asarray(ampli)
    
    np.save(f'{location}/recon_data.npy', patches)
    np.save(f'{location}/diff_data.npy',data)
    np.save(f'{location}/phase.npy', phase)
    np.save(f'{location}/ampli.npy', ampli)
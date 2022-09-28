# This module contains the classes/functions associated with applying 
# a trained model to unknown LIBS spectra
import pickle
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares

import torch
from libs_unet.models import peakyfinder_0001
from libs_unet.training.spec_maker import spectrum_maker

top_dir = Path(__file__).parent.parent.parent
rel_path = 'data' 
datapath = top_dir / rel_path

#Model specific parameters/files here. Update as models change
#Reference data, used but not changed within classes
with open(datapath / 'training' / 'el77_meta.pickle', 'rb') as f:
    wave = pickle.load(f)
    el_symbol = pickle.load(f)
    el_index = pickle.load(f)

max_z = len(el_symbol)
model = peakyfinder_0001.LIBSUNet(max_z,len(wave))
param_path = top_dir / 'trained_models' / 'el77_pairs_0001'
model.load_state_dict(torch.load(param_path))
rel_int_scale = 10**4
input_scale = 5
thresh = 7
fit_tol = 0.01
#Generate atomic reference spectra
spec_maker = spectrum_maker()
el_spec = np.zeros((max_z,760))
for i in range(max_z):
    fracs_dict = {el_symbol[i]:1}
    wave, el_spec[i], spec_dict = spec_maker.make_spectra(fracs_dict)


def spec_resid(x, x_spec, ref_specs):
    return np.squeeze(np.sum((ref_specs.transpose() * x).transpose(), axis=0) - x_spec)

def fit_spec(x_spec, exp_wave=wave):
    if not np.array_equal(exp_wave, wave):
        raise ValueError("Experimental wavelength domain does not match model")
    if type(x_spec) != np.ndarray:
        raise ValueError("Spectrum data type must be numpy.ndarray")
    if len(x_spec) != len(wave):
        raise ValueError("Spectrum array length does not match model")
    
    #Transform the input spectrum to model domain
    x_spec_tensor = torch.tensor(x_spec.astype('float32'))[None,None,:]
    x_spec_trans = input_scale * torch.log(rel_int_scale * x_spec_tensor + 1)
    #Run through model and obtain predicted elements
    model.eval()
    with torch.no_grad():
        y_pred = model(x_spec_trans).detach().numpy()
    y_pred = np.squeeze(y_pred) #(max_z,760)
    el_pred = 1*(np.max(y_pred[0:max_z,:], axis=1) > thresh)

    #Obtain element weights from least squares fit to input spectrum
    #define bounds based on elements identified by prediction
    bnd_low = np.zeros(max_z)
    #upper bound on candidates is 1, for which we can use el_pred mask
    #need a small delta to avoid lb = ub
    bnd_up = el_pred + 0.001
    el_bounds = (bnd_low, bnd_up)
    #initial weights guess, start with balanced allocation to candidates
    x0 = el_pred / np.sum(el_pred)
    #obtain fit object
    test_fit = least_squares(spec_resid, x0, args=(x_spec, el_spec), bounds=el_bounds)
    #weights below tolerance discarded and total weights rescaled to 1
    fit_wts = test_fit.x.copy()
    fit_wts[fit_wts < fit_tol] = 0
    fit_wts = fit_wts / np.sum(fit_wts)
    #build dictionary to return with element weights
    el_weights = {}
    for i in range(len(fit_wts)):
        if fit_wts[i] > 0:
            el_weights[el_symbol[i]] = np.round(fit_wts[i],4)

    return el_weights
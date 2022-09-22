import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.special import voigt_profile as voigt
from pathlib import Path

top_dir = Path(__file__).parent.parent.parent
rel_path = 'data' 
datapath = top_dir / rel_path
datafile = "rel_int/valid77_spec.pickle" #this is specific to the avail_elem for the class
#create element list for this data file
with open(datapath / datafile, 'rb') as f:
    atom_dict = pickle.load(f)
avail_elem = [key for key in atom_dict.keys()]

class spectrum_maker():
    #class docstring, parameters, public methods
    """ generates LIBS spectra """

    def __init__(self) -> None:
        super().__init__()

    def peak_maker(self,
    element,
    inc=1,
    w_lo=190,
    w_hi=950,
    voigt_sig=1,#stdev of normal part of voigt convolution
    voigt_gam=1, #half-width at half max parameter of cauchy part of convolution
    shift=False,
    shift_type='random',
    shift_mean=5,
    height=False,
    height_type='random',
    height_mean=0,
    height_mag=0.001,
    plot=False):
        
        peak_loc = atom_dict[element][:,0]
        rel_int = atom_dict[element][:,1]
        if plot:
        #    plot histogram of element intensities
            plt.bar(x=peak_loc, height=rel_int, width=3,color="red")
            plt.xlabel('wavelength [nm]')
            plt.ylabel('intensity')
            plt.xlim([190, 950]) #note data may go beyond this range
            plt.show
        
        peak_count = len(rel_int)
        wave = np.arange(w_lo, w_hi, inc)
        
        # jitter peak positions and intensities
        if shift:
            if shift_type=='sys': # apply systematic peak shift
                peak_loc = peak_loc + shift_mean
            if shift=='random': # apply random peak wavelength shift, mean 0
                mag = shift_mean * (np.random.rand(peak_count) - 0.5)
                peak_loc = peak_loc + mag
        
        if height:
            if height_type=='random':
                h_mult = np.random.rand(peak_count) + 0.5 #min 0.5, mean 1.0, max 1.5
                rel_int = rel_int * h_mult
                rel_int = rel_int / np.sum(rel_int, axis=0) #re-scale to 1.0
            if height_type=='lin':
                h_add = height_mag * peak_loc + height_mean
                rel_int = np.where(rel_int + h_add < 0, 0, rel_int + h_add)
                rel_int = rel_int / np.sum(rel_int, axis=0) #re-scale to 1.0
        
        # create peaks with defined Voigt profiles from peak location and intensities derived from database
        peaks = np.array([a * voigt(wave - x, voigt_sig, voigt_gam) for a, x in zip(rel_int, peak_loc)])
        #sum the wave profiles across all the (rel_int, peak_loc) tuples, now smoothed spectra  on range w_lo:w_hi
        #scale the end result
        spec = np.sum(peaks, axis=0)
        spec = spec/np.sum(spec)
        
        if plot:
            plt.plot(wave, spec)
            plt.xlabel('wavelength [nm]')
            plt.ylabel('intensity')
            plt.xlim([190, 950])
            plt.show
        
        return wave, spec    
    
    #make_spectra provides the weighted superposition of peak_maker spectra with artifacts/noise added
    def make_spectra(self, 
        fracs_dict, #dict of element of positive fractions/proportions (will normalize to sum 1)
        inc=1,
        w_lo=190, # lower limit of spectrum
        w_hi=950, # upper limit of spectrum
        artifact=False, # flag to include spectral artifacts ('constant', 'square', or 'Gaussian')
        art_type=['square', 'Gaussian'], # types of artifacts to be included - must be a list for now
        art_mag=0.1, # relative magnitude of artifact to spectrum intensity
        noise=False, # noise flag
        noise_type='Gaussian', # noise type
        snr=10):
        
        wave = np.arange(w_lo, w_hi, inc)
        frac_total = 0
        for k, v in fracs_dict.items():
            frac_total += v
            if k not in atom_dict.keys():
                raise ValueError(f"Unsupported element {k}")
            if v < 0:
                raise ValueError("Element fractions must be non-negative")
        if frac_total <= 0:
            raise ValueError("Positive element fractions required")
        
        #scale fractions to sum to 1.0
        for k, v in fracs_dict.items():
            fracs_dict[k] = v / frac_total


        #gen individual element spectra and combine into weighted sum. (weighted sum should remain 1.0)
        spec = np.zeros(len(wave)) #composite of weighted atomic spectra
        wave = np.arange(w_lo, w_hi, inc)
        spec_dict = {k:np.zeros(len(wave)) for k in fracs_dict.keys()}
        for elem, frac in fracs_dict.items():
            if frac > 0:#only process elements with non-zero weight
        # TODO use **kwargs to pass on parameters from this method invocation to next
                _, spec_dict[elem] = self.peak_maker(elem)
                spec_dict[elem] = fracs_dict[elem] * spec_dict[elem]
                spec += spec_dict[elem]
        
        #rescale
        spec /= np.sum(spec)
        maximum = np.max(spec)
        
        # --- add artifacts
        spec_dict['art'] = np.zeros(len(wave))
        if artifact:
            if any([i=='const' for i in art_type]):
                spec_dict['art'] += art_mag * maximum
                
            if any([i=='square' for i in art_type]):
                lim = np.sort(np.random.choice(wave, 2))
                idx = (wave>lim[0]) * (wave<lim[1])
                sq_loc = np.where(idx)[0]
                art_scale = art_mag * maximum
                spec_dict['art'][sq_loc] += art_scale
                
            if any([i=='Gaussian' for i in art_type]):
                #TODO check if sigma should be parametrized with method arg
                sigma = (w_hi-w_lo)*0.5
                mu = np.random.randint(w_lo,w_hi)
                bg = 100 * np.random.rand() * maximum * 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (wave - mu)**2 / (2 * sigma**2))
                spec_dict['art'] += bg
                
        spec += spec_dict['art']
        # --- add noise
        spec_dict['noi'] = np.zeros(len(wave))
        if noise:
            if noise_type=='Gaussian':
                spec_dict['noi'] += np.random.normal(0, 1/snr**0.5, len(wave))
        
        #limit the net spectrum to nonnegative intensity values
        spec = np.where(spec + spec_dict['noi'] < 0, 0, spec + spec_dict['noi'])
        spec /= np.sum(spec)

        return wave, spec, spec_dict

        
    #TODO
    """ def batch_spectra(self,
        focus_el=[], #optional list of specific elements within class max_z
        n_elem=4, #defines the mean number of elements included
        n_delta=2, #defines the +/- range for number of elements to vary
        abund_scale=0.5, #max variation factor on natural abundance (<=1)
        inc=1,
        w_lo=190, # lower limit of spectrum
        w_hi=950, # upper limit of spectrum
        artifact=False, # flag to include spectral artifacts ('constant', 'square', or 'Gaussian')
        art_type=['square', 'Gaussian'], # types of artifacts to be included - must be a list for now
        art_mag=0.1, # relative magnitude of artifact to spectrum intensity
        noise=False, # noise flag
        noise_type='Gaussian', # noise type
        snr=10,
        batch=16): #number of samples to create
        
        max_elem = self.max_z
        if len(focus_el):
            max_elem = len(focus_el)
            if not all (x in self.elements for x in focus_el):
                raise ValueError(f"Elements must be among the first {self.max_z}.")
        if n_elem + n_delta > max_elem:
            raise ValueError("n_elem + n_delta cannot exceed available elements") 
        if n_delta > n_elem-1:
            raise ValueError("n_delta must be less than n_elem to avoid empty samples")
        if abund_scale <0 or abund_scale >1:
            raise ValueError(f"abund_scale must lie on interval [0,1], {abund_scale} given")
        #generate the element fractions
        #first identify the number of elements that will be in each sample in the batch
        num_elem = (n_elem + np.round(2 * (n_delta+0.5) * np.random.rand(batch) - (n_delta+0.5))).astype(int)
        #next identify the elements to include
        #TODO add option for non-uniform sampling, e.g. sample appearance ~ abundance or other factor
        if len(focus_el):
            sample_el= [np.random.choice(focus_el, num_elem[i]) for i in range(batch)] #list, not array
        else:
            sample_el = [np.random.choice(self.elements, num_elem[i]) for i in range(batch)] #list, not array
        
        #with the elements selected, generate fracs arrays for make_spectra
        #note the length of fracs is determined by max_z on object instantiation
        #central proportion are based on abundance (entropic), maybe another mode for enthalpic
        #create a boolean mask to filter element abundance
        samp_mask = np.array([np.in1d(self.elements, sample_el[i]) for i in range(batch)]) #shape (batch, max_z)
        sample_abund = self.elem_abund * samp_mask #rightmost dims = max_z for broadcasting
        #with relative sample proportions set to abundance, introduce variation scaled by parameter
        sample_var = 2 * abund_scale * (np.random.rand(batch, self.max_z)-0.5)
        sample_fracs = sample_abund * (1 + sample_var)
        #scale to sum to 1.0, maybe redundant with later scaling
        fracs = sample_fracs / np.sum(sample_fracs, axis=1, keepdims=True) 

        #The goal of UNet model is prediction of spectra array from composite spectra
        wave = np.arange(w_lo,w_hi,inc) #only needed for correct length
        x_data = np.zeros((batch, len(wave)))
        y_data = np.zeros((batch, int(self.max_z+2), len(wave)))
        
        #note the wave range generated from w_lo, w_hi, inc is going to be same in each sample
        # make_spectra also returns composite spectra and array of weighted element spectra, artifacts, noise
        for i in np.arange(batch):
            wave, x_data[i], y_data[i] = self.make_spectra(fracs=fracs[i], inc=inc, w_lo=w_lo, w_hi=w_hi, 
                                                        artifact=artifact, art_type=art_type, art_mag=art_mag,
                                                        noise=noise, noise_type=noise_type,snr=snr)
        
        return fracs, wave, x_data, y_data """
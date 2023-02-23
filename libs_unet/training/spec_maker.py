import numpy as np
import pickle
from scipy.special import voigt_profile as voigt
from pathlib import Path

top_dir = Path(__file__).parent.parent.parent
rel_path = 'data' 
datapath = top_dir / 'data' / 'nist_libs'
lines_file = "nist_libs_hi_res.pickle"
#create element list for this data file
with open(datapath / lines_file, 'rb') as f:
    wave = pickle.load(f)
    atom_lines = pickle.load(f)
avail_elem = [key for key in atom_lines.keys()]
el_index = {} #lookup from el symb to array index of element
ind = 0
for el in avail_elem:
    el_index[el] = ind
    ind += 1
w_lo = np.min(wave)
w_hi = np.max(wave)

class spectrum_maker():
    #class docstring, parameters, public methods
    """ generates LIBS spectra """

    def __init__(self) -> None:
        super().__init__()
        self.avail_elem = avail_elem
        self.max_z = len(avail_elem)
        self.el_index = el_index

#peak_maker receives an array of line locations/intensities and creates peaks with optional perturbations
    def peak_maker(self,
    line_loc,
    lines,
    voigt_sig=1,#stdev of normal part of voigt convolution
    voigt_gam=1, #half-width at half max parameter of cauchy part of convolution
    shift=False,
    shift_type='random',
    shift_mean=5,
    height=False,
    height_type='random',
    height_mean=0,
    height_mag=0.001,
    ):
        
        # jitter peak positions and intensities
        if shift:
            if shift_type=='sys': # apply systematic peak shift
                line_loc = line_loc + shift_mean
            if shift=='random': # apply random peak wavelength shift, mean 0
                mag = shift_mean * (np.random.rand(len(line_loc)) - 0.5)
                line_loc = line_loc + mag
        
        if height:
            if height_type=='random':
                h_mult = np.random.rand(len(line_loc)) + 0.5 #min 0.5, mean 1.0, max 1.5
                lines = lines * h_mult
            if height_type=='lin':
                h_add = height_mag * lines + height_mean
                lines = lines + h_add
        
        lines[lines < 0] = 0 #no negative intensities 
        
        # create peaks with defined Voigt profiles from peak location and intensities 
        peaks = np.array([a * voigt(wave - x, voigt_sig, voigt_gam) for a, x in zip(lines, line_loc)])
        #sum the wave profiles across all the (intensity, loc) tuples, now smoothed spectra  on range w_lo:w_hi
        #scale the end result
        spec = np.sum(peaks, axis=0)

        return wave, spec    
    
    #make_spectra calculates the weighted combination of atomic lines and returns the composite spectrum
    def make_spectra(self, 
        fracs_dict, #dict of element of positive fractions/proportions (will normalize to sum 1)
        rescale = True, #if true fractions rescaled to sum=1
        artifact=False, # flag to include spectral artifacts ('constant', 'square', or 'Gaussian')
        art_type=['square', 'Gaussian'], # types of artifacts to be included - must be a list for now
        art_mag=0.1, # relative magnitude of artifact to spectrum intensity
        noise=False, # noise flag
        noise_type='Gaussian', # noise type
        snr=10,
        comp_only=False):
        
        frac_total = 0
        for k, v in fracs_dict.items():
            frac_total += v
            if k not in avail_elem:
                raise ValueError(f"Unsupported element {k}")
            if v < 0:
                raise ValueError("Element fractions must be non-negative")
        if frac_total <= 0:
            raise ValueError("Positive element fractions required")
        
        #scale fractions to sum to 1.0
        if rescale == True:
            for k, v in fracs_dict.items():
                fracs_dict[k] = v / frac_total

        #create dictionary of weighted atomic lines to combine
        lines_dict = { el : frac * atom_lines[el] for el,frac in fracs_dict.items() }
        spec_dict = {} #store weighted line spectrum for each element to return
        comp_lines = np.zeros(len(wave))
        for el in lines_dict.keys():
            comp_lines += lines_dict[el] #composite lines array to make spectra
            #generate the weighted line spectrum for this element only
            if comp_only == False:
                _, spec_dict[el] = self.peak_maker(wave, lines_dict[el])
        
        lines_dict['comp'] = comp_lines
        
        
        #create composite spectrum
        _, spec = self.peak_maker(wave, lines_dict['comp'])
        
        maximum = np.max(spec)
        
        # --- add artifacts
        art_mod = np.zeros(len(wave))
        if artifact:
            if any([i=='const' for i in art_type]):
                art_mod += art_mag * maximum
                
            if any([i=='square' for i in art_type]):
                lim = np.sort(np.random.choice(wave, 2))
                idx = (wave>lim[0]) * (wave<lim[1])
                sq_loc = np.where(idx)[0]
                art_scale = art_mag * maximum
                art_mod[sq_loc] += art_scale
                
            if any([i=='Gaussian' for i in art_type]):
                #TODO check if sigma should be parametrized with method arg
                sigma = (w_hi-w_lo)*0.5
                mu = np.random.randint(w_lo,w_hi)
                bg = 100 * np.random.rand() * maximum * 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (wave - mu)**2 / (2 * sigma**2))
                art_mod += bg
        
        spec_dict['art'] = art_mod        
        spec += art_mod
        
        # --- add noise
        noise_mod = np.zeros(len(wave))
        if noise:
            if noise_type=='Gaussian':
                noise_mod += np.random.normal(0, 1/snr**0.5, len(wave))
        
        #limit the net spectrum to nonnegative intensity values
        spec_pre = np.ndarray.copy(spec) #store initial to back out net noise modification
        spec_dict['comp'] = np.where(spec + noise_mod < 0, 0, spec + noise_mod)
        spec_dict['noi'] = spec_dict['comp'] - spec_pre
        return wave, spec_dict, lines_dict

        
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
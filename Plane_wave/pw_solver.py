import numpy as np

from examples.seismic.acoustic.wavesolver import AcousticWaveSolver
from examples.seismic.elastic.wavesolver import ElasticWaveSolver

class PlaneWaveSolver(AcousticWaveSolver):
    def __init__(self, *args, p=[0.], taper=True, trim_t=False, **kwargs): #TODO wavelet='ricker',
        super().__init__(*args, **kwargs)
        self.rayp = p # ray parameter in s/km. Can be a list or an integer
        self.taper = taper
        self.trim_t = trim_t

    def forward_all(self, *args, p=None, **kwargs):#exp=0.5
        """Forward modelling of plane-waves for an array of p values.

        Args:
            p (list/1d array, optional): p values . Defaults to None.

        Returns:
            list/3d array: plane-wave gathers with dimension [Np, Nt, Nrec]
        """
        rayp = p or self.rayp
        shots = []
        # Loop over ray-parameter
        for p in rayp: #TODO more pythonic
            # Cases for positive and negative ray parameter
            if p >= 0.:
                x0 = self.geometry.src_positions[0,0]
            else:
                x0 = self.geometry.src_positions[-1,0]
            
            # Populate src data with time delays
            src = self.geometry.src
            amplitude = self.cosine_taper(exp=0.5)
            for i in range(self.geometry.nsrc):
                delta_t = p*(self.geometry.src_positions[i,0] - x0)
                src.data[:, i] = ricker_wavelet(self.geometry.time_axis.time_values, self.geometry.f0, 1/self.geometry.f0 + delta_t, a=amplitude[i]) 
            
            # Forward modeling and save shot gathers
            d, _, _ = super().forward(*args, src=src, **kwargs)

            if self.trim_t:
                shot, _, _ = trim_time(d.data, self.geometry, self.model, p, x0)            
            else:
                shot = d.data

            shots.append(shot)

        return shots 
    
    def cosine_taper(self, exp=0.5):
        
        amplitude = np.ones(self.geometry.nsrc, dtype='float32')
        
        if self.taper:
            for i in range(self.model.nbl):
                amplitude[i] = np.cos(((i-self.model.nbl)/self.model.nbl)*0.5*np.pi)**exp
            for i in range(self.geometry.nsrc-self.model.nbl, self.geometry.nsrc):
                amplitude[i] = np.cos(((i-(self.geometry.nsrc-self.model.nbl-1))/self.model.nbl)*0.5*np.pi)**exp    
        
        return amplitude 


class PlaneWaveSolver:
    def __init__(self, model, geometry, solver="acoustic", p=[0.], taper=True, trim_t=False, **kwargs): #TODO wavelet='ricker',
        self.model = model
        self.geometry = geometry
        if solver == "acoustic":
            self.solver = AcousticWaveSolver(model, geometry, **kwargs)
        elif solver == "elastic":
            self.solver = ElasticWaveSolver(model, geometry, **kwargs)
        self.rayp = p # ray parameter in s/km. Can be a list or an integer
        self.taper = taper
        self.trim_t = trim_t
    
    def forward_all(self, *args, p=None, **kwargs):#exp=0.5
        """Forward modelling of plane-waves for an array of p values.
        
        Args:
            p (list/1d array, optional): p values . Defaults to None.
        
        Returns:
            list/3d array: plane-wave gathers with dimension [Np, Nt, Nrec]
        """
        rayp = p or self.rayp
        shots = []
        # Loop over ray-parameter
        for p in rayp: #TODO more pythonic
            # Cases for positive and negative ray parameter
            if p >= 0.:
                x0 = self.geometry.src_positions[0,0]
            else:
                x0 = self.geometry.src_positions[-1,0]
            
            # Populate src data with time delays
            src = self.geometry.src
            amplitude = self.cosine_taper(exp=0.5)
            for i in range(self.geometry.nsrc):
                delta_t = p*(self.geometry.src_positions[i,0] - x0)
                src.data[:, i] = ricker_wavelet(self.geometry.time_axis.time_values, self.geometry.f0, 1./self.geometry.f0 + delta_t, a=amplitude[i]) 
            
            # Forward modeling and save shot gathers
            d = self.solver.forward(*args, src=src, **kwargs)[0]

            if self.trim_t:
                shot, _, _ = trim_time(d.data, self.geometry, self.model, p, x0)            
            else:
                shot = d.data

            shots.append(shot)

        return shots 
    
    def cosine_taper(self, exp=0.5):
        
        amplitude = np.ones(self.geometry.nsrc, dtype='float32')
        
        if self.taper:
            for i in range(self.model.nbl):
                amplitude[i] = np.cos(((i-self.model.nbl)/self.model.nbl)*0.5*np.pi)**exp
            for i in range(self.geometry.nsrc-self.model.nbl, self.geometry.nsrc):
                amplitude[i] = np.cos(((i-(self.geometry.nsrc-self.model.nbl-1))/self.model.nbl)*0.5*np.pi)**exp    
        
        return amplitude 


def ricker_wavelet(time_values, f0, t0, a=1.):
    r = (np.pi * f0 * (time_values - t0))
    return a * (1.-2.*r**2)*np.exp(-r**2.)


def find_nearest(time_axis, value):
    idx = (np.abs(time_axis-value)).argmin()
    return time_axis[idx], idx


def trim_time(shot, geometry, model, p, x0):
    if p >= 0.:
        init_src_coord = np.where(geometry.src_positions==0.)[0]
    else:
        init_src_coord = np.where(geometry.src_positions==model.domain_size[0])[0]
    
    init_delta_t = p*(geometry.src_positions[init_src_coord,0] - x0) - 1./geometry.f0

    init_t, idx = find_nearest(geometry.time_axis.time_values, init_delta_t)

    trim_shot = np.zeros(shot, dtype='float32')
    trim_shot[:-idx, :] = shot[idx:, :]

    return trim_shot, init_t, idx

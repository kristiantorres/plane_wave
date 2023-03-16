import numpy as np

from examples.seismic.acoustic.wavesolver import AcousticWaveSolver

class PlaneWaveSolver(AcousticWaveSolver):
    def __init__(self, *args, p=[0.], taper=True,  **kwargs):
        super().__init__(*args, **kwargs)
        self.rayp = p # ray parameter in s/km. Can be a list or an integer
        self.taper = taper

    def forward(self, *args, p=None, **kwargs):#exp=0.5
        rayp = p or self.rayp
        shots = []
        # Loop over ray-parameter
        for p in rayp: #TODO more pythonic
            # Cases for positive and negative ray parameter
            if p>=0.:
                x0 = self.geometry.src_positions[0,0]
            else:
                x0 = self.geometry.src_positions[-1,0]
            
            # Populate src data with time delays
            src = self.geometry.src
            amplitude = self.cosine_taper(exp=0.5)
            for i in range(self.geometry.nsrc):
                delta_t = p*(self.geometry.src_positions[i,0] - x0)
                src.data[:, i] = self.ricker_wavelet(self.geometry.time_axis.time_values, self.geometry.f0, 1/self.geometry.f0 + delta_t, a=amplitude[i]) 

            # Forward modeling and save shot gathers
            d = super().forward(*args, src=src, **kwargs)[0]
            shots.append(d)

        return shots 
    
    def cosine_taper(self, exp=0.5):
        
        amplitude = np.ones(self.geometry.nsrc, dtype='float32')
        
        if self.taper:
            for i in range(self.model.nbl):
                amplitude[i] = np.cos(((i-self.model.nbl)/self.model.nbl)*0.5*np.pi)**exp
            for i in range(self.geometry.nsrc-self.model.nbl, self.geometry.nsrc):
                amplitude[i] = np.cos(((i-(self.geometry.nsrc-self.model.nbl-1))/self.model.nbl)*0.5*np.pi)**exp    
        
        return amplitude 

    @staticmethod
    def ricker_wavelet(time_values, f0, t0, a=1):
        r = (np.pi * f0 * (time_values - t0))
        return a * (1-2.*r**2)*np.exp(-r**2)

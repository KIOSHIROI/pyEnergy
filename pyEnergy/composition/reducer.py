import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter

def reduction(reducer):
    reducers = {
        'savgol': Savgol,
        'gaussian': GaussianSmoothing,
        'conv': Conv,
        'moving average': MovingAverage,
        'my': MyReducer,
        'my2': MyReducer2
    }
    for k, v in reducers.items():
        if reducer == k:
            return v
    raise ValueError("Invalid reducer name or missing parameters.")
    
class Reducer:
    def __init__(self, **params):
        self.threshold = params.get("threshold", 3)

    def use(self, reducer):
        self.reducer = reducer

    def __call__(self, signal, **params):
        return self.reduce(signal, **params)

    def reduce(self, signal, **params):
        smoothed_signal = self.reducer(signal, **params)
        return np.where(smoothed_signal > self.threshold, smoothed_signal, 0), smoothed_signal

class Savgol(Reducer):
    def __init__(self, **params):
        super().__init__(**params)
        self.use(savgol_filter)

    def reduce(self, signal, **params):
        window_length=params.get("window_length", 5)
        polyorder = params.get("polyorder", 3)
        return super().reduce(signal, window_length=window_length, polyorder=polyorder)
    
class GaussianSmoothing(Reducer):
    def __init__(self, **params):
        super().__init__(**params)
        self.use(gaussian_filter)

    def reduce(self, signal, **params):
        sigma = params.get("sigma", 0.5)
        return super().reduce(signal, sigma=sigma)

class Conv(Reducer):
    def __init__(self, **params):
        super().__init__(self, **params)
        self.use(np.convolve)
        self.create_kernel(**params)
    
    def create_kernel(self, **params):
        self.kernel = params.get("kernel", np.ones(5)/5)

    def reduce(self, signal, **params):
        mode = params.get("mode", 'valid')
        return super().reduce(signal, self.kernel, mode)
        
class MovingAverage(Conv):
    def __init__(self, **params):
        super().__init__(self, **params)

    def create_kernel(self, **params):
        l = params.get("kernel_length", 5)
        self.kernel = params.get("kernel", np.ones(l)/l)
    
    
class MyReducer(Reducer):
    def __init__(self, **params):
        super().__init__(**params)

    def reduce(self, signal, **params):
        trigger = params.get("trigger", 1)
        delta = params.get("delta", 1)
        for i in range(0, len(signal), 1):
            if len(signal) - i < 3: 
                break
            if np.abs(signal[i+2] - signal[i]) < trigger and np.abs(signal[i+2] + signal[i] - 2*signal[i+1]) > delta:
                signal[i+1] = signal[i] + (signal[i+2] - signal[i]) / 2

        for i in range(0, len(signal), 1):
            if len(signal) - i < 4: 
                break
            if np.abs(signal[i+3] - signal[i]) < trigger and np.abs(signal[i+3] + signal[i] - (signal[i+1]+signal[i+2])/2) > delta:
                signal[i+1] = signal[i] + (signal[i+3] - signal[i]) / 2
                signal[i+2] = signal[i] + (signal[i+3] - signal[i]) / 2
        return np.where(signal > self.threshold, signal, 0), signal

class MyReducer2(Reducer):
    def __init__(self, **params):
        super().__init__(**params)

    def reduce(self, signal, **params):
        trigger = params.get("trigger", 3)
        delta = params.get("delta", 1)
        for i in range(3):
            for i in range(0, len(signal), 1):
                if len(signal) - i < 3: 
                    break
                if signal[i+1] - signal[i] < -trigger and signal[i+1] - signal[i+2] < -trigger:
                    signal[i+1] = signal[i+2]
                elif signal[i+1] - signal[i] > trigger and signal[i+1] - signal[i+2] > trigger:
                    signal[i+1] = signal[i]
                elif np.abs(signal[i+2] - signal[i]) < delta and np.abs(signal[i+1] - signal[i]) < delta:
                    tmp = (signal[i] + signal[i+1] + signal[i+2]) / 3
                    signal[i], signal[i+1], signal[i+2] = tmp, tmp, tmp
        return np.where(signal > self.threshold, signal, 0), signal
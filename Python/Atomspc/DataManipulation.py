import numpy as np


def reshape_signal(signal, n_signals):
    len_init_signal = len(signal)
    remainder = len_init_signal%n_signals


    if remainder == 0:
        signal = signal.reshape(n_signals, -1)
    else:
        signal = signal[:-remainder].reshape(n_signals, -1)
    return signal


def random_sine(samples,
                seq_length,
                # min_index, max_index, #Indices in the data array that delimit which timesteps to draw from (used for data that isn't randomly generated)
                freq = np.array([0.1, 1]),
                ampl = np.array([0.1, 1]),
                offset = np.array([-1, 1]),
                num_sine_waves = 2,
                seed = 43):
    np.random.seed(seed)
    rows = np.linspace(0, 10, seq_length)

    amplitude = np.random.uniform(ampl[0], ampl[1], (samples, num_sine_waves, 1))
    frequency = np.random.uniform(freq[0], freq[1], (samples, num_sine_waves, 1))
    offset =    np.random.uniform(offset[0], offset[1], (samples, num_sine_waves, 1))
    phase =     np.random.uniform(0, 2*np.pi, (samples, num_sine_waves, 1))
    sinewaves = amplitude * np.sin(2 * np.pi * frequency * rows + phase) + offset

    signals = np.sum(sinewaves, axis = 1) #Sums the number of sine waves  

    return signals
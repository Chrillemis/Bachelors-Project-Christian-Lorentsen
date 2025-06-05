import numpy as np

def save_scaled_csv(arr, file):
    mean = np.mean(arr)
    std = np.std(arr)
    np.savetxt(file, (arr-mean)/std, delimiter = ",")

def save_csv(arr, file):
    np.savetxt(file, arr, delimiter = ",")

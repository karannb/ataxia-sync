import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks

# Arnav's code
def moving_average_filter(data, N):
    '''
    -
    '''
    filtered_data = []
    for i in range(len(data)):
        if i < N - 1:
            # Use as many elements as possible
            window = data[:i+1]
        else:
            window = data[i-N+1:i+1]
        average = sum(window) / len(window)
        filtered_data.append(average)
    return filtered_data

def calcDiff(video:int, keypoints:tuple=(11, 14)):
    '''
    Plots the moving average of the difference between the 
    0th and 1st elements of the keypoints tuple.
    '''
    left, right = keypoints
    plot_list = []
    
    video_keypoints = np.load(f"data/keypoints_ver2/{video}.npy")
    left_keypoints = video_keypoints[:, left, :2]
    right_keypoints = video_keypoints[:, right, :2]
    
    for i in range(len(left_keypoints)):
        plot_list.append(np.linalg.norm(left_keypoints[i] - right_keypoints[i])) #
    
    plot_list = moving_average_filter(plot_list, 11)
    
    window = 11
    plot_list = savgol_filter(plot_list, window, 3)
    
    return plot_list

def plotDiff(video:int):
    '''
    Plots the moving average of the difference between the 
    0th and 1st elements of the keypoints tuple.
    '''
    plot_list = calcDiff(video)
    
    plt.plot(plot_list)
    plt.title(f"Video {video}")
    plt.savefig(f"data/diff_plots/{video}.png")
    plt.close('all')
    
    return

def findPeaks(video:int):
    '''
    Finds peaks in the plot.
    '''
    plot_list = calcDiff(video)
    
    peaks, _ = find_peaks(plot_list) #, height=0.05
    
    return peaks    

if __name__ == "__main__":
    peak_lengths = []
    for video in tqdm(os.listdir("data/keypoints_ver6")):
        if int(video[:-4]) == 73:
            continue
        peaks = findPeaks(int(video[:-4]))
        if len(peaks) == 2:
            print(f"Video {video[:-4]}")
        peak_lengths.append(len(peaks))
    
    print(f"Minimum number of peaks: {min(peak_lengths)}")
    print(f"Average number of peaks: {sum(peak_lengths) / len(peak_lengths)}")
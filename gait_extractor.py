import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

def plotDiff(video:int, keypoints:tuple=(11, 14)):
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
    
    plt.plot(plot_list)
    plt.savefig(f"plots/diff_{video}.png")
    plt.close('all')
    
    return

if __name__ == "__main__":
    plotDiff(1)
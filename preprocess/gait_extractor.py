import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


def moving_average_filter(data, N):
    '''
    -
    '''
    filtered_data = []
    for i in range(len(data)):
        if i < N - 1:
            # Use as many elements as possible
            window = data[:i + 1]
        else:
            window = data[i - N + 1:i + 1]
        average = sum(window) / len(window)
        filtered_data.append(average)
    return filtered_data


def calcDiff(video: int, keypoints: tuple = (11, 14)):
    '''
    Plots the moving average of the difference between the 
    0th and 1st elements of the keypoints tuple.
    '''
    left, right = keypoints
    plot_list = []

    video_keypoints = np.load(f"data/final_keypoints/{video}/kypts.npy")
    left_keypoints = video_keypoints[:, left, :2]
    right_keypoints = video_keypoints[:, right, :2]

    for i in range(len(left_keypoints)):
        plot_list.append(np.linalg.norm(left_keypoints[i] -
                                        right_keypoints[i]))  #

    plot_list = moving_average_filter(plot_list, 11)

    window = 11
    plot_list = savgol_filter(plot_list, window, 4)

    return plot_list


def plotDiff(plot_list: list, peaks: list, video: int):
    '''
    Plots the moving average of the difference between the 
    0th and 1st elements of the keypoints tuple.
    '''

    plt.plot(plot_list)
    for peak in peaks:
        plt.plot(peak, plot_list[peak], 'ro')
    plt.title(f"Video {video}")
    plt.savefig(f"plots/gait_cycles/{video}.png")
    plt.close('all')

    return


def findPeaks(video: int):
    '''
    Finds peaks in the plot.
    '''
    plot_list = calcDiff(video)

    peaks, _ = find_peaks(plot_list, distance=15)  #, height=0.05

    plotDiff(plot_list, peaks, video)

    return peaks


def store_gait_cycles(video: int, peaks: list, non_overlapping: bool = False):
    '''
    Stores the gait cycles in a numpy array.
    3 peaks => one gait cycle; we are storing
    in an overlapping manner, so [1, 2, 3] and 
    [2, 3, 4] are two gait cycles, with the second
    one overlapping with the first.
    '''
    video_keypoints = np.load(f"data/final_keypoints/{video}/kypts.npy")
    gait_cycles = []
    gait_lengths = []
    i = 1
    while (i < len(peaks) - 1):
        # for i in range(1, len(peaks) - 1):
        # Search from the first (because -1)
        # and ignore the last (because +1)
        begin = peaks[i - 1]
        end = peaks[i + 1]
        add2i = 1
        if end - begin < 30:  # too short to capture a complete gait cycle
            pass
        elif end - begin > 75:  # Most such gait cycles have captured multiple gait cycles
            middle = (begin + end) // 2
            gait_cycles.append(video_keypoints[begin:middle])
            gait_cycles.append(video_keypoints[middle:end])
            gait_lengths.append(middle - begin)
            gait_lengths.append(end - middle)
            if non_overlapping:
                add2i = 3
        else:
            gait_cycles.append(video_keypoints[begin:end])
            gait_lengths.append(end - begin)
            if non_overlapping:
                add2i = 3
        i += add2i  # for the overlapping case, this is just a while loop.

    directory = f"data/non_overlapping_gait_cycles" if non_overlapping else f"data/gait_cycles"

    if not os.path.exists(f"{directory}/{video}"):
        os.makedirs(f"{directory}/{video}")
    for i, cycle in enumerate(gait_cycles):
        np.save(f"{directory}/{video}/{i}.npy", cycle)

    return gait_lengths, len(gait_cycles)


if __name__ == "__main__":
    peak_lengths = []
    ovr_gait_lengths = []
    gaits = []
    for video in range(151):
        if not os.path.exists(f"data/final_keypoints/{video}"):
            continue
        peaks = findPeaks(video)
        if len(peaks) <= 2:
            print(f"Video {video}")
        peak_lengths.append(len(peaks))
        gait_lengths, num_gaits = store_gait_cycles(video,
                                                    peaks,
                                                    non_overlapping=True)

        if num_gaits == 0:
            print(f"Video {video} Gait Lengths: {gait_lengths} Peaks: {peaks}")
        ovr_gait_lengths.extend(gait_lengths)
        gaits.append(num_gaits)

    print(
        f"Minimum number of peaks: {min(peak_lengths)} at {peak_lengths.index(min(peak_lengths))}"
    )
    print(f"Average number of peaks: {sum(peak_lengths) / len(peak_lengths)}")
    print(f"Maximum number of peaks: {max(peak_lengths)}")

    print(
        f"Minimum number of gait cycles: {min(gaits)} at {gaits.index(min(gaits))}"
    )
    print(f"Average number of gait cycles: {sum(gaits) / len(gaits)}")
    print(f"Maximum number of gait cycles: {max(gaits)}")

    print(f"Number of gait cycles: {len(ovr_gait_lengths)}")
    print(
        f"Minimum length of gait cycle: {min(ovr_gait_lengths)} at {ovr_gait_lengths.index(min(ovr_gait_lengths))}"
    )
    print(
        f"Average length of gait cycle: {sum(ovr_gait_lengths) / len(ovr_gait_lengths)}"
    )
    print(
        f"Maximum length of gait cycle: {max(ovr_gait_lengths)} at {ovr_gait_lengths.index(max(ovr_gait_lengths))}"
    )

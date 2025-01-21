"""
This module plots and extracts gait cycles from the keypoint data.
The pipeline is:
1. `diffInKeypoints` computes the difference between the left and right keypoints
    using the L2-norm, then applies `movingAverageFilter` and a Savgol filter.
2. `findPeaks` finds the peaks in the difference using `find_peaks` from `scipy.signal`.
3. `storeGAITCycles` stores the gait cycles in a numpy array.

This is the second preprocessing setp for both datasets, before this we extract keypoints 
from videos.
"""

import re
import os
import sys
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks


PLOT=True # turn off if you don't want to plot (we enabled this to find hyperparameters)


def movingAverageFilter(data: np.ndarray, N: int) -> List:
    """
    Simple Moving average

    Args:
        data (ndarray): input data, shape (L)
        N (int): window size

    Returns:
        filtered_data: shape (L)
    """
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


def diffInKeypoints(fname: str, keypoints: tuple = (11, 14), xy: bool =True) -> List:
    """
    Plots the difference between the left and right keypoints for the foot.
    Helps w/ detecting gait cycles.

    Args:
        fname (str): filename of the video
        keypoints (tuple, optional): tuple containing the keypoint locations of the left and right foot. Defaults to (11, 14).
        xy (bool, optional): whether the dataset has (x, y, conf) values. Defaults to True.

    Returns:
        diffs: list of differences
    """
    left, right = keypoints
    diffs = []

    video_keypoints = np.load(fname)
    if xy:
        left_keypoints = video_keypoints[:, left, :2]
        right_keypoints = video_keypoints[:, right, :2]
    else:
        left_keypoints = video_keypoints[:, left]
        right_keypoints = video_keypoints[:, right]

    for i in range(len(left_keypoints)):
        diffs.append(np.linalg.norm(left_keypoints[i] -
                                    right_keypoints[i]))

    # Apply filters
    window = 11
    diffs = movingAverageFilter(diffs, window)
    diffs = savgol_filter(diffs, window, 4)

    return diffs


def plotDiff(diffs: list, peaks: list, identifier: int, dataset_ver: int = 1):
    """
    Plots the moving average computed using `diffInKeypoints` and marks the peaks.

    Args:
        diffs (list): output of `diffInKeypoints`
        peaks (list): list of peaks
        identifier (int): identifier number (used to save the plot)
        dataset_ver (int, optional): version of the dataset. Defaults to 1.
    """

    plt.figure(figsize=(20, 6))  # Set the figure size (width, height) in inches
    plt.plot(diffs)
    for peak in peaks:
        plt.plot(peak, diffs[peak], 'ro')
    plt.title(f"Video {identifier}")
    plt.savefig(f"plots/dataset_{dataset_ver}/{identifier}.png")
    plt.close('all')


def findPeaks(identifier: int, dataset_ver: int = 1) -> List:
    """
    Find the peaks in the `diffInKeypoints` output,
    also plots the whole graph with the peaks marked.

    Args:
        identifier (int): identifier number
        dataset_ver (int, optional): version of the dataset. Defaults to 1.

    Returns:
        peaks (list): list of peaks
        diffs: list of differences
    """
    # different datasets have different keypoint saving formats
    if dataset_ver == 1:
        diffs = diffInKeypoints(f"data/final_keypoints/{identifier}/kypts.npy")
    else:
        diffs = diffInKeypoints(f"data/V2/keypoints/{identifier}.npy", 
                                keypoints=(10, 11), 
                                xy=False)

    # find peaks
    distance = 15 if dataset_ver == 1 else 30
    peaks, _ = find_peaks(diffs, distance=distance)

    return peaks, diffs


def storeGAITCycles(dataset_ver: int, fname: str, peaks: list, non_overlapping: bool = True) -> Tuple[List, int]:
    """
    Store the gait cycles in a numpy array.
    3 peaks => one gait cycle; if we are storing
    in an overlapping manner, so [1, 2, 3] and
    [2, 3, 4] are two gait cycles, with the second
    one overlapping with the first.

    Args:
        dataset_ver (int): version of the dataset
        fname (str): filename of the video
        peaks (list): list of peaks
        non_overlapping (bool, optional): whether or not to use overlapping GAIT cycles. Defaults to True.

    Returns:
        gait_lengths (list): lengths of the gait cycles
        len(gait_cycles) (int): number of gait cycles
    """
    video_keypoints = np.load(fname)
    gait_cycles = []
    gait_lengths = []
    i = 1
    while (i < len(peaks) - 1):
        # for i in range(1, len(peaks) - 1):
        # Search from the first (because -1: L154)
        # and ignore the last (because +1: L155)
        begin = peaks[i - 1]
        end = peaks[i + 1]
        min_length = 30 if dataset_ver == 1 else 0
        max_length = 75 if dataset_ver == 1 else 120
        add2i = 1
        if end - begin < min_length:  # too short to capture a complete gait cycle
            pass
        elif end - begin > max_length:  # such gait cycles have captured multiple gait cycles
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
        i += add2i  # for the overlapping case, this is a simple while loop (because we always add 1)

    directory = f"data/non_overlapping_gait_cycles" if non_overlapping else f"data/gait_cycles"

    # extract the identifier from the filename
    pattern = re.compile(r"data/(?:final_keypoints/(?P<video>\d+)/kypts\.npy|V2/keypoints/(?P<idx>\d+)\.npy)")
    match = pattern.match(fname)
    if match:
        idx = match.group("video") if match.group("video") else match.group("idx")
    else:
        raise ValueError("Invalid filename.")

    # save the gait cycles
    if not os.path.exists(f"{directory}/{idx}"):
        os.makedirs(f"{directory}/{idx}")
    for i, cycle in enumerate(gait_cycles):
        np.save(f"{directory}/{idx}/{i}.npy", cycle)

    return gait_lengths, len(gait_cycles)


def main():

    # find dataset version
    if len(sys.argv) < 2:
        print("Please provide the dataset version.")
        print("Usage: python gait_extractor.py <dataset_version>")
        print("Dataset Version: 1 or 2")
        sys.exit(1)

    dataset_ver = int(sys.argv[1])
    assert dataset_ver in [1, 2], f"Invalid dataset version: {dataset_ver}. Allowed values: 1, 2."

    # containers
    peak_lengths = []
    ovr_gait_lengths = []
    gaits = []

    # create directories for plots
    if PLOT:
        if not os.path.exists("plots"):
            os.makedirs("plots")
        if not os.path.exists(f"plots/dataset_{dataset_ver}"):
            os.makedirs(f"plots/dataset_{dataset_ver}")

    if dataset_ver == 1:
        for video in range(151):
            if not os.path.exists(f"data/final_keypoints/{video}"):
                continue

            # find peaks
            peaks, plot_list = findPeaks(video)
            if len(peaks) <= 2:
                print(f"Less than 2 peaks for Video {video}")

            # plot the signal
            if PLOT:
                plotDiff(plot_list, peaks, video)

            # store gait cycles
            peak_lengths.append(len(peaks))
            gait_lengths, num_gaits = storeGAITCycles(dataset_ver,
                                                      f"data/final_keypoints/{video}/kypts.npy", peaks,
                                                      non_overlapping=True)

            if num_gaits == 0:
                print(f"No Gait Cycles for Video {video} Gait Lengths: {gait_lengths} Peaks: {peaks}")
            ovr_gait_lengths.extend(gait_lengths)
            gaits.append(num_gaits)
    else:
        for idx in range(40):
            # find peaks
            peaks, plot_list = findPeaks(idx, dataset_ver)
            if len(peaks) <= 2:
                print(f"Less than 2 peaks for ID {idx}")

            # plot the signal
            if PLOT:
                plotDiff(plot_list, peaks, idx, dataset_ver)

            # store gait cycles
            peak_lengths.append(len(peaks))
            gait_lengths, num_gaits = storeGAITCycles(dataset_ver,
                                                      f"data/V2/keypoints/{idx}.npy", peaks,
                                                      non_overlapping=True)

            if num_gaits == 0:
                print(f"No Gait Cycles for Video {idx} Gait Lengths: {gait_lengths} Peaks: {peaks}")
            ovr_gait_lengths.extend(gait_lengths)
            gaits.append(num_gaits)

    print(f"Minimum number of peaks: {min(peak_lengths)} at {peak_lengths.index(min(peak_lengths))}")
    print(f"Average number of peaks: {sum(peak_lengths) / len(peak_lengths)}")
    print(f"Maximum number of peaks: {max(peak_lengths)}")

    print(f"Minimum number of gait cycles: {min(gaits)} at {gaits.index(min(gaits))}")
    print(f"Average number of gait cycles: {sum(gaits) / len(gaits)}")
    print(f"Maximum number of gait cycles: {max(gaits)}")

    print(f"Number of gait cycles: {len(ovr_gait_lengths)}")
    print(f"Minimum length of gait cycle: {min(ovr_gait_lengths)} at {ovr_gait_lengths.index(min(ovr_gait_lengths))}")
    print(f"Average length of gait cycle: {sum(ovr_gait_lengths) / len(ovr_gait_lengths)}")
    print(f"Maximum length of gait cycle: {max(ovr_gait_lengths)} at {ovr_gait_lengths.index(max(ovr_gait_lengths))}")


if __name__ == "__main__":
    main()

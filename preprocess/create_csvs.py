import os
import sys
import pandas as pd


def createOVRcsv(dataset_ver: int, non_overlapping: bool = True):
    """
    Creates a CSV file with the following columns:
    video_num: The video number
    gait_num: The number of gait cycles in the video
    score: The score of the video (if it exists)
    label: 1 if the score is greater than 0, 0 otherwise / assigned already

    Args:
        dataset_ver (int): Dataset version being processed (1 or 2)
        non_overlapping (bool, optional): Wether to use non-overlapping gait cycles. Defaults to True.
    """
    dict2csv = {"video_num": [], "gait_num": [], "score": [], "label": []}

    if dataset_ver == 1:
        labels = pd.read_csv("data/Anonymized_ratings.csv")
    else:
        labels = pd.read_csv("data/V2.csv")

    # Check if the number of videos is correct
    num_videos = len(labels)
    if dataset_ver == 1:
        assert num_videos == 151, f"Expected 151 videos, got {num_videos}."
    else:
        assert num_videos == 40, f"Expected 40 videos, got {num_videos}."

    for video in range(num_videos):
        if not os.path.exists(f"data/{'non_overlapping_' if non_overlapping else ''}gait_cycles/{video}"):
            print(f"Skipping video {video} as no gait cycles were detected.")
            continue
        if dataset_ver == 1:
            dict2csv["video_num"].append(video)
            dict2csv["gait_num"].append(len(os.listdir(f"data/{'non_overlapping_' if non_overlapping else ''}gait_cycles/{video}")))
            dict2csv["score"].append(labels["Score"][video])
            dict2csv["label"].append(int(dict2csv["score"][-1] > 0))
        else:
            dict2csv["video_num"].append(labels["idx"][video])
            dict2csv["gait_num"].append(len(os.listdir(f"data/{'non_overlapping_' if non_overlapping else ''}gait_cycles/{video}")))
            dict2csv["label"].append(labels["label"][video])

    if dataset_ver == 2:
        dict2csv.pop("score")
    df = pd.DataFrame.from_dict(dict2csv)
    df.to_csv(f"data/{'non_overlapping_' if non_overlapping else ''}overall.csv", index=False)
    print("Created *_overall.csv.")


def createGAITcsv(dataset_ver: int, non_overlapping: bool = True):
    """
    Creates a CSV file with the following columns:
    index: The index of the gait cycle
    video: The video number
    gait: The gait cycle file name
    score: The score of the video (if it exists)
    label: 1 if the score is greater than 0, 0 otherwise / already assigned

    Args:
        dataset_ver (int): Dataset version being processed
        non_overlapping (bool, optional): Wether to use non-overlapping gait cycles. Defaults to True.
    """
    overall = pd.read_csv(f"data/{'non_overlapping_' if non_overlapping else ''}overall.csv")
    all_gait = {"index": [], "video": [], "gait": [], "score": [], "label": []}
    global_num = 0
    for _, record in overall.iterrows():
        for gait in range(record["gait_num"]):
            all_gait["video"].append(record["video_num"])
            all_gait["gait"].append(str(gait) + ".npy")
            if dataset_ver == 1:
                all_gait["score"].append(record["score"])
            all_gait["label"].append(record["label"])
            all_gait["index"].append(global_num)  # used while splitting the data
            global_num += 1

    if dataset_ver == 2:
        all_gait.pop("score")
    df = pd.DataFrame.from_dict(all_gait)
    df.to_csv(f"data/{'non_overlapping_' if non_overlapping else ''}all_gait.csv", index=False)
    print("Created *_all_gait.csv.")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python create_csvs.py <dataset_version>")
        print("dataset_version: 1 or 2")
        sys.exit(1)

    dataset_ver = int(sys.argv[1])
    assert dataset_ver in [1, 2], f"Invalid dataset version {dataset_ver}"

    createOVRcsv(dataset_ver)
    createGAITcsv(dataset_ver)
    # createOVRcsv(dataset_ver, non_overlapping=False)
    # createGAITcsv(dataset_ver, non_overlapping=False)

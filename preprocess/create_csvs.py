import os
import pandas as pd


def createOVRcsv(non_overlapping=False):
    """
    Creates a CSV file with the following columns:
    video_num: The video number
    gait_num: The number of gait cycles in the video
    score: The score of the video
    label: 1 if the score is greater than 0, 0 otherwise

    Args:
        non_overlapping (bool, optional): Wether to use non-overlapping gait cycles. Defaults to False.
    """
    dict2csv = {"video_num": [], "gait_num": [], "score": [], "label": []}

    labels = pd.read_csv("data/Anonymized_ratings.csv")

    for video in range(151):
        if not os.path.exists(f"data/{'non_overlapping_' if non_overlapping else ''}gait_cycles/{video}"):
            continue
        dict2csv["video_num"].append(video)
        dict2csv["gait_num"].append(len(os.listdir(f"data/{'non_overlapping_' if non_overlapping else ''}gait_cycles/{video}")))
        dict2csv["score"].append(labels["Score"][video])
        dict2csv["label"].append(int(dict2csv["score"][-1] > 0))

    df = pd.DataFrame.from_dict(dict2csv)
    df.to_csv(f"data/{'non_overlapping_' if non_overlapping else ''}overall.csv", index=False)

    return


def createGAITcsv(non_overlapping=False):
    """
    Creates a CSV file with the following columns:
    index: The index of the gait cycle
    video: The video number
    gait: The gait cycle file name
    score: The score of the video
    label: 1 if the score is greater than 0, 0 otherwise

    Args:
        non_overlapping (bool, optional): Wether to use non-overlapping gait cycles. Defaults to False.
    """
    overall = pd.read_csv(f"data/{'non_overlapping_' if non_overlapping else ''}overall.csv")
    all_gait = {"index": [], "video": [], "gait": [], "score": [], "label": []}
    global_num = 0
    for _, record in overall.iterrows():
        for gait in range(record["gait_num"]):
            all_gait["video"].append(record["video_num"])
            all_gait["gait"].append(str(gait) + ".npy")
            all_gait["score"].append(record["score"])
            all_gait["label"].append(record["label"])
            all_gait["index"].append(global_num)  # used while splitting the data
            global_num += 1

    df = pd.DataFrame.from_dict(all_gait)
    df.to_csv(f"data/{'non_overlapping_' if non_overlapping else ''}all_gait.csv", index=False)

    return


if __name__ == "__main__":

    createOVRcsv()
    createGAITcsv()
    createOVRcsv(non_overlapping=True)
    createGAITcsv(non_overlapping=True)

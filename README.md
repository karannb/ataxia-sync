# AtGCN
![AtGCN](assets/atgcn.png)

A lightweight Spatiotemporal GCN for Ataxia Detection.
Official implementation for AtGCN, accepted to [AIME-2025](https://aime25.aimedicine.info/) as a full length paper + talk.

[![arXiv](https://img.shields.io/badge/arXiv-2410.22862-b31b1b.svg)](https://arxiv.org/abs/2410.22862)

To make our pipeline fuly reproducible, we provide all the code used and also instructions on how to reproduce.

## Data Preparation - V1
Please download the first dataset we have used from [here](https://github.com/ROC-HCI/Automated-Ataxia-Gait); if you want to use your own dataset, the same instructions as below follow.

1. First create a folder named `data/` in the root directory of the project.
2. Use the notebooks, `FRCNN.ipynb`, `SORT.ipynb` and finally `OpenPose.ipynb` in this order from the `src/preprocess/` directory on Google Colab to get keypoints of all patients, store these inside the `data/` directory in a folder named `final_keypoints/`. (Note: this step takes **~4 hours** for the 149 videos we had using a T4 GPU)
3. Using these keypoints we can now extract *non-overlapping* gait cycles. To do this, run the `gait_extractor.py` script from the `src/preprocess/` directory. This will create a folder named `non_overlapping_gait_cycles/` inside the `data/` directory (or `gait_cycles/` if you want to extract overlapping gait cycles, we use non-overlapping gait-cycles in the paper and have not tested the overlapping case). Run with -
```bash
python src/preprocess/gait_extractor.py 1
```
This should print something like - 
```bash
Minimum number of peaks: xx at yy
Average number of peaks: xy.z
Maximum number of peaks: zz
Minimum number of gait cycles: xx at yy
Average number of gait cycles: xy.z
Maximum number of gait cycles: zz
Number of gait cycles: abc
Minimum length of gait cycle: xx at yy
Average length of gait cycle: xy.z
Maximum length of gait cycle: zz at yy
```
4. Finally, we create CSV files from which we can quickly retrieve data and use it in our training loop, you can use `create_csvs.py` from the `src/preprocess/` directory to create these CSV files, in our format.

This will create the folder with all the files and extracted Gait Cycles (overlapping and non-overlapping, can do both).

## Data Preparation - V2
We have another dataset of 40 videos, which can be downloaded from [here](https://data.mendeley.com/datasets/2vkk2r9tx3/1) / [paper](https://hisham246.github.io/uploads/iecbes2022khalil.pdf). This already has extracted keypoints. **NOTE:** according to current preprocessing, you can only use one dataset at a time, as the CSV files and folders are overwritten. To switch between datasets, just rename the folders accordingly.

1. Use the `create_npy.py` from the `src/preprocess/` directory, which will store the keypoints in .npy files and create a V2.csv file which contains a mapping of original files to assigned IDs and their labels, we also add a **center** coordinate to the keypoints (last keypoint).
2. Then use the `gait_extractor.py` to extract non-overlapping gait cycles (will be saved similarly to the first dataset). Run with 
```bash
python src/preprocess/gait_extractor.py 2
```
3. Finally, create the CSV files using `create_csvs.py` from the `src/preprocess/` directory.

A few notes on this dataset:
- The dataset available online is the **augmented** version, we deaugment and use it in our case.
- The authors report 31 augmentations + 1 original, however, their [implementation](https://github.com/hisham246/AtaxiaNet/tree/main) and the trend in the dataset observed by plotting shows original + 32 augmentations, so we divide the dataset by 33 and use the first \[:x//33\] frames per video.


## Model checkpoints
We have already uploaded the required checkpoints inside `ckpts/`. These were taken from -
- STGCN checkpoints : [here](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).
- GaitGraph checkpoints : [here](https://github.com/tteepe/GaitGraph).


## Training
First we need to install the dependencies (We have used python 3.9.19)
```bash
pip install -r requirements.txt
```
OR if you use conda, then first run
```bash
conda create -n ataxia-sync python=3.9.19
```
To train a model you can use the `runner.sh` with `sbatch` on an HPC or with `bash` on your local machine.
```bash
bash runner.sh
```
You can also check the flags using 
```python
python src/trainer.py --help # or -h
```
To reproduce our results, you can run the `runner.sh` as is. A full 10-fold CV run takes about **40 minutes** on a single V100 GPU.

## Analysis
You can also reproduce all the plots and tables in the paper using `src/analyze.py`, just uncomment all the lines after `if __name__ == "__main__"`, and run,
```bash
python3 src/analyze.py
```
### Main results
![Results](assets/results.png)

Please create an issue if you need some functionality or the code doesn't work as intended. Thank you!

## Citation
We will update this with a newer version soon, for now please use the following:
```bibtex
@misc{bania2024atgcn,
      title={AtGCN: A Graph Convolutional Network For Ataxic Gait Detection}, 
      author={Karan Bania and Tanmay Verlekar},
      year={2024},
      eprint={2410.22862},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.22862}, 
}
```
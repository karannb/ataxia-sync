# ataxia-sync
Spatio Temporal GCN for Ataxia Detection.

## Data Preparation
To make our pipeline fuly reproducible, we provide all the code used and also instructions on how to run it; we also provide a zipped version of the dataset so you can skip the data preparation step. However, if you want to use your own dataset, you can follow the instructions below.

1. First create a folder named `data/` in the root directory of the project.
2. Use the `FRCNN_OpenPose_SORT.ipynb` from the `preprocess/` directory on Google Colab to get keypoints of the patient, store these inside the `data/` directory in a folder named `final_keypoints/`. (Note: this step takes **~4 hours** for the 149 videos we had)
3. Using these keypoints we can now extract overlapping as well as non-overlapping gait cycles. To do this, run the `gait_extractor.py` script from the `preprocess/` directory. This will create a folder named `gait_cycles/` inside the `data/` directory. (or `non_overlapping_gait_cycles/` if you want to extract non-overlapping gait cycles)
4. Finally, we create CSV files from which we can quickly retrieve data and use it in our training loop, you can use `create_csvs.py` from the `preprocess/` directory to create these CSV files, in our format.

To use our dataset directly, you can just run 
```bash
unzip data.zip
```
This will create the folder withh all the files and extracted Gait Cycles (overlapping and non-overlapping, both).

Done!

## Training
To train a model you can use the `runner.sh` with `sbatch` on an HPC or with `bash` on your local machine.
```bash
bash runner.sh
```
You can also check the flags using 
```python
python main.py --help # or -h
```
To reproduce our results, you can run the `runner.sh` as is. A full 10-fold CV run took about **40 minutes** on a single V100 GPU.

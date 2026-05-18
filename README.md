# SINet

This repository is a lightweight working version of **SINet: Camouflaged Object Detection**.

It keeps the core SINet model code and adds practical scripts/notebooks for training and inference. The training script now supports multiple GPUs through `torch.nn.DataParallel` when more than one CUDA device is available.

Original project: https://github.com/DengPingFan/SINet

## Files

```text
Dataset/                         # Local train/test datasets
Snapshot/                        # Model checkpoints
Src/                             # SINet model and data utilities
MyTrainKaggle.py                 # Training script with multi-GPU support
MyTest.py                        # Batch inference and MAE calculation
SINet_kaggle_train_eval.ipynb    # Kaggle training/evaluation notebook
SINet_single_image_prediction.ipynb
requirement.txt
```

## Setup

```bash
pip install -r requirement.txt
```

Install a PyTorch build that matches your CUDA environment if the default dependency install does not provide the right one.

## Dataset Layout

```text
Dataset/
├── TrainDataset/
│   ├── Imgs/
│   └── GT/
└── TestDataset/
    └── <DATASET_NAME>/
        ├── Imgs/
        └── GT/
```

## Training

```bash
python3 MyTrainKaggle.py \
  --epoch 40 \
  --batchsize 64 \
  --train_img_dir ./Dataset/TrainDataset/Imgs/ \
  --train_gt_dir ./Dataset/TrainDataset/GT/ \
  --save_model ./Snapshot/new-SINet/
```

If multiple GPUs are available, the script will use them automatically. Kaggle can provide two GPUs for free, and `--batchsize 64` works well with that setup.

## Testing

```bash
python3 MyTest.py \
  --test_root ./Dataset/TestDataset \
  --model_path ./Snapshot/2020-CVPR-SINet/SINet_40.pth \
  --test_save ./Result/2020-CVPR-SINet-New/ \
  --datasets CAMO COD10K CHAMELEON \
  --device auto
```

Predicted masks are saved under:

```text
Result/2020-CVPR-SINet-New/<DATASET_NAME>/
```

For a quick check:

```bash
python3 MyTest.py --limit 5 --device cpu
```

## Citation

If you use this project in research, please cite the original SINet paper:

```bibtex
@inproceedings{fan2020Camouflage,
  title={Camouflaged Object Detection},
  author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}
```

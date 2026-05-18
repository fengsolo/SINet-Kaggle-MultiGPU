import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import sys
from PIL import Image
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset


def print_progress(dataset, current, total, mae, mean_mae, width=40):
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    bar = '#' * filled + '-' * (width - filled)
    sys.stdout.write(
        '\r[Eval-Test] Dataset: {} [{}] {}/{} | MAE: {:.6f} | Mean MAE: {:.6f}'.format(
            dataset, bar, current, total, mae, mean_mae
        )
    )
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet/SINet_40.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/')
parser.add_argument('--test_root', type=str, default='./Dataset/TestDataset')
parser.add_argument('--datasets', nargs='+', default=['CAMO', 'COD10K', 'CHAMELEON'])
parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
parser.add_argument('--limit', type=int, default=0, help='optional max images per dataset; 0 means all')
opt = parser.parse_args()

if opt.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(opt.device)

model = SINet_ResNet50(pretrained=False).to(device)
model.load_state_dict(torch.load(opt.model_path, map_location=device))
model.eval()
print('[INFO] Using device: {}'.format(device))

for dataset in opt.datasets:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    test_loader = test_dataset(image_root='{}/{}/Imgs/'.format(opt.test_root, dataset),
                               gt_root='{}/{}/GT/'.format(opt.test_root, dataset),
                               testsize=opt.testsize)
    total = test_loader.size if opt.limit <= 0 else min(opt.limit, test_loader.size)
    img_count = 1
    mae_sum = 0.0
    for iteration in range(total):
        # load data
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        # inference
        with torch.no_grad():
            _, cam = model(image)
        # reshape and squeeze
        cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        Image.fromarray((cam * 255).astype(np.uint8)).save(save_path+name)
        # evaluate
        mae = np.mean(np.abs(cam - gt))
        mae_sum += mae
        print_progress(dataset, img_count, total, mae, mae_sum / img_count)
        img_count += 1
    print('[Eval-Test] Dataset: {}, Mean MAE: {}'.format(dataset, mae_sum / max(total, 1)))

print("\n[Congratulations! Testing Done]")

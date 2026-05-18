import os
import argparse
import torch
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import get_loader

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batchsize', type=int, default=8)
parser.add_argument('--trainsize', type=int, default=352)
parser.add_argument('--save_epoch', type=int, default=10) # save every N epochs
parser.add_argument('--save_model', type=str, default='/kaggle/working/Snapshot/new-SINet/')
parser.add_argument(
    '--train_img_dir',
    type=str,
    default='/kaggle/input/datasets/wenfengk/cod-sinet/SINet/Dataset/TrainDataset/Imgs/'
)
parser.add_argument(
    '--train_gt_dir',
    type=str,
    default='/kaggle/input/datasets/wenfengk/cod-sinet/SINet/Dataset/TrainDataset/GT/'
)
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(opt.save_model, exist_ok=True)

model = SINet_ResNet50(channel=32, pretrained=True).to(device)

if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = torch.nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
loss_func = torch.nn.BCEWithLogitsLoss()

train_loader = get_loader(
    opt.train_img_dir,
    opt.train_gt_dir,
    batchsize=opt.batchsize,
    trainsize=opt.trainsize,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

print(f'Using device: {device}')
print(f'Train image dir: {opt.train_img_dir}')
print(f'Train GT dir: {opt.train_gt_dir}')
print(f'Save model dir: {opt.save_model}')
print(f'Training images: {len(train_loader.dataset)}')
print(f'Steps per epoch: {len(train_loader)}')

for epoch in range(1, opt.epoch + 1):
    model.train()
    total_loss = 0.0

    for step, (images, gts) in enumerate(train_loader, start=1):
        images = images.to(device)
        gts = gts.to(device)

        optimizer.zero_grad()
        cam_sm, cam_im = model(images)

        loss_sm = loss_func(cam_sm, gts)
        loss_im = loss_func(cam_im, gts)
        loss = loss_sm + loss_im

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 20 == 0 or step == len(train_loader):
            print(
                f'Epoch [{epoch}/{opt.epoch}] '
                f'Step [{step}/{len(train_loader)}] '
                f'Loss: {loss.item():.4f}'
            )

    mean_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} mean loss: {mean_loss:.4f}')

    if epoch % opt.save_epoch == 0 or epoch == opt.epoch:
        save_path = os.path.join(opt.save_model, f'SINet_{epoch}.pth')
        torch.save(model.state_dict(), save_path)
        print(f'Saved: {save_path}')
from doctr.datasets.datasets import AbstractDataset
import json
from pathlib import Path
import os
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import (
    Compose,
    GaussianBlur,
    RandomGrayscale,
    Normalize,
    RandomPerspective,
    RandomPhotometricDistort,
)
from doctr.models import recognition
import torch
from doctr import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from doctr.utils.metrics import TextMatch
from torch.utils.tensorboard import SummaryWriter
import time


class DSet(AbstractDataset):
    def __init__(self,
                 img_folder: str,
                 labels,
                 **kwargs,
                 ):
        super().__init__(img_folder, **kwargs)
        self.data = []
        for line in labels:
            data = json.loads(line)
            img_name = data['filename'].replace('data2/', 'data/')
            label = data['text']
            if not os.path.exists(os.path.join(self.root, img_name)):
                continue
            self.data.append((img_name, label))

    def merge_dataset(self, ds: AbstractDataset) -> None:
        # Update data with new root for self
        self.data = [(str(Path(self.root).joinpath(img_path)), label) for img_path, label in self.data]
        # Define new root
        self.root = Path("/")
        # Merge with ds data
        for img_path, label in ds.data:
            self.data.append((str(Path(ds.root).joinpath(img_path)), label))


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, epoch, writer):
    model.train()
    # Iterate over the batches of the dataset
    for idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        images = batch_transforms(images)
        optimizer.zero_grad()
        train_loss = model(images, targets)["loss"]
        train_loss.backward()
        writer.add_scalar('Loss/train', train_loss.cpu().detach().numpy(), (epoch + 1) * len(train_loader) + idx)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, writer):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    for idx, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        images = batch_transforms(images)
        out = model(images, targets, return_preds=True)
        # Compute metric
        if len(out["preds"]):
            words, _ = zip(*out["preds"])
        else:
            words = []
        val_metric.update(targets, words)

        val_loss += out["loss"].item()
        writer.add_scalar('Loss/val', out["loss"].cpu().detach().numpy(), (epoch + 1) * len(val_loader) + idx)
        batch_cnt += 1

    val_loss /= batch_cnt
    result = val_metric.summary()
    for key in result:
        writer.add_scalar('key', result[key], (epoch + 1))
    return val_loss, result["raw"], result["unicase"]


if __name__ == '__main__':
    batch_size = 48
    workers = 1
    input_size = 48
    weight_decay = 1e-6
    lr = 1e-3
    epochs = 10
    arch = 'vitstr_base'
    exp_name = 'tt1'
    device = 'cuda'
    # Load a recognition Dataset
    charmap = open('data/charset_enth.txt').read().split('\n')
    charset = [x.split('	')[1] for x in charmap]
    vocabs = ''.join(charset)

    jsonl = 'data/train.jsonl'
    if os.path.exists("/project/lt200060-capgen/coco"):
        src_dir = "/project/lt200060-capgen/palm/capocr"
        workdir = 'workdir'
    elif os.path.exists("/media/palm/Data/capgen/"):
        src_dir = "/media/palm/Data/ocr/"
        workdir = '/media/palm/Data/ocr/cp/doctr'
    else:
        src_dir = "/media/palm/Data/ocr/"
        workdir = '/media/palm/Data/ocr/cp/doctr'
    labels = open(jsonl).read().split('\n')[:-1]
    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))
    train_set = DSet(
        img_folder=src_dir,
        labels=labels[:450000],
        img_transforms=Compose(
            [
                T.Resize((input_size, 4 * input_size), preserve_aspect_ratio=True),
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                RandomGrayscale(p=0.1),
                RandomPhotometricDistort(p=0.1),
                T.RandomApply(T.RandomShadow(), p=0.4),
                T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
                T.RandomApply(GaussianBlur(3), 0.3),
                RandomPerspective(distortion_scale=0.2, p=0.3),
            ]
        ))
    val_set = DSet(img_folder=src_dir, labels=labels[450000:])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        drop_last=True,
        num_workers=workers,
        sampler=RandomSampler(train_set),
        collate_fn=train_set.collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        drop_last=False,
        num_workers=workers,
        sampler=SequentialSampler(val_set),
        collate_fn=val_set.collate_fn,
    )

    model = recognition.__dict__[arch](pretrained=False, vocab=vocabs).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr,
        betas=(0.95, 0.99),
        eps=1e-6,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, epochs * len(train_loader), eta_min=lr / 25e4)

    val_metric = TextMatch()
    writer = SummaryWriter(log_dir=f'log/{time.time()}')

    min_loss = np.inf
    for epoch in range(epochs):
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, epoch, writer)

        # Validation loop at the end of each epoch
        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric, epoch, writer)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            torch.save(model.state_dict(), f"{workdir}/{arch}/{epoch:02d}.pt")
            min_loss = val_loss

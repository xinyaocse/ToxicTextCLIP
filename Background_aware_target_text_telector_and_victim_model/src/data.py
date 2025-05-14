import os
import torch
import logging
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.index_sampler import IndexSampler

from utils.augment_text import _augment_text
from utils.augment_image import _augment_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor, cross_aug=False):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep=delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions = processor.process_text(df[caption_key].tolist())
        self.processor = processor
        self.cross_aug = cross_aug

        if (cross_aug):
            self.augment_captions = processor.process_text(
                [_augment_text(caption) for caption in df[caption_key].tolist()])

        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}

        if (self.cross_aug):
            item["input_ids"] = self.augment_captions["input_ids"][idx]
            item["attention_mask"] = self.augment_captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(
                _augment_image(os.path.join(self.root, self.images[idx])))
            item["index"] = idx
        else:
            item["input_ids"] = self.captions["input_ids"][idx]
            item["attention_mask"] = self.captions["attention_mask"][idx]
            item["pixel_values"] = self.processor.process_image(
                Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
            item["index"] = idx

        return item


def get_train_dataloader(options, processor):
    path = options.train_data
    if (path is None): return None

    batch_size = options.batch_size

    dataset = ImageCaptionDataset(path, image_key=options.image_key, caption_key=options.caption_key,
                                  delimiter=options.delimiter, processor=processor, cross_aug=options.cross_aug)
    if options.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    logging.info(str(sampler))
    # if not fixmatch:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), num_workers=options.num_workers,
                            pin_memory=True, sampler=sampler, drop_last=True)
    dataloader.num_samples = len(dataloader) * batch_size
    dataloader.num_batches = len(dataloader)
    return dataloader


def get_validation_dataloader(options, processor):
    path = options.validation_data
    if (path is None): return

    dataset = ImageCaptionDataset(path, image_key=options.image_key, caption_key=options.caption_key,
                                  delimiter=options.delimiter, processor=processor, cross_aug=options.cross_aug)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers,
                            pin_memory=True, sampler=None, drop_last=False)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


def load(options, processor):
    data = {}

    data["train"] = get_train_dataloader(options, processor)
    data["validation"] = get_validation_dataloader(options, processor)

    return data
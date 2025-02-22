import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, hflip

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        self._create_mask_values()

    def _create_mask_values(self):
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class HalfDataset(BasicDataset):
    def __init__(self, root_dir: str):
        root_dir = Path(root_dir)
        super().__init__(root_dir / 'imgs', root_dir / 'masks')

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        mask[mask>0] = 1

        # heatmaps = []
        # for n in ['left', 'middle', 'right']:
        #     fn = img_file[0].parent.parent / 'endpoints' / img_file[0].name.replace('.jpg', f'_{n}.png')
        #     heatmaps.append(self.preprocess(self.mask_values, load_image(fn), self.scale, is_mask=False)[0])
        # heatmaps = torch.stack(heatmaps)
        # heatmaps = heatmaps.sum(0, keepdim=True)
        fn = img_file[0].parent.parent / 'endpoints' / img_file[0].name.replace('.jpg', f'_all.png')
        heatmaps = self.preprocess(self.mask_values, load_image(fn), self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(img).float().contiguous(),
            'mask': torch.as_tensor(mask).long().contiguous(),
            'endpoints': torch.as_tensor(heatmaps).float().contiguous(),
        }


    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        # pil_img = pil_img.resize((256, 256), resample=Image.NEAREST)
        if is_mask:
            tensor = torch.tensor(np.array(pil_img)).to(int)
        else:
            tensor = to_tensor(pil_img)
        return tensor

    def _create_mask_values(self):
        self.mask_values = None

class HFlipDataset(Dataset):
    def __init__(self, parent) -> None:
        self.parent = parent

    def __len__(self):
        return 2 * len(self.parent)

    def __getitem__(self, index):
        item = self.parent[index // 2]
        if index & 1:
            remap = torch.tensor([0, 2, 1])
            # item['mask'] = remap[hflip(item['mask'])]
            item['mask'] = hflip(item['mask'])
            item['image'] = hflip(item['image'])
            if False:
                flipped = hflip(item['endpoints'])
                item['endpoints'][0] = flipped[2]
                item['endpoints'][1] = flipped[1]
                item['endpoints'][2] = flipped[0]
            else:
                item['endpoints'] = hflip(item['endpoints'])

        return item


if __name__ == '__main__':
    from vi3o import view, flipp
    flipp()
    for item in HFlipDataset(HalfDataset('pdata/train')):
        view(255 * item['image'].numpy().transpose(1, 2, 0))
        view(127 * item['mask'].numpy())
        view(255 * item['endpoints'].numpy()[0])
        flipp(pause=True)
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torch.nn.functional import max_pool2d
from time import time

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, experiment=None):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    stats = EvalStats(16)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, endpoint_true = batch['image'], batch['mask'], batch['endpoints']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            endpoint_true = endpoint_true.to(device=device)

            # predict the mask
            mask_pred, endpoint_pred = net(image)
            endpoint_pred = torch.sigmoid_(endpoint_pred)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred.squeeze(1)) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true_onehot[:, 1:], reduce_batch_first=False)

            # view(255*(endpoint_pred[0,0].detach().cpu().numpy() > 0.3))
            # view(255*(endpoint_true[0,0].detach().cpu().numpy() > 0.3))

            for i in range(len(endpoint_true)):
                expected = map2points(endpoint_true[i], 0.5, max_objs=50)
                result = map2points(endpoint_pred[i], 0.5, max_objs=50)
                stats.update(result, expected)

            segmentation_loss, endpoint_loss, loss = net.loss((mask_pred, endpoint_pred), (mask_true, endpoint_true))
            stats.update_loss(loss)

        if experiment is not None:
            import wandb
            experiment.log(dict(eval={
                'images': wandb.Image(image[0].cpu()),
                'endpoints': {
                    'loss': endpoint_loss,
                    'true': wandb.Image(endpoint_true[0].float().cpu()),
                    'pred': wandb.Image(endpoint_pred[0].float().cpu()),
                },
                'masks': {
                    'loss': segmentation_loss,
                    'true': wandb.Image(mask_true[0].float().cpu()),
                    'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu() if net.n_classes > 1 else torch.sigmoid(mask_pred[0].float().cpu())),
                },
                'loss': loss,
            }))

    net.train()
    stats.dice_score = (dice_score / max(num_val_batches, 1)).item()
    return stats

def map2points(pointmap, th=0.5, r=4, max_objs=None):
    mask = torch.zeros_like(pointmap)
    maxmap = max_pool2d(pointmap.unsqueeze(0), 2*r+1, 1, r, 1, False, False).squeeze()
    n, h, w = pointmap.shape
    pkt = []
    for c, y, x in torch.nonzero((maxmap == pointmap) & (pointmap > th), as_tuple=False):
        if mask[c, y, x] == 0:
            pkt.append((x, y, c))
            mask[c, max(y-r,0):min(y+r+1,h), max(x-r,0):min(x+r+1,w)] = 1
            if max_objs is not None and len(pkt) >= max_objs:
                break
    return torch.tensor(pkt)

class EvalStats:
    def __init__(self, max_gt_distance=4, border=None):
        self.max_gt_distance = max_gt_distance
        self.reset()
        self.border = border

    def distance(self, a, b):  # (x, y, class)
        d = ((a[:, :2] - b[:2]) ** 2).sum(1).to(float)
        d[a[:, 2] != b[2]] = float('Inf')
        return d

    def reset(self):
        self.detected = self.missed = self.extra = self.border_miss = 0
        self.loss_sum = self.loss_cnt = self.updates = 0
        self.reset_time = self.last_update_time = time()

    def ignore(self, p, shape):
        if self.border is None:
            return False
        _, h, w = shape
        brd = self.border
        return not (brd <= p[0] <= w-brd and brd <= p[1] <= h-brd)

    def update(self, result, expected, shape=None):
        self.last_update_time = time()
        self.updates += 1
        detected, detected_location, missed, extra = [], [], [], []
        matched = set()
        if len(result) > 0:
            for p in expected:
                dists = self.distance(result, p)
                i = dists.argmin().item()
                if dists[i] < self.max_gt_distance**2 and i not in matched:
                    matched.add(i)
                    self.detected += 1
                    detected.append(p)
                    detected_location.append(result[i])
                else:
                    if not self.ignore(p, shape):
                        missed.append(p)
            for i in range(len(result)):
                if i not in matched:
                    if not self.ignore(result[i], shape):
                        extra.append(result[i])
        else:
            for p in expected:
                if not self.ignore(p, shape):
                    missed.append(p)

        self.extra += len(extra)
        self.missed += len(missed)
        return detected, detected_location, missed, extra

    def update_loss(self, loss):
        self.loss_sum += loss
        self.loss_cnt += 1

    @property
    def precision(self):
        n = self.detected + self.extra
        if n == 0:
            return 0.0
        return self.detected / n

    @property
    def recall(self):
        n = self.detected + self.missed
        if n == 0:
            return 0.0
        return self.detected / n

    @property
    def f1(self):
        p, r =  self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def total(self):
        return self.detected + self.missed

    @property
    def loss(self):
        if self.loss_cnt == 0:
            return float('NaN')
        return self.loss_sum / self.loss_cnt

    @property
    def sample_time(self):
        return (self.last_update_time - self.reset_time) / self.updates

    def __str__(self) -> str:
        return f'Detected {self.detected}/{self.total} with {self.extra} extra. Precition: {self.precision*100:.1f}% Recall: {self.recall*100:.1f}% F1: {self.f1*100:.1f}% SamplesPerS: {1/self.sample_time}'

    def dict(self, prefix=''):
        d = dict(loss=self.loss, f1=self.f1, precision=self.precision, recall=self.recall)
        return {prefix + k: v for k, v in d.items()}

if __name__ == '__main__':
    from unet import UNet4k
    import os
    from utils.data_loading import HalfDataset, HFlipDataset
    from torch.utils.data import DataLoader

    device = torch.device('cuda')
    model = UNet4k(3, 1, 1).to(device)
    model = model.to(memory_format=torch.channels_last)

    # state_dict = torch.load("checkpoints3/checkpoint_epoch26.pth", map_location=device)
    state_dict = torch.load("checkpoints3/checkpoint_epoch50.pth", map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)

    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    # val_set = HFlipDataset(HalfDataset('pdata/eval'))
    val_set = HFlipDataset(HalfDataset('pdata/train'))
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    stats = evaluate(model, val_loader, device, True)
    print('Points:', stats)
    print('Segmentation F1:', stats.dice_score)
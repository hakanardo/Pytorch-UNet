import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, experiment=None):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, endpoint_true = batch['image'], batch['mask'], batch['endpoints']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred, endpoint_pred = net(image)
            endpoint_pred = torch.sigmoid_(endpoint_pred)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true_onehot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true_onehot[:, 1:], reduce_batch_first=False)
        if experiment is not None:
            import wandb
            experiment.log(dict(eval={
                'images': wandb.Image(image[0].cpu()),
                'endpoints': {
                    'true': wandb.Image(endpoint_true[0].float().cpu()),
                    'pred': wandb.Image(endpoint_pred[0].float().cpu()),
                },
                'masks': {
                    'true': wandb.Image(mask_true[0].float().cpu()),
                    'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
                }
            }))

    net.train()
    return dice_score / max(num_val_batches, 1)

if __name__ == '__main__':
    from unet import UNet4k
    import os
    from utils.data_loading import HalfDataset, HFlipDataset
    from torch.utils.data import DataLoader

    device = 'cuda'
    model = UNet4k(3, 3, 0).to(device)
    model = model.to(memory_format=torch.channels_last)

    state_dict = torch.load("checkpoints1/checkpoint_epoch14.pth", map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)

    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    val_set = HFlipDataset(HalfDataset('pdata/eval'))
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    evaluate(model, val_loader, device, True)
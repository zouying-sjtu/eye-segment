import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
import copy

from dice_loss import dice_coeff
def save_pred_mask(mask_pred,mask_gt, idx):
    save_path = './preds'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_mask_pred = os.path.join(save_path, str(idx) + '.jpg')
    mask_pred[mask_pred < 0.5] = 0
    mask_pred[mask_pred > 0.5] = 255
    save_mask = copy.deepcopy(mask_pred.detach().numpy())
    save_mask = np.transpose(save_mask, [1,2,0])
    cv2.imwrite(save_mask_pred, np.array(save_mask, np.uint8))

    save_mask_path = os.path.join(save_path, str(idx) + 'gt.jpg')
    mask_gt[mask_gt < 0.5] = 0
    mask_gt[mask_gt > 0.5] = 255
    save_mask_gt = copy.deepcopy(mask_gt.detach().numpy())
    save_mask_gt = np.transpose(save_mask_gt, [1, 2, 0])
    cv2.imwrite(save_mask_path, np.array(save_mask_gt, np.uint8))


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        true_mask[true_mask == 255] = 1

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]

        save_pred_mask(mask_pred, true_mask, i)
        mask_pred = (mask_pred > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
    return tot / (i + 1)

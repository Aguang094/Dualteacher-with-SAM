import os
import argparse
import random
import copy
import json
import math
import sys
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from tqdm import tqdm


from dataset import *
from module_list import *
from utilsm.loss import *
from validate import *
from Model.Unet import UNet_kMaX, opmoudle
from utils.pyt_utils import load_model

try:
    import medim
except ImportError:
    print("Warning: can't import")
    sys.exit(1)

try:
    from boundary_contrastive import BoundaryContrastiveLoss
except ImportError:
    print("Warning: can't find boundary_contrastive")


def resize_volume(tensor, target_size, mode='trilinear', align_corners=False):
    if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        return F.interpolate(tensor, size=target_size, mode=mode, align_corners=align_corners)
    else:
        return F.interpolate(tensor, size=target_size, mode=mode)

def get_points_from_mask(mask, num_points=10):
    B, D, H, W = mask.shape
    device = mask.device

    point_coords = []
    point_labels = []

    for b in range(B):
        fg_indices = torch.nonzero(mask[b] == 1, as_tuple=False)
        bg_indices = torch.nonzero(mask[b] == 0, as_tuple=False)

        if len(fg_indices) < 5 or len(bg_indices) < 5:
            point_coords.append(torch.zeros((num_points, 3), device=device))
            point_labels.append(torch.zeros((num_points), device=device))
            continue

        n_pos = num_points // 2
        n_neg = num_points - n_pos

        idx = torch.randint(0, len(fg_indices), (n_pos,), device=device)
        pos_points = fg_indices[idx]

        idx = torch.randint(0, len(bg_indices), (n_neg,), device=device)
        neg_points = bg_indices[idx]

        coords = torch.cat([pos_points, neg_points], dim=0)
        coords = torch.flip(coords, [1])

        labels_pos = torch.ones(len(pos_points), device=device)
        labels_neg = torch.zeros(len(neg_points), device=device)
        labels = torch.cat([labels_pos, labels_neg], dim=0)

        point_coords.append(coords)
        point_labels.append(labels)

    return torch.stack(point_coords), torch.stack(point_labels)


parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with SAM Assist')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
parser.add_argument('--sam_ckpt', type=str, default='checkpoint/sam_med3d_turbo.pth', help='Path to SAM checkpoint')
args = parser.parse_args()

num_classes = 2
alpha_ema = 0.99
dynamic_threshold_bg = 1 / num_classes
dynamic_threshold_fg = 1 / num_classes

VAL_INTERVAL = 1


def compute_unsupervised_loss_with_uncertainty(predict, target, logits, strong_threshold, uncertain_area=None):
    if uncertain_area is not None:
        certain_index = (uncertain_area == 0)
        if certain_index.sum() > 0:
            ce_loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
            ce_loss = ce_loss[certain_index].mean()
            predict_soft = F.softmax(predict, dim=1)
            dice_loss_val = dice_loss(
                predict_soft[:, 1, :, :, :][certain_index],
                (target == 1)[certain_index]
            )
            return (ce_loss + dice_loss_val) / 2
        else:
            return torch.tensor(0.0, device=predict.device)
    return torch.tensor(0.0, device=predict.device)


# Load config
with open(args.config, 'r') as f:
    config = json.load(f)

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])
os.environ['PYTHONHASHSEED'] = str(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

data_loader = BuildDataLoader(config["dataset"], config["num_labels"], config["data_path"])
train_l_loader, train_u_loader, val_loader, test_loader = data_loader.build(supervised=False)

device = torch.device("cuda:{:d}".format(config["gpu"]) if torch.cuda.is_available() else "cpu")

model = UNet_kMaX(
    in_channels=1,
    is_batchnorm=True,
    n_classes=2,
    mun_pro=config["mun_pro"],
    num_queries=16,
    query_dim=64
).to(device)

op_module = opmoudle().to(device)

print(f"Loading SAM-Med 3D via medim from {args.sam_ckpt} ...")
sam_model = medim.create_model(
    "SAM-Med3D",
    pretrained=True,
    checkpoint_path=args.sam_ckpt
).to(device)

for param in sam_model.parameters():
    param.requires_grad = False
sam_model.eval()

bc_criterion = BoundaryContrastiveLoss(
    temperature=0.07,
    lambda_soft=1.0,
    lambda_global=0.5,
    lambda_hard=0.5,
    feature_dim=config["mun_pro"],
    query_dim=64
).to(device)

lambda_bc = 0.1
lambda_sam = 0.1

total_epoch = 200
optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)

optimizer_op = optim.SGD(op_module.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, nesterov=True)
scheduler_op = PolyLR(optimizer_op, total_epoch, power=0.9)

ema = EMA(model, 0.99)

train_epoch = len(train_l_loader)
avg_cost = np.zeros((total_epoch, 10))
iteration = 0

current_index = -1
best_dice = 0.0

model_dir = os.path.join(config["snapshot_path"], "model")
os.makedirs(model_dir, exist_ok=True)

save_best = os.path.join(model_dir, "model_best.pth")
save_best_op = os.path.join(model_dir, "model_op_best.pth")
save_last = os.path.join(model_dir, "model_last.pth")
save_last_op = os.path.join(model_dir, "model_op_last.pth")

history = {
    'epoch': [],
    'train_loss_total': [],
    'train_loss_sup': [],
    'train_loss_unsup': [],
    'train_loss_op': [],
    'train_loss_bc': [],
    'train_loss_sam': [],
    'val_dice': [],
    'val_jaccard': [],
    'val_hd95': [],
    'val_asd': []
}


# ==================== Training Loop ====================

for index in range(current_index + 1, total_epoch):
    cost = np.zeros(6)  # [Sup, Unsup, CPS, OP, BC, SAM]

    train_l_dataset = iter(train_l_loader)
    train_u_dataset = iter(train_u_loader)

    model.train()
    op_module.train()
    ema.model.train()

    for i in range(train_epoch):
        train_l_data, train_l_label, _ = train_l_dataset.__next__()
        train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device)

        train_u_data, train_u_label, name = train_u_dataset.__next__()
        train_u_data, train_u_label = train_u_data.to(device), train_u_label.to(device)

        optimizer.zero_grad()
        optimizer_op.zero_grad()

        # --- Generate Pseudo-labels (Student/EMA) ---
        with torch.no_grad():
            uncertain_area = None

            op_module.eval()
            pred_u_t, _, mm_map_1, query_t, class_logits_t = ema.model(train_u_data, mm=True)

            op_feature_1 = op_module(pred_u_t, mm_map_1)

            pseudo_logits, pseudo_labels = torch.max(torch.softmax(op_feature_1, dim=1), dim=1)

            if config.get("use_dynamic_threshold", False):
                output_soft = F.softmax(op_feature_1, dim=1).detach().cpu()
                pseudo_label_bg = (output_soft[:, 0, :, :, :] > dynamic_threshold_bg).long()
                pseudo_label_fg = (output_soft[:, 1, :, :, :] > dynamic_threshold_fg).long()
                uncertain_area = (pseudo_label_bg + pseudo_label_fg) == 0
                uncertain_area = uncertain_area.to(device)

            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                data_loader.crop_size, data_loader.scale_size, apply_augmentation=False)

            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=config["apply_aug"])

            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)

            train_u_aug_data = train_u_aug_data.float()
            train_u_aug_label = train_u_aug_label.long()
            train_u_aug_logits = train_u_aug_logits.float()

        # --- Forward Pass (Student) ---
        out_l = model(train_l_data, mm=True)
        pred_l, rep_l, mm_l, *others = out_l

        out_u = model(train_u_aug_data.float(), mm=True)
        pred_u, rep_u, mm_u, current_query, class_logits = out_u

        # --- Supervised Loss ---
        sup_ce_loss2 = F.cross_entropy(pred_l, train_l_label, ignore_index=-1).mean()
        sup_dice_loss2 = dice_loss(F.softmax(pred_l, dim=1)[:, 1, ...], train_l_label == 1).mean()

        sup_ce_loss3 = F.cross_entropy(mm_l, train_l_label, ignore_index=-1).mean()
        sup_dice_loss3 = dice_loss(F.softmax(mm_l, dim=1)[:, 1, ...], train_l_label == 1).mean()

        sup_loss = 0.5 * (sup_ce_loss2 + sup_dice_loss2 + sup_ce_loss3 + sup_dice_loss3)

        # --- Unsupervised Loss (Student Consistency) ---
        unsup_loss = torch.tensor(0.0, device=device)
        if config.get("use_dynamic_threshold", False) and (uncertain_area is not None):
            unsup_loss = compute_unsupervised_loss_with_uncertainty(
                pred_u, train_u_aug_label, train_u_aug_logits,
                config["strong_threshold"], uncertain_area
            )

        # --- Boundary Contrastive Loss ---
        D_rep, H_rep, W_rep = rep_l.shape[2:]
        y_l_small = F.interpolate(train_l_label.float().unsqueeze(1), size=(D_rep, H_rep, W_rep), mode='nearest').squeeze(1).long()
        seg_l_small = F.interpolate(pred_l, size=(D_rep, H_rep, W_rep), mode='trilinear', align_corners=True)
        seg_u_small = F.interpolate(pred_u, size=(D_rep, H_rep, W_rep), mode='trilinear', align_corners=True)

        loss_bc = bc_criterion(
            seg_l=seg_l_small, feat_l=rep_l, y_l=y_l_small,
            seg_u=seg_u_small, feat_u=rep_u,
            current_query=current_query, kmax_class_logits=class_logits
        )

        # ==================== SAM-Med 3D Auxiliary Loss ====================
        loss_sam_aux = torch.tensor(0.0, device=device)

        if index > 5:
            with torch.no_grad():
                out_ema_aug = ema.model(train_u_aug_data.float(), mm=False)
                teacher_prob = F.softmax(out_ema_aug[0], dim=1)
                teacher_mask = torch.max(teacher_prob, dim=1)[1]

                sam_input_img = resize_volume(train_u_aug_data.float(), (128, 128, 128), mode='trilinear')
                teacher_mask_128 = resize_volume(teacher_mask.float().unsqueeze(1), (128, 128, 128), mode='nearest').squeeze(1)

                points_coords, points_labels = get_points_from_mask(teacher_mask_128, num_points=10)
                valid_batch_indices = [idx for idx in range(len(points_coords)) if points_coords[idx].sum() > 0]

                if len(valid_batch_indices) > 0:
                    image_embeddings = sam_model.image_encoder(sam_input_img)

                    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                        points=[points_coords, points_labels],
                        boxes=None,
                        masks=None,
                    )

                    low_res_masks, _ = sam_model.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=sam_model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                    )

                    sam_logits_128 = F.interpolate(
                        low_res_masks,
                        size=(128, 128, 128),
                        mode='trilinear',
                        align_corners=False
                    )

                    sam_logits_96 = F.interpolate(
                        sam_logits_128,
                        size=(96, 96, 96),
                        mode='trilinear',
                        align_corners=False
                    )

                    sam_prob = torch.sigmoid(sam_logits_96)
                    student_fg_prob = F.softmax(pred_u, dim=1)[:, 1:2, ...]

                    student_valid = student_fg_prob[valid_batch_indices]
                    sam_valid = sam_prob[valid_batch_indices]

                    loss_sam_aux = F.mse_loss(student_valid, sam_valid)

        # --- Total Loss Update ---
        loss = sup_loss + 0.5 * unsup_loss + lambda_bc * loss_bc + lambda_sam * loss_sam_aux
        loss.backward()
        optimizer.step()
        optimizer_op.step()

        ema.update(model)

        # --- DuCiCS Update Parameters ---
        if config.get("use_dynamic_threshold", False):
            try:
                current_unlabeled_soft = F.softmax(pred_u, dim=1).detach().cpu()
                index_gt_fg = (pseudo_labels == 1).cpu()
                mean_prob_fg = current_unlabeled_soft[:, 1][index_gt_fg].mean().item() if index_gt_fg.sum() > 0 else 0.5
                dynamic_threshold_fg = dynamic_threshold_fg * alpha_ema + (1 - alpha_ema) * mean_prob_fg
                dynamic_threshold_fg = max(0.5, min(0.95, dynamic_threshold_fg))
            except:
                pass

        # --- Op Module Supervision ---
        model.eval()
        op_module.train()
        with torch.no_grad():
            out_l2 = model(train_l_data, mm=True)
            Pred_ll, _, Mm_map_ll, _, _ = out_l2
            Pred_ll = Pred_ll.detach()
            Mm_map_ll = Mm_map_ll.detach()

        Op_feature = op_module(Pred_ll, Mm_map_ll)
        op_loss = 0.5 * (
            F.cross_entropy(Op_feature, train_l_label, ignore_index=-1) +
            dice_loss(F.softmax(Op_feature, dim=1)[:, 1, ...], train_l_label == 1)
        )

        optimizer_op.zero_grad()
        op_loss.backward()
        optimizer_op.step()

        model.train()

        # --- Logging ---
        cost[0] = sup_loss.item()
        cost[1] = unsup_loss.item()
        cost[3] = op_loss.item()
        cost[4] = loss_bc.item()
        cost[5] = loss_sam_aux.item()

        avg_cost[index, :6] += cost / train_epoch
        iteration += 1

    scheduler.step()
    scheduler_op.step()

    train_loss_sup = avg_cost[index, 0]
    train_loss_unsup = avg_cost[index, 1]
    train_loss_op = avg_cost[index, 3]
    train_loss_bc = avg_cost[index, 4]
    train_loss_sam = avg_cost[index, 5]
    train_loss_total = (train_loss_sup + 0.5 * train_loss_unsup +
                        lambda_bc * train_loss_bc + lambda_sam * train_loss_sam)

    do_validate = (index == 0) or ((index + 1) % VAL_INTERVAL == 0) or (index == total_epoch - 1)

    if do_validate:
        with torch.no_grad():
            metric_record = 0.0
            ema.model.eval()
            op_module.eval()

            dataloader = iter(val_loader)
            tbar = tqdm(range(len(val_loader)), ncols=135, desc=f"Val Epoch {index+1}/{total_epoch}")
            for batch_idx in tbar:
                x, y, _ = next(dataloader)
                y = y.squeeze(0)

                y_tilde, y_hat = test_single_case(
                    ema.model, op_module, x,
                    stride_xy=16, stride_z=16,
                    patch_size=(96, 96, 96),
                    num_classes=data_loader.num_segments
                )

                if np.sum(y_tilde) == 0:
                    single_metric = (0, 0, 0, 0)
                else:
                    single_metric = calculate_metric_percase(np.array(y_tilde), np.array(y[:]))

                metric_record += np.asarray(single_metric)

            metric_record = metric_record / len(val_loader)

        dice, jc, hd95, asd = [float(x) for x in metric_record[:4]]

        print(
            f"[VAL] Epoch {index+1}/{total_epoch} | "
            f"Dice={dice:.4f} | Jaccard={jc:.4f} | HD95={hd95:.4f} | ASD={asd:.4f} | "
            f"BestDice={best_dice:.4f}"
        )

        history['epoch'].append(index + 1)
        history['train_loss_total'].append(train_loss_total)
        history['train_loss_sup'].append(train_loss_sup)
        history['train_loss_unsup'].append(train_loss_unsup)
        history['train_loss_op'].append(train_loss_op)
        history['train_loss_bc'].append(train_loss_bc)
        history['train_loss_sam'].append(train_loss_sam)
        history['val_dice'].append(dice)
        history['val_jaccard'].append(jc)
        history['val_hd95'].append(hd95)
        history['val_asd'].append(asd)

        if dice >= best_dice:
            best_dice = dice
            torch.save(ema.model.state_dict(), save_best)
            torch.save(op_module.state_dict(), save_best_op)
            print(f"New best saved: Dice={best_dice:.4f} -> {save_best}")

        torch.save(ema.model.state_dict(), save_last)
        torch.save(op_module.state_dict(), save_last_op)

        ema.model.train()

print("Training Finished.")

csv_path = os.path.join(config["snapshot_path"], "training_history.csv")
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = [
        'epoch', 'train_loss_total', 'train_loss_sup', 'train_loss_unsup',
        'train_loss_op', 'train_loss_bc', 'train_loss_sam',
        'val_dice', 'val_jaccard', 'val_hd95', 'val_asd'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(history['epoch'])):
        row = {key: history[key][i] for key in fieldnames}
        writer.writerow(row)

print(f"Training history saved to {csv_path}")
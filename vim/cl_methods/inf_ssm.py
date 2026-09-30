import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import ModelEma

from losses import DistillationLoss
import utils


def solve_P(A1, C1, A2, C2):

    A_diag = torch.cat([A1, A2], dim=1)  # (2D,)

    denom = 1.0 - A_diag.unsqueeze(2) * A_diag.unsqueeze(1)

    C_combined = torch.cat([C1, C2], dim=1)  # (B, N, 2D)

    Q = C_combined.unsqueeze(2) * C_combined.unsqueeze(1)


    P = Q / denom

    # X is (B, 2D, 2D)
    return P


def trace(mat):
    return torch.einsum('bii->b', mat)


def solve_distance_obs(A1, C1, A2, C2):

    if torch.allclose(A1, A2) and torch.allclose(C1, C2):
        return torch.tensor(0., device=A1.device, dtype=A1.dtype)


    N = A1.shape[1]

    P = solve_P(A1, C1, A2, C2)


    diff = 1 - trace(P[:, :N, N:] @ P[:, N:, :N]) / (trace(P[:, :N, :N]) * trace(P[:, N:, N:]))

    dist = torch.mean(diff)

    return dist


def reg_loss(args, A1, C1, A2, C2, mode='inf_ssm'):
    if mode == 'mse':
        loss = args.reg_lambda * (torch.nn.functional.mse_loss(A1, A2) + torch.nn.functional.mse_loss(C1, C2))
    elif mode == 'inf_ssm':
        loss = args.reg_lambda * solve_distance_obs(A1, C1, A2, C2)
    else:
        raise ValueError(f"Unknown reg mode {mode}")

    return loss


def get_reg_loss(args, model, old_model, mode='inf_ssm'):

    loss = 0
    count = 0
    num_layers = len(model.layers)
    for layer in range(num_layers):
        if model.layers[layer].mixer.bimamba_type == 'v2_state_ac':
            loss += reg_loss(args, model.layers[layer].mixer.A_s_r, model.layers[layer].mixer.C_s_r, old_model.layers[layer].mixer.A_s_r, old_model.layers[layer].mixer.C_s_r, mode=mode)
            loss += reg_loss(args, model.layers[layer].mixer.A_b_s_r, model.layers[layer].mixer.C_b_s_r, old_model.layers[layer].mixer.A_b_s_r, old_model.layers[layer].mixer.C_b_s_r, mode=mode)
            count += 2

    return loss / count


def train_one_epoch_inf_ssm(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, old_net=None):

    task_idx = data_loader.dataset.cur_task

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('reg_loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    # debug
    count = 0
    correct = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with amp_autocast():
            outputs = model(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)

            penalty = None
            if old_net is not None:
                _ = old_net(samples, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)
                penalty = get_reg_loss(args, model, old_net, mode = args.reg_mode)

            pc = task_idx * args.n_classes_per_task
            ac = (task_idx + 1) * args.n_classes_per_task
            outputs = outputs[:, pc:ac]
            targets = targets[:, pc:ac] / targets[:, pc:ac].sum(dim=1, keepdim=True) # get back to sum = 1

            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets)
                loss = loss + 0.25 * criterion(outputs[1], targets)
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid())

            if penalty is not None:
                if torch.isnan(penalty):
                    print('nan at penalty')
                loss +=  penalty

        if args.if_nan2num:
            with amp_autocast():
                loss = torch.nan_to_num(loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            if args.if_continue_inf:
                optimizer.zero_grad()
                continue
            else:
                sys.exit(1)



        # this attribute is added by timm on one optimizer (adahessian)
        if isinstance(loss_scaler, timm.utils.NativeScaler):
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            if max_norm != None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)


        with torch.no_grad():
            correct += (outputs.detach().cpu().argmax(dim=1) == targets.detach().cpu().argmax(dim=1)).sum().item()
            count += samples.shape[0]

        metric_logger.update(loss=loss_value)
        metric_logger.update(reg_loss=penalty.detach().item() if penalty is not None else 0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc'].update(correct / count * 100)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

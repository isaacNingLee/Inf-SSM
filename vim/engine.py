# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

import timm
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import matplotlib.pyplot as plt
import os
# from sklearn.manifold import TSNE
import numpy as np


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def cl_evaluate(args, data_loader, model, device, amp_autocast, task_id, output_path, current = False):

    # switch to evaluation mode
    model.eval()

    result = {}

    start_task = 0 if not current else task_id

    if not current:
        print(f'Evaluate all tasks up to {task_id}')

    for task in range(start_task, task_id + 1):
        data_loader.dataset.set_task(task)

        result[f'class_{task}_acc'] = 0
        result[f'task_{task}_acc'] = 0 

        # disp_im, disp_lab, disp_out = [], [], []


        for images, target in data_loader:
            images = images.to(device)
            target = target.to(device)

            # compute output
            with amp_autocast():

                output = model(images, if_random_cls_token_position=args.if_random_cls_token_position, if_random_token_rank=args.if_random_token_rank)

                output = output[:, 0 : (task_id + 1) * data_loader.dataset.n_classes_per_task]
                # loss = criterion(output, target)

            cil_acc, til_acc = cl_accuracy(output, target, task, data_loader.dataset.n_classes_per_task)
            
            bs = images.shape[0]
            result[f'class_{task}_acc'] += cil_acc.item() * bs
            result[f'task_{task}_acc'] += til_acc.item() * bs

            # for i in range(6):
            #     disp_im.append(images[i])
            #     disp_lab.append(target[i])
            #     disp_out.append(output[i])


        # update the result
        result[f'class_{task}_acc'] /= len(data_loader.dataset)
        result[f'task_{task}_acc'] /= len(data_loader.dataset)
        

        # # # # visualize the random 36 images
        # rnd_idx = torch.randperm(len(disp_im))[:36]
        # images = torch.stack([disp_im[i] for i in rnd_idx])
        # target = torch.stack([disp_lab[i] for i in rnd_idx])
        # output = torch.stack([disp_out[i] for i in rnd_idx])
        # plt.figure(figsize=(10, 10))
        # for i in range(min(36, len(images))):
        #     plt.subplot(6, 6, i + 1)
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.grid(False)
        #     # unnormalize
        #     images[i] = images[i] * 0.5 + 0.5
        #     plt.imshow(images[i].cpu().numpy().transpose((1, 2, 0)).clip(0, 1))
        #     plt.xlabel(f'pred: {output[i].argmax()}, label: {target[i]}')

        # # makedir
        # if not os.path.exists(f'{output_path}/phase_{task_id}'):
        #     os.makedirs(f'{output_path}/phase_{task_id}')

        # plt.savefig(f'{output_path}/phase_{task_id}/task_{task}.png')
        # plt.close()
        
        if args.if_tsne and (task==0 or task==1 or task==2):
            if not os.path.exists(f'{output_path}/phase_{task_id}/task_{task}'):
                os.makedirs(f'{output_path}/phase_{task_id}/task_{task}')

            latent_tsne(data_loader, model, device, f'{output_path}/phase_{task_id}/task_{task}')

        
    if not current:
        # mean acc
        result['mean_class_acc'] = sum([result[f'class_{task}_acc'] for task in range(task_id + 1)]) / (task_id + 1)
        result['mean_task_acc'] = sum([result[f'task_{task}_acc'] for task in range(task_id + 1)]) / (task_id + 1)


    return result


def cl_accuracy(output, target, task_id, cls_per_task):
    # target is already per task

    # cil accuracy
    cil_pred = output.argmax(dim=1)
    cil_acc = (cil_pred == target).float().mean()

    # til accuracy
    til_pred = output[:, task_id * cls_per_task: (task_id + 1) * cls_per_task].argmax(dim=1) + task_id * cls_per_task
    til_acc = (til_pred == target).float().mean()

    return cil_acc, til_acc

def forgetting(accum_results, task_id):
    # calculate forgetting
    forgetting = 0
    for i in range(task_id):
        forgetting += max(accum_results[f'class_{i}_acc']) - accum_results[f'class_{i}_acc'][-1]

    return forgetting / task_id


def latent_tsne(data_loader, model, device,  output_path):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation
    model.eval()

    patch_out, layer50_out, layer100_out, h_out, logits_out, target_out = [], [], [], [], [], []
    

    for images, target in metric_logger.log_every(data_loader, 10, header):

        images = images.to(device)
        target = target.to(device)

        target_out.append(target.cpu())

        # compute output
        with torch.no_grad():
            _ , patch, layer50, layer100, h, logits = model(images, return_intermediate=True)

        patch_out.extend(patch)
        layer50_out.extend(layer50)
        layer100_out.extend(layer100)
        h_out.extend(h)
        logits_out.extend(logits)


        
    # visualize with tsne
    patch_out = torch.cat(patch_out, dim=0).cpu().numpy()
    layer50_out = torch.cat(layer50_out, dim=0).cpu().numpy()
    layer100_out = torch.cat(layer100_out, dim=0).cpu().numpy()
    h_out = torch.cat(h_out, dim=0).cpu().numpy()
    logits_out = torch.cat(logits_out, dim=0).cpu().numpy()
    target = torch.cat(target_out, dim=0).cpu().numpy()

    # save
    np.save(f'{output_path}/patch_embed.npy', patch_out)
    np.save(f'{output_path}/layer50.npy', layer50_out)
    np.save(f'{output_path}/layer100.npy', layer100_out)
    np.save(f'{output_path}/h.npy', h_out)
    np.save(f'{output_path}/logits.npy', logits_out)
    np.save(f'{output_path}/target.npy', target)


    # # patch
    # tsne = TSNE(n_components=2, random_state=0)
    # patch_out = tsne.fit_transform(patch_out.reshape(patch_out.shape[0], -1))
    # plt.figure(figsize=(6, 6))
    # scatter = plt.scatter(patch_out[:, 0], patch_out[:, 1], c=target, cmap=colors)
    # plt.colorbar(scatter, ticks=range(20), label='Category')
    # plt.title('patch_embed')
    # plt.savefig(f'{output_path}/patch_embed.png')
    # plt.close()

    # # layer50
    # tsne = TSNE(n_components=2, random_state=0)
    # layer50_out = tsne.fit_transform(layer50_out.reshape(layer50_out.shape[0], -1))
    # plt.figure(figsize=(6, 6))
    # scatter = plt.scatter(layer50_out[:, 0], layer50_out[:, 1], c=target, cmap=colors)
    # plt.colorbar(scatter, ticks=range(20), label='Category')
    # plt.title('layer50')
    # plt.savefig(f'{output_path}/layer50.png')
    # plt.close()

    # # layer100
    # tsne = TSNE(n_components=2, random_state=0)
    # layer100_out = tsne.fit_transform(layer100_out.reshape(layer100_out.shape[0], -1))
    # plt.figure(figsize=(6, 6))
    # scatter = plt.scatter(layer100_out[:, 0], layer100_out[:, 1], c=target, cmap=colors)
    # plt.colorbar(scatter, ticks=range(20), label='Category')
    # plt.title('layer100')
    # plt.savefig(f'{output_path}/layer100.png')
    # plt.close()

    # # logits
    # tsne = TSNE(n_components=2, random_state=0)
    # logits_out = tsne.fit_transform(logits_out.reshape(logits_out.shape[0], -1))
    # plt.figure(figsize=(6, 6))
    # scatter = plt.scatter(logits_out[:, 0], logits_out[:, 1], c=target, cmap=colors)
    # plt.colorbar(scatter, ticks=range(20), label='Category')
    # plt.title('logits')
    # plt.savefig(f'{output_path}/logits.png')
    # plt.close()
        


            
# -*- coding: utf-8 -*-

import os
import torch
import shutil
import numbers
import requests
import warnings
import numpy as np
from importlib import import_module
from src.eval.evaluate import Evaluator
import logging

logger = logging.getLogger()

MAX_INT = np.iinfo(np.int32).max
MIN_INT = np.iinfo(np.int32).min

evaluator = Evaluator()

def check_parameter(param, low=MIN_INT, high=MAX_INT, param_name='',
                    include_left=False, include_right=False):
    if not isinstance(param, (numbers.Integral, int, float)):
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))
        raise TypeError('{param_name} is set to {param} Not numerical'.format(
            param=param, param_name=param_name))
    if not isinstance(low, (numbers.Integral, int, float)):
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>low is set to {low}. Not numerical'.format(low=low))
        raise TypeError('low is set to {low}. Not numerical'.format(low=low))
    if not isinstance(high, (numbers.Integral, int, float)):
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>high is set to {high}. Not numerical'.format(
            high=high))
        raise TypeError('high is set to {high}. Not numerical'.format(
            high=high))
    # at least one of the bounds should be specified
    if low is MIN_INT and high is MAX_INT:
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>Neither low nor high bounds is undefined')
        raise ValueError('Neither low nor high bounds is undefined')
    # if wrong bound values are used
    if low > high:
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>Lower bound > Higher bound')
        raise ValueError('Lower bound > Higher bound')
    # value check under different bound conditions
    if (include_left and include_right) and (param < low or param > high):
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>{param_name} is set to {param}. Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))
    elif (include_left and not include_right) and (
            param < low or param >= high):
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>{param_name} is set to {param}. Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of [{low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    elif (not include_left and include_right) and (
            param <= low or param > high):
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>{param_name} is set to {param}. Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}].'.format(
                param=param, low=low, high=high, param_name=param_name))
    elif (not include_left and not include_right) and (
            param <= low or param >= high):
        logger.error('>>>src.utils.utility.py|>>>check_parameter()|>>>{param_name} is set to {param}. Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
        raise ValueError(
            '{param_name} is set to {param}. '
            'Not in the range of ({low}, {high}).'.format(
                param=param, low=low, high=high, param_name=param_name))
    else:
        return True
    
def validate_device(gpu_id):
    gpu_id = int(gpu_id)
    if gpu_id == -1:
        return 'cpu'
    if torch.cuda.is_available():
        check_parameter(gpu_id, 0, torch.cuda.device_count(),
                        param_name='gpu id', include_left=True,
                        include_right=False)
        device = 'cuda:{}'.format(gpu_id)
    else:
        if gpu_id != 'cpu':
            logger.warning(">>>src.utils.utility.py|>>>validate_device()|>>>The cuda is not available. Set to cpu.")
            warnings.warn('The cuda is not available. Set to cpu.')
        device = 'cpu'
    return device

def loggerInfo(suffix="",
           epoch=0,
           loss=0,
           score=None,
           target=None,
           time=None,
           verbose=0,
           train=True,
           deep=True,
           return_metrics=False):
    log_msg = ""
    if verbose > 0:
        if deep:
            if train:
                log_msg = f"{suffix} Epoch {epoch:04d}: "
            else:
                log_msg = f"{suffix} Test: "

            if isinstance(loss, tuple):
                log_msg += f"Loss I {loss[0]:.4f} | Loss O {loss[1]:.4f} | "
            else:
                log_msg += f"Loss {loss:.4f} | "
        if verbose > 1:
            if target is not None:
                auc = evaluator.eval_roc_auc(target, score)
                log_msg += f"AUC {auc:.4f}"
            if verbose > 2:
                if target is not None:
                    pos_size = target.nonzero().size(0)
                    rec = evaluator.eval_recall_at_k(target, score, pos_size)
                    pre = evaluator.eval_precision_at_k(target, score, pos_size)
                    ap = evaluator.eval_average_precision(target, score)
                    contamination = sum(target) / len(target)
                    threshold = np.percentile(score, 100 * (1 - contamination))
                    pred = (score > threshold).long()
                    f1 = evaluator.eval_f1(target, pred)
                    log_msg += f" | Recall {rec:.4f} | Precision {pre:.4f} | AP {ap:.4f} | F1 {f1:.4f}"
            if time is not None:
                log_msg += f" | Time {time:.2f}"
        # 只有当有内容时才记录日志
        if log_msg:
            logger.info(log_msg)
    if return_metrics and target is not None and verbose > 2:
        return auc, rec, pre, ap, f1
    return None
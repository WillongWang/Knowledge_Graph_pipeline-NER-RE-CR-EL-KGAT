# coding=utf-8

import random
import numpy as np
import torch

import os
import json
import pickle
import torch.nn as nn
from collections import OrderedDict
from pathlib import Path
import logging

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)

def save_numpy(data, file_path):
    '''
    保存成.npy文件
    :param data:
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.save(str(file_path),data)

def load_numpy(file_path):
    '''
    加载.npy文件
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.load(str(file_path))

def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data

def json_to_text(file_path,data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')

def save_model(model, model_path):
    """ 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_name:
    :param only_param:
    :return:
    """
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)

def load_model(model, model_path):
    '''
    加载模型
    :param model:
    :param model_name:
    :param model_path:
    :param only_param:
    :return:
    '''
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    state = states['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)
    return model


def summary(model, *inputs, batch_size=-1, show_input=True):
    '''
    打印模型结构信息
    :param model:
    :param inputs:
    :param batch_size:
    :param show_input:
    :return:
    Example:
        >>> print("model summary info: ")
        >>> for step,batch in enumerate(train_data):
        >>>     summary(self.model,*batch,show_input=True)
        >>>     break
    '''

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )

        total_params += summary[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        print(line_new)

    print("=======================================================================")
    print(f"Total params: {total_params:0,}")
    print(f"Trainable params: {trainable_params:0,}")
    print(f"Non-trainable params: {(total_params - trainable_params):0,}")
    print("-----------------------------------------------------------------------")
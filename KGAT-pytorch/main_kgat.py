import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import random
import logging
import argparse
from time import time

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from model.KGAT import KGAT
from utility.parser_kgat import *
from utility.log_helper import *
from utility.metrics import *
from utility.helper import *
from utility.loader_kgat import DataLoaderKGAT
from tqdm import tqdm

def train(args):
    # seed 随机种子设定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # logger设定
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU 设备信息设置
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoaderKGAT(args, logging)

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    # user ids
    user_ids = list(data.test_user_dict.keys())
    # 方便一个batch 一个batch进行评估
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    #
    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # construct model & optimizer
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations, user_pre_embed, item_pre_embed)
    if args.use_pretrain == 2:  # 如果使用已经训练过的模型，模型从文件中加载
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    # if n_gpu > 1:
    #     model = nn.parallel.DistributedDataParallel(model)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # move graph data to GPU： 注意这里DGL的graph数据使用GPU的方式
    train_graph = data.train_graph
    train_nodes = torch.LongTensor(train_graph.ndata['id'])
    train_edges = torch.LongTensor(train_graph.edata['type'])
    if use_cuda:
        train_nodes = train_nodes.to(device)
        train_edges = train_edges.to(device)
        train_graph = train_graph.to(device)
    train_graph.ndata['id'] = train_nodes
    train_graph.edata['type'] = train_edges

    test_graph = data.test_graph
    test_nodes = torch.LongTensor(test_graph.ndata['id'])
    test_edges = torch.LongTensor(test_graph.edata['type'])
    if use_cuda:
        test_nodes = test_nodes.to(device)
        test_edges = test_edges.to(device)
        test_graph = test_graph.to(device)
    test_graph.ndata['id'] = test_nodes
    test_graph.edata['type'] = test_edges

    # initialize metrics
    best_epoch = -1
    epoch_list = []
    precision_list = []
    recall_list = []
    ndcg_list = []

    # train model
    for epoch in tqdm(range(1, args.n_epoch + 1)):
        time0 = time()
        model.train()

        # update attention scores
        #   ： 每个三元组都计算一次attention score
        with torch.no_grad():
            att = model('calc_att', train_graph)
        train_graph.edata['att'] = att
        logging.info('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # train cf： 训练用户-物品交互
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict)
            if use_cuda:
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)
            cf_batch_loss = model(
                'calc_cf_loss',
                train_graph,
                cf_batch_user,
                cf_batch_pos_item,
                cf_batch_neg_item,
            )

            cf_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

            if (iter % args.cf_print_every) == 0:
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg: TransR
        time1 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1

        for iter in range(1, n_kg_batch + 1):
            time2 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(
                data.train_kg_dict
            )
            if use_cuda:
                kg_batch_head = kg_batch_head.to(device)
                kg_batch_relation = kg_batch_relation.to(device)
                kg_batch_pos_tail = kg_batch_pos_tail.to(device)
                kg_batch_neg_tail = kg_batch_neg_tail.to(device)
            kg_batch_loss = model('calc_kg_loss', kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail)

            kg_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            kg_total_loss += kg_batch_loss.item()

            if (iter % args.kg_print_every) == 0:
                logging.info('KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_kg_batch, time() - time2, kg_batch_loss.item(), kg_total_loss / iter))
        logging.info('KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_kg_batch, time() - time1, kg_total_loss / n_kg_batch))

        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0:
            time1 = time()
            _, precision, recall, ndcg = evaluate(
                model, train_graph,
                data.train_user_dict,
                data.test_user_dict,
                user_ids_batches,
                item_ids,
                args.K
            )
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg))

            epoch_list.append(epoch)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            best_recall, should_stop = early_stopping(recall_list, args.stopping_steps)

            if should_stop:
                break

            if recall_list.index(best_recall) == len(recall_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save model
    save_model(model, args.save_dir, epoch)

    # save metrics
    _, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
    logging.info('Final CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))

    epoch_list.append(epoch)
    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)

    metrics = pd.DataFrame([epoch_list, precision_list, recall_list, ndcg_list]).transpose()
    metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K)]
    metrics.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)


def evaluate(model, train_graph, train_user_dict, test_user_dict, user_ids_batches, item_ids, K):
    model.eval()

    with torch.no_grad():
        att = model.compute_attention(train_graph)
    train_graph.edata['att'] = att

    n_users = len(test_user_dict.keys())
    item_ids_batch = item_ids.cpu().numpy()

    cf_scores = []
    precision = []
    recall = []
    ndcg = []

    # 对测试user， 预测其购买的商品
    with torch.no_grad():
        for user_ids_batch in user_ids_batches:
            cf_scores_batch = model('predict', train_graph, user_ids_batch, item_ids)       # (n_batch_users, n_eval_items)

            cf_scores_batch = cf_scores_batch.cpu()
            user_ids_batch = user_ids_batch.cpu().numpy()
            precision_batch, recall_batch, ndcg_batch = calc_metrics_at_k(
                cf_scores_batch,
                train_user_dict,
                test_user_dict,
                user_ids_batch,
                item_ids_batch,
                K
            )

            cf_scores.append(cf_scores_batch.numpy())
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)

    cf_scores = np.concatenate(cf_scores, axis=0)
    precision_k = sum(np.concatenate(precision)) / n_users
    recall_k = sum(np.concatenate(recall)) / n_users
    ndcg_k = sum(np.concatenate(ndcg)) / n_users
    return cf_scores, precision_k, recall_k, ndcg_k


def predict(args):
    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoaderKGAT(args, logging)

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)
    # if n_gpu > 1:
    #     model = nn.parallel.DistributedDataParallel(model)

    # move graph data to GPU
    train_graph = data.train_graph
    train_nodes = torch.LongTensor(train_graph.ndata['id'])
    train_edges = torch.LongTensor(train_graph.edata['type'])
    if use_cuda:
        train_nodes = train_nodes.to(device)
        train_edges = train_edges.to(device)
        train_graph = train_graph.to(device)
    train_graph.ndata['id'] = train_nodes
    train_graph.edata['type'] = train_edges

    test_graph = data.test_graph
    test_nodes = torch.LongTensor(test_graph.ndata['id'])
    test_edges = torch.LongTensor(test_graph.edata['type'])
    if use_cuda:
        test_nodes = test_nodes.to(device)
        test_edges = test_edges.to(device)
        test_graph = test_graph.to(device)
    test_graph.ndata['id'] = test_nodes
    test_graph.edata['type'] = test_edges

    # predict
    cf_scores, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))



if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)
    # predict(args)







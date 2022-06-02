import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils_test import encoding_test
from utils.accuracy import compute_accuracy_for_test
from utils.tool import get_ei2, to_directed, reverse, reduce_loops
from GNN.model import WLGNN_d, WLGNN_d_hy

import time

# parse the argument
def tst_params(directed, dataset, model_dir, model_name, num_runs, printv=True, **kwargs):

    device = torch.device('cuda')

    # encoding the data

    train_dataset = dataset
    test_dataset = dataset
    model_ver = 'dir' if directed else 'undir'

    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    false_positive_rate_list = []
    false_negative_rate_list = []
    roc_auc_list = []
    auc_pr_list = []
    r_mr_list = []
    r_mrr_list = []
    r_hits1_list = []
    r_hits3_list = []
    r_hits10_list = []

    for run in range(num_runs):

        ei1, ei2, adj, features, labels, masks, num_node, num_type, num_relation, constants, relations, types, pairs, hits_true, r_hits_candidates = encoding_test(run, train_dataset, test_dataset)
        #labels = to_directed(labels)
        #masks = to_directed(masks)

        ei1 = torch.cat([ei1, ei1[[1, 0]]], -1)
        e2 = get_ei2(num_node, ei1, ei1)
        edge2, edge2_r = reverse(e2, ei1.shape[1]//2)
        edge2 = reduce_loops(edge2)
        edge2_r = reduce_loops(edge2_r)
        del e2

        if directed:
            ei2 = None
            edge2 = edge2.to(device)
            edge2_r = edge2_r.to(device)
            features = to_directed(features)
        else:
            edge2 = None
            edge2_r = None
            ei2 = ei2.to(device)

        features = features.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # define the Model

        if kwargs["use_hy"]:
            model = WLGNN_d_hy(n_feat=features.shape[1],
                            latent_size=kwargs["latent_size"],
                            depth2=2,
                            dropout=kwargs["dropout"],
                            nclass=labels.shape[1],
                            directed=directed)
        else:
            model = WLGNN_d(n_feat=features.shape[1],
                            dim=kwargs["dim"],
                            depth2=1,
                            ln1=kwargs["ln"],
                            dp1=kwargs["dp1"],
                            act1=kwargs["act"],
                            nclass=labels.shape[1],
                            directed=directed)

        model = model.to(device)
        model_path = "models/{}/{}".format(model_ver, model_dir)

        # load the model
        model.load_state_dict(torch.load("{}/{}.pkl".format(model_path, model_name)))

        """test the model"""

        model.eval()

        output_test = model(features, edge2, edge2_r, ei2)

        output_test_accuracy = output_test.clone()
        if printv:
            f_test = open("predictions_{}.txt".format(dataset), "w+")
            for p_id in range(len(pairs)):
                for r_id in range(num_type + 2 * num_relation):
                    if masks[p_id][r_id] == 1:
                        if r_id < num_type:
                            f_test.write(constants[pairs[p_id][0]])
                            f_test.write("\t")
                            f_test.write("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>")
                            f_test.write("\t")
                            f_test.write(types[r_id])
                        elif r_id >= num_type and r_id < num_type + num_relation:
                            f_test.write(constants[pairs[p_id][0]])
                            f_test.write("\t")
                            f_test.write(relations[r_id - num_type])
                            f_test.write("\t")
                            f_test.write(constants[pairs[p_id][1]])

                        else:
                            f_test.write(constants[pairs[p_id][1]])
                            f_test.write("\t")
                            f_test.write(relations[r_id - num_type - num_relation])
                            f_test.write("\t")
                            f_test.write(constants[pairs[p_id][0]])
                        f_test.write("\t")
                        f_test.write(str(output_test_accuracy[p_id][r_id].item()))
                        f_test.write("\t")
                        f_test.write(str(labels[p_id][r_id].item()))
                        f_test.write("\n")
            f_test.close()
            print("Predicted triples saved in file predictions_{}.txt".format(dataset))
        output_test = torch.mul(output_test, masks)

        loss = nn.BCELoss()
        loss_test = loss(output_test, labels)

        score_threshold = 0.5

        acc_test, precision_test, recall_test, f1_test, false_positive_rate_test, false_negative_rate_test, \
        roc_auc_test, auc_pr_test, r_mr_test, r_mrr_test, r_hits1_test, r_hits3_test, r_hits10_test = compute_accuracy_for_test(
            output_test_accuracy.cpu(), labels.cpu(), masks.cpu(), score_threshold, num_relation, num_type, hits_true, r_hits_candidates)

        acc_list.append(acc_test.item())
        precision_list.append(precision_test.item())
        recall_list.append(recall_test.item())
        f1_list.append(f1_test.item())
        false_positive_rate_list.append(false_positive_rate_test.item())
        false_negative_rate_list.append(false_negative_rate_test.item())
        roc_auc_list.append(roc_auc_test)
        auc_pr_list.append(auc_pr_test)

        r_mr_list.append(r_mr_test)
        r_mrr_list.append(r_mrr_test)
        r_hits1_list.append(r_hits1_test)
        r_hits3_list.append(r_hits3_test)
        r_hits10_list.append(r_hits10_test)


    print('------------Classification-------')

    print('accuracy: {:.4f}, var:{:.4f}\n'.format(np.mean(acc_list), np.var(acc_list)),
          'precision: {:.4f}, var:{:.4f}\n'.format(np.mean(precision_list), np.var(precision_list)),
          'recall: {:.4f}, var:{:.4f}\n'.format(np.mean(recall_list), np.var(recall_list)),
          'f1: {:.4f}, var:{:.4f}\n'.format(np.mean(f1_list), np.var(f1_list)),
          'false_positive_rate: {:.4f}, var:{:.4f}\n'.format(np.mean(false_positive_rate_list),
                                                             np.var(false_positive_rate_list)),
          'false_negative_rate: {:.4f}, var:{:.4f}\n'.format(np.mean(false_negative_rate_list),
                                                             np.var(false_negative_rate_list)),
          'roc_auc: {:.4f}, var:{:.4f}\n'.format(np.mean(roc_auc_list), np.var(roc_auc_list)),
          'auc_pr: {:.4f}, var:{:.4f}\n'.format(np.mean(auc_pr_list), np.var(auc_pr_list)))

    print('------------Ranking--------------')

    print(
        'r-MR: {:.4f}, var:{:.4f}\n'.format(np.mean(r_mr_list), np.var(r_mr_list)),
        'r-MRR: {:.4f}, var:{:.4f}\n'.format(np.mean(r_mrr_list), np.var(r_mrr_list)),
        'r-HITS@1: {:.4f}, var:{:.4f}\n'.format(np.mean(r_hits1_list), np.var(r_hits1_list)),
        'r-HITS@3: {:.4f}, var:{:.4f}\n'.format(np.mean(r_hits3_list), np.var(r_hits3_list)),
        'r-HITS@10: {:.4f}, var:{:.4f}\n'.format(np.mean(r_hits10_list), np.var(r_hits10_list)),
    )
    acc, auc, r3 = np.mean(acc_list), np.mean(roc_auc_list), np.mean(r_hits3_list)

    return acc + auc + 0.8 * r3
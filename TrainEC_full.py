__author__ = 'v-penlu'

import EmoClassify_full as ECF
import time
import cPickle
import os
import numpy as np
import multiprocessing
from collections import defaultdict
from random import shuffle


def Load_Feed(csv_dir, pair, params, mode=0):
    sample = Load_pair(csv_dir, pair)
    if mode == 0:
        grads, loss, cos = ECF.Feed_forward_backward(params, sample[0], sample[1],
                                                     sample[2])
        return grads, loss, cos
    else:
        loss, cos = ECF.Feed_forward_backward(params, sample[0], sample[1], sample[2],
                                              mode)
        return loss, cos


def Train_multiprocess(params, grad_acc, train_list, valid_list, csv_dir,
                       save_dir="save_params",
                       epoches=8, lr=0.1, cl=0, lr_halve_times=5, mode="ada",
                       momentum=0.95,
                       mini_batch=100, process_num=20):
    multiprocessing.freeze_support()
    train_epoch = 1
    pp_span = 2
    best_epoch_loss = 1e9
    ht = 0
    if mode in ["ada", "adaautocoor"]:
        lr_halve_times = 0
    while train_epoch <= epoches:
        print "Epoch %d, shuffling samples" % train_epoch
        shuffle(train_list)  # shuffle samples
        print "Start training epoch %d" % train_epoch
        t0 = time.time()
        pp_loss = 0
        pos_loss = 0
        pos_cnt = 0
        pos_cos = 0
        neg_cos = 0
        for i in range(len(train_list) / mini_batch):
            grads = defaultdict(list)
            pool = multiprocessing.Pool(processes=process_num)
            results = []
            for j in range(mini_batch):
                pair = train_list[i * mini_batch + j]
                results.append(pool.apply_async(Load_Feed, (csv_dir, pair,
                                                            params)))
            pool.close()
            pool.join()
            for j, ret in enumerate(results):
                p_grads, loss, cos = ret.get()
                pp_loss += loss
                is_pos = train_list[i * mini_batch + j][-1]
                if is_pos:
                    pos_loss += loss
                    pos_cnt += 1
                    pos_cos += cos
                else:
                    neg_cos += cos
                for k in p_grads.keys():
                    if k in grads:
                        if isinstance(grads[k], list):
                            for idx_g in range(len(grads[k])):
                                grads[k][idx_g] += p_grads[k][idx_g]
                        else:
                            grads[k] += p_grads[k]
                    else:
                        grads[k] = p_grads[k]
            for k in grads.keys():
                if isinstance(grads[k], list):
                    for idx_g in range(len(grads[k])):
                        grads[k][idx_g] /= mini_batch
                else:
                    grads[k] /= mini_batch
            ECF.All_params_SGD(params, grads, lr, cl, mode=mode,
                               grad_acc=grad_acc, momentum=momentum)
            if i % pp_span == 1:
                t1 = time.time()
                neg_loss = pp_loss - pos_loss
                neg_cnt = pp_span * mini_batch - pos_cnt
                print "    Average loss: %.6f, Pos loss: %.6f, Neg loss: %.6f" % \
                      (pp_loss / float(pp_span * mini_batch), pos_loss / float(pos_cnt),
                       neg_loss / float(neg_cnt))
                print "                            Pos cos:  %.6f, Neg cos:  %.6f" % \
                      (pos_cos / float(pos_cnt), neg_cos / float(neg_cnt))
                print "    samples per sec %.2f" % (pp_span * mini_batch /
                                                    float(t1 - t0))
                t0 = t1
                pp_loss = 0
                pos_loss = 0
                pos_cnt = 0
                pos_cos = 0
                neg_cos = 0
        rest_num = len(train_list) - (len(train_list) / mini_batch * mini_batch)
        if rest_num > 0:
            pool = multiprocessing.Pool(processes=process_num)
            results = []
            grads = defaultdict(list)
            for i in range((len(train_list) / mini_batch * mini_batch),
                           len(train_list)):
                pair = train_list[i]
                results.append(pool.apply_async(Load_Feed, (csv_dir, pair,
                                                            params)))
            pool.close()
            pool.join()
            for ret in results:
                p_grads, _, _ = ret.get()
                for k in p_grads.keys():
                    if k in grads:
                        if isinstance(grads[k], list):
                            for idx_g in range(len(grads[k])):
                                grads[k][idx_g] += p_grads[k][idx_g]
                        else:
                            grads[k] += p_grads[k]
                    else:
                        grads[k] = p_grads[k]
            for k in grads.keys():
                if isinstance(grads[k], list):
                    for idx_g in range(len(grads[k])):
                        grads[k][idx_g] /= rest_num
                else:
                    grads[k] /= rest_num
            ECF.All_params_SGD(params, grads, lr, cl, mode=mode,
                               grad_acc=grad_acc, momentum=momentum)
        valid_loss, pos_loss, pos_cos, neg_loss, neg_cos = \
            Calculate_loss_multiprocess(params, valid_list, csv_dir,
                                        mini_batch, process_num)
        print "Epoch %d      Average valid loss: %.6f, Pos loss: %.6f, Neg loss: %.6f" % \
              (train_epoch, valid_loss, pos_loss, neg_loss)
        print "                                             Pos cos:  %.6f, Neg cos:  %.6f" % \
              (pos_cos, neg_cos)
        Save_params(params, save_dir, train_epoch)
        if valid_loss > best_epoch_loss and ht < lr_halve_times:
            lr /= 2
            print "\t\tHalve learning rates to %.6f" % lr
            ht += 1
        else:
            best_epoch_loss = valid_loss
        train_epoch += 1


def Save_params(params, dir, epoch):
    if not os.path.exists(dir):
        os.mkdir(dir)
    save_name = os.path.join(dir, ("params_epoch%d.pkl" % epoch))
    f = open(save_name, "wb")
    cPickle.dump(params, f)
    f.close()

def Calculate_loss_multiprocess(params, valid_list, csv_dir, mini_batch,
                                process_num):
    total_loss = 0
    pos_loss = 0
    pos_cnt = 0
    pos_cos = 0
    neg_cos = 0
    for i in range(len(valid_list) / mini_batch):
        results = []
        pool = multiprocessing.Pool(processes=process_num)
        for j in range(mini_batch):
            pair = valid_list[i * mini_batch + j]
            results.append(pool.apply_async(Load_Feed, (csv_dir, pair, params,
                                                        1)))  # mode=1
        pool.close()
        pool.join()
        for j, loss in enumerate(results):
            loss, cos = loss.get()
            total_loss += loss
            is_pos = valid_list[i * mini_batch + j][-1]
            if is_pos:
                pos_loss += loss
                pos_cnt += 1
                pos_cos += cos
            else:
                neg_cos += cos
    rest_num = len(valid_list) - len(valid_list) / mini_batch * mini_batch
    if rest_num > 0:
        pool = multiprocessing.Pool(processes=process_num)
        results = []
        for i in range(len(valid_list) / mini_batch * mini_batch,
                       len(valid_list)):
            pair = valid_list[i]
            results.append(pool.apply_async(Load_Feed, (csv_dir, pair, params,
                                                        1)))  # mode=1
        pool.close()
        pool.join()
        for i, loss in enumerate(results):
            loss, cos = loss.get()
            total_loss += loss
            is_pos = valid_list[len(valid_list) / mini_batch * mini_batch + i][-1]
            if is_pos:
                pos_loss += loss
                pos_cnt += 1
                pos_cos += cos
            else:
                neg_cos += cos
    return (total_loss / len(valid_list), pos_loss / float(pos_cnt), pos_cos / float(pos_cnt),
            (total_loss - pos_loss) / float(len(valid_list) - pos_cnt),
            neg_cos / float(len(valid_list) - pos_cnt))


def Load_pair(csv_dir, pair):
    is_pos = pair[-1]
    csv_1 = os.path.join(csv_dir, ("Session%d" % pair[0][0]), pair[0][1],
                         ("%s.csv" % pair[0][2]))
    csv_2 = os.path.join(csv_dir, ("Session%d" % pair[1][0]), pair[1][1],
                         ("%s.csv" % pair[1][2]))
    arr_1 = np.loadtxt(csv_1, delimiter=";", skiprows=1)
    arr_2 = np.loadtxt(csv_2, delimiter=";", skiprows=1)
    dim = arr_1.shape[1]
    sample_1 = [arr_1[i].reshape(dim, 1) for i in range(arr_1.shape[0])]
    sample_2 = [arr_2[i].reshape(dim, 1) for i in range(arr_2.shape[0])]
    return [sample_1, sample_2, is_pos]


if __name__ == "__main__":
    pkl_name = r"train_valid_list_5emo1.pkl"
    csv_dir = r"D:\IEMOCAP_full_release\emobase"
    save_dir = r"pool3_max"     # pool_len = 3, max_pooling
    # pkl_name = r"train_valid_list_5emo1.pkl"
    # csv_dir = r"/home/lau/IEMOCAP_full_release/emobase"
    # save_dir = r"hidden-100-100_kmax-3_full-100"     # pool_len = 3, max_pooling

    mode = "adaautocoor"
    learn_rate = 0.05
    cln = 0
    momentum = 0.90
    epoches = 100
    lr_ht = 0
    batch_size = 50
    process_num = 5

    kmax_pool = [3]
    lmax_pool = [3, False]
    # act = "tanh"
    act = "relu"

    pkl = open(pkl_name, "rb")
    train_pairs = cPickle.load(pkl)
    valid_pairs = cPickle.load(pkl)
    pkl.close()
    # hidden_size_list = [50, 50, 50]
    hidden_size_list = [100, 100]
    in_dim = 30     # consistent with csv column
    out_dim = 100
    params, grad_acc = ECF.Construct_net(hidden_size_list, in_dim, out_dim, kmax_pool,
                                         act, sgd_mode=mode)
    # params_n = "save_params\hidden-100-100-100_kmax-3_epoch-6.pkl"
    # params_f = open(params_n, "rb")
    # params = cPickle.load(params_f)
    # params_f.close()
    print "Start Training"
    Train_multiprocess(params, grad_acc, train_pairs, valid_pairs, csv_dir, save_dir,
                       lr=learn_rate, epoches=epoches, lr_halve_times=lr_ht,
                       cl=cln, mode=mode, momentum=momentum,
                       mini_batch=batch_size, process_num=process_num)

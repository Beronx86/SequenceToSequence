__author__ = 'v-penlu'
import EmoClassify as EC
import time
import cPickle
import os
import numpy as np
import multiprocessing
from collections import defaultdict
from random import shuffle


def Train(params, train_list, valid_list, csv_dir, pool_len=3,
          average=False, epochs=8, lr=0.1, cl=5,
          lr_halve_times=5):
    """
    :param params:
    :param samples: [seq_1, seq_2, is_positive]
    :param epochs:
    :param lr_decay:
    :param lr:
    :return:
    """
    trained_epoch = 1
    pp_span = 10
    best_epoch_loss = 1e9
    ht = 0
    while trained_epoch <= epochs:
        trained_samples = 0
        pp_loss = 0
        t0 = time.time()
        for pair in train_list:
            sample = Load_pair(csv_dir, pair)
            grads, loss = EC.Feed_forward_backward(params, sample[0], sample[1],
                                                   sample[2], pool_len, average)
            EC.All_params_SGD(params, grads, lr, cl)
            pp_loss += loss
            trained_samples += 1
            if trained_samples % pp_span == 0:
                t1 = time.time()
                print "Average loss: %.5f" % (pp_loss / float(pp_span)),
                print "\tsample per sec %.2f" % (pp_span / float(t1 - t0))
                t0 = t1
                pp_loss = 0
        valid_loss = Calculate_loss(params, valid_list, csv_dir, pool_len,
                                    average)
        print "Epoch %d\tAverage valid loss: %.5f" % (trained_epoch, valid_loss)
        Save_params(params, "save_params", trained_epoch)
        if valid_loss > best_epoch_loss and ht < lr_halve_times:
            lr /= 2
            print "Halve learning rate to %.6f" % lr
            ht += 1
        else:
            best_epoch_loss = valid_loss
        trained_epoch += 1


def Load_Feed(csv_dir, pair, params, pool_len, average, mode=0):
    sample = Load_pair(csv_dir, pair)
    if mode == 0:
        grads, loss = EC.Feed_forward_backward(params, sample[0], sample[1],
                                               sample[2], pool_len, average)
        return grads, loss
    else:
        loss = EC.Feed_forward_backward(params, sample[0], sample[1], sample[2],
                                        pool_len, average, mode)
        return loss


def Train_multiprocess(params, train_list, valid_list, csv_dir, pool_len=3,
                       average=False, epoches=8, lr=0.1, cl=5, lr_halve_times=5,
                       mini_batch=100, process_num=20):
    multiprocessing.freeze_support()
    train_epoch = 1
    pp_span = 2
    best_epoch_loss = 1e9
    ht = 0
    while train_epoch <= epoches:
        print "Epoch %d, shuffling samples" % train_epoch
        shuffle(train_list)  # shuffle samples
        print "Start training epoch %d" % train_epoch
        t0 = time.time()
        pp_loss = 0
        for i in range(len(train_list) / mini_batch):
            grads = defaultdict(list)
            pool = multiprocessing.Pool(processes=process_num)
            results = []
            for j in range(mini_batch):
                pair = train_list[i * mini_batch + j]
                results.append(pool.apply_async(Load_Feed, (csv_dir, pair,
                               params, pool_len, average)))
            pool.close()
            pool.join()
            for ret in results:
                p_grads, loss = ret.get()
                pp_loss += loss
                for k in p_grads.keys():
                    if k in grads:
                        for idx_g in range(len(grads[k])):
                            grads[k][idx_g] += p_grads[k][idx_g]
                    else:
                        grads[k] = p_grads[k]
            for k in grads.keys():
                for idx_g in range(len(grads[k])):
                    grads[k][idx_g] /= mini_batch
            EC.All_params_SGD(params, grads, lr, cl)
            if i % pp_span == 0 and i != 0:
                t1 = time.time()
                print "\tAverage loss: %.12f" % (pp_loss /
                                                 float(pp_span * mini_batch)),
                print "\tsamples per sec %.2f" % (pp_span * mini_batch /
                                                  float(t1 - t0))
                t0 = t1
                pp_loss = 0
        rest_num = len(train_list) - (len(train_list) / mini_batch * mini_batch)
        if rest_num > 0:
            pool = multiprocessing.Pool(processes=process_num)
            results = []
            grads = defaultdict(list)
            for i in range((len(train_list) / mini_batch * mini_batch),
                           len(train_list)):
                pair = train_list[i]
                results.append(pool.apply_async(Load_Feed, (csv_dir, pair,
                               params, pool_len, average)))
            pool.close()
            pool.join()
            for ret in results:
                p_grads, _ = ret.get()
                for k in p_grads.keys():
                    if k in grads:
                        for idx_g in range(len(grads[k])):
                            grads[k][idx_g] += p_grads[k][idx_g]
                    else:
                        grads[k] = p_grads[k]
            for k in grads.keys():
                for idx_g in range(len(grads[k])):
                    grads[k][idx_g] /= rest_num
            EC.All_params_SGD(params, grads, lr, cl)
        valid_loss = Calculate_loss_multiprocess(params, valid_list, csv_dir,
                                                 pool_len, average, mini_batch,
                                                 process_num)
        print "Epoch %d \t Average valid loss: %.12f" % (train_epoch, valid_loss)
        Save_params(params, "save_params", train_epoch)
        if valid_loss > best_epoch_loss and ht < lr_halve_times:
            lr /= 2
            print "\t\tHalve learning rates to %.6f" % lr
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


def Calculate_loss(params, valid_list, csv_dir, pool_len, average):
    total_loss = 0
    num_samples = 0
    for pair in valid_list:
        sample = Load_pair(csv_dir, pair)
        loss = EC.Feed_forward_backward(params, sample[0], sample[1], sample[2],
                                        pool_len, average, mode=1)
        total_loss += loss
        num_samples += 1
    return total_loss / num_samples


def Calculate_loss_multiprocess(params, valid_list, csv_dir, pool_len, average,
                                mini_batch, process_num):
    total_loss = 0
    for i in range(len(valid_list) / mini_batch):
        results = []
        pool = multiprocessing.Pool(processes=process_num)
        for j in range(mini_batch):
            pair = valid_list[i * mini_batch + j]
            results.append(pool.apply_async(Load_Feed, (csv_dir, pair, params,
                           pool_len, average, 1)))  # mode=1
        pool.close()
        pool.join()
        for loss in results:
            total_loss += loss.get()
    rest_num = len(valid_list) - len(valid_list) / mini_batch * mini_batch
    if rest_num > 0:
        pool = multiprocessing.Pool(processes=process_num)
        results = []
        for i in range(len(valid_list) / mini_batch * mini_batch,
                       len(valid_list)):
            pair = valid_list[i]
            results.append(pool.apply_async(Load_Feed, (csv_dir, pair, params,
                           pool_len, average, 1)))  # mode=1
        pool.close()
        pool.join()
        for loss in results:
            total_loss += loss.get()
    return total_loss / len(valid_list)


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
    pkl_name = r"D:\EclipseProjects\SequenceToSequence\\train_valid_list.pkl"
    pkl = open(pkl_name, "rb")
    train_pairs = cPickle.load(pkl)
    valid_pairs = cPickle.load(pkl)
    # train_pairs = train_pairs[:17]
    # valid_pairs = valid_pairs[:8]
    pkl.close()
    csv_dir = "D:\IEMOCAP_full_release\emobase"
    hidden_size_list = [50, 50, 50]
    in_dim = 30     # consistent with csv column
    params = EC.Construct_net(hidden_size_list, in_dim)
    # Train(params, train_pairs, valid_pairs, csv_dir, pool_len=3, average=False,
    #       epochs=30)
    print "Start Training"
    Train_multiprocess(params, train_pairs, valid_pairs, csv_dir, pool_len=3,
                       average=False, lr=0.1, epoches=100, lr_halve_times=8,
                       mini_batch=100, process_num=20)

__author__ = 'v-penlu'
import EmoClassify as EC
import time
import cPickle
import os
import numpy as np


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
    pp_span = 2
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
    pkl = open("train_valid_list.pkl", "rb")
    train_pairs = cPickle.load(pkl)
    valid_pairs = cPickle.load(pkl)
    train_pairs = train_pairs[:2]
    valid_pairs = valid_pairs[:1]
    pkl.close()
    csv_dir = "D:\IEMOCAP_full_release\emobase"
    hidden_size_list = [50, 50, 50]
    in_dim = 28     # consistent with csv column
    params = EC.Construct_net(hidden_size_list, in_dim)
    Train(params, train_pairs, valid_pairs, csv_dir, pool_len=3, average=False,
          epochs=30)


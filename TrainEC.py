__author__ = 'v-penlu'
import EmoClassify as EC
import time


def Train(params, samples, valid_samples, test_samples, pool_len=3,
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
    trained_epoch = 0
    pp_span = 100
    best_epoch_loss = 1e9
    ht = 0
    while trained_epoch < epochs:
        trained_samples = 0
        pp_loss = 0
        t0 = time.time()
        for sample in samples:
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
        valid_loss = Calculate_loss(params, valid_samples, pool_len, average)
        test_loss = Calculate_loss(params, test_samples, pool_len, average)
        print "Average valid loss: %.5f, test loss: %.5f" % (valid_loss,
                                                             test_loss)
        if valid_loss > best_epoch_loss and ht < lr_halve_times:
            lr /= 2
            print "Halve learning rate to %.6f" % lr
            ht += 1
        else:
            best_epoch_loss = valid_loss
        trained_epoch += 1


def Calculate_loss(params, valid_samples, pool_len, average):
    total_loss = 0
    num_samples = 0
    for sample in valid_samples:
        loss = EC.Feed_forward_backward(params, sample[0], sample[1], sample[2],
                                        pool_len, average, mode=1)
        total_loss += loss
        num_samples += 1
    return total_loss / num_samples

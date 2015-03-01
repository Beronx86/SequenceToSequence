import SequenceToSequence as STS


train_file = "train.txt"
valid_file = "valid.txt"
test_file = "test.txt"
in_vocab_size = 160000 + 1 + 1   # 1 for <UNK>, 1 for <NUL>, <EOS> is included in 160000
out_vocab_size = 80000 + 1 + 1   # 1 for <UNK>, 1 for <NUL>
hidden_size_list = [1000, 1000, 1000]
we_size = 1000
embeding_range = 0.5 / we_size
print "Initializing networks..."
params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size,
                           out_vocab_size, embedding_range=embeding_range)
print "Start training..."
STS.Train(params, train_file, valid_file, test_file, ff_lr=0.1, lstm_lr=0.7)
print "Saving parameters..."
STS.Save_params(params, "pc4.pkl")

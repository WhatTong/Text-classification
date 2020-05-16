

class Config(object):

    train_path = 'Dataset/train.txt'
    validation_path = 'Dataset/validation.txt'
    test_path = 'Dataset/test.txt'

    epoch = 15

    embedding_dim = 100
    hidden_dim = 100
    batch_size = 64

    lr = 1e-3

    layer_size = 2

    bidirectional = False
    if bidirectional:
        num_direction = 2
    else:
        num_direction = 1

    sequence_length = 60  # 句子长度
    attention_size = 60

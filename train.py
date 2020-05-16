import torch
import torch.nn as nn
import torch.optim as optim
from Config import *
from process_data import *
import torch.utils.data as Data
from torch.autograd import Variable
from tqdm import tqdm
from LSTM import *
from LSTM_Attention import *


def get_data():
    print("Load Word2id...")
    word2id, tag2id = build_word2id(Config.train_path, Config.validation_path, Config.test_path)

    print("Load Data...")
    out = load_data(Config.train_path, Config.validation_path, Config.test_path, word2id, tag2id)

    print("Process Data...")
    x_train, y_train, x_validation, y_validation, x_test, y_test = process_data(out)

    train_data = Data.TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
    validation_data = Data.TensorDataset(torch.LongTensor(x_validation), torch.LongTensor(y_validation))
    test_data = Data.TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

    train_data = Data.DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    validation_data = Data.DataLoader(validation_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    test_data = Data.DataLoader(test_data, batch_size=Config.batch_size, shuffle=True, drop_last=True)

    vocab_size = len(word2id)
    tag_size = len(tag2id)

    return vocab_size, tag_size, train_data, validation_data, test_data


def train(vocab_size, tag_size, train_data, validation_data, test_data):

    model = LSTM_Attention(vocab_size, tag_size)
    criterion = nn.CrossEntropyLoss()
    optimzier = optim.Adam(model.parameters(), lr=Config.lr)
    best_acc = 0
    best_model = None

    for epoch in range(Config.epoch):
        train_loss = 0
        train_acc = 0

        model.train()
        print("Epoch{}:".format(epoch+1))
        for i, data in tqdm(enumerate(train_data), total=len(train_data)):
            x, y = data
            x, y = Variable(x), Variable(y)

            # forward
            out = model(x)
            loss = criterion(out, y)
            train_loss += loss.data.item()
            _, pre = torch.max(out, 1)

            num_acc = (pre == y).sum()
            train_acc += num_acc.data.item()

            # backward
            optimzier.zero_grad()
            loss.backward()
            optimzier.step()

        print('epoch [{}]: train loss is:{:.6f},train acc is:{:.6f}'
              .format(epoch+1, train_loss / (len(train_data) * Config.batch_size), train_acc / (len(train_data) * Config.batch_size)))

        model.eval()
        eval_loss = 0
        eval_acc = 0
        for i, data in enumerate(test_data):
            x, y = data

            with torch.no_grad():
                x = Variable(x)
                y = Variable(y)

            out = model(x)
            loss = criterion(out, y)
            eval_loss += loss.data.item()
            _, pre = torch.max(out, 1)

            num_acc = (pre == y).sum()
            eval_acc += num_acc.data.item()

        print('test loss is:{:.6f},test acc is:{:.6f}'
              .format(eval_loss / (len(test_data) * Config.batch_size), eval_acc / (len(test_data) * Config.batch_size)))

        if best_acc < (eval_acc / (len(test_data) * Config.batch_size)):
            best_acc = eval_acc / (len(test_data) * Config.batch_size)
            best_model = model.state_dict()
            # print(best_model)
            print('best acc is {:.6f},best model is changed'.format(best_acc))

    torch.save(model.state_dict(), './model/LSTM.pth')


if __name__ == '__main__':
    vocab_size, tag_size, train_data, validation_data, test_data = get_data()
    train(vocab_size, tag_size, train_data, validation_data, test_data)
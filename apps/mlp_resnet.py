import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    feat_block = []
    residual_block = []
    feat_block.append(nn.Linear(dim, hidden_dim))
    feat_block.append(norm(hidden_dim))
    feat_block.append(nn.ReLU())
    feat_block.append(nn.Dropout(drop_prob))
    feat_block.append(nn.Linear(hidden_dim, dim))
    feat_block.append(norm(dim))
    residual_block.append(nn.Residual(nn.Sequential(*feat_block)))
    residual_block.append(nn.ReLU())
    return nn.Sequential(*residual_block)


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    resnet = []
    resnet.append(nn.Linear(dim, hidden_dim))
    resnet.append(nn.ReLU())
    for i in range(num_blocks):
        resnet.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    resnet.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*resnet)



def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    softmax = nn.SoftmaxLoss()
    mean_err_rate, mean_loss, size = 0, 0, 0
    if opt is not None:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        x, y = data
        out = model(x)
        loss = softmax(out, y)

        mean_err_rate += np.sum(np.argmax(out.numpy(), axis=-1) != y.numpy())
        mean_loss += loss.numpy()

        if opt is not None:
            loss.backward()
            opt.step()
        size += 1

    return mean_err_rate / len(dataloader.dataset), mean_loss / size



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    train_dataset = ndl.data.MNISTDataset(\
            "./data/train-images-idx3-ubyte.gz",
            "./data/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(\
            "./data/t10k-images-idx3-ubyte.gz",
            "./data/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(\
             dataset=train_dataset,
             batch_size=batch_size,
             shuffle=True)
    test_dataloader = ndl.data.DataLoader(\
             dataset=test_dataset,
             batch_size=batch_size)

    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for i in range(epochs):
        train_err_rate, train_loss = epoch(train_dataloader, model, opt)
        test_err_rate, test_loss = epoch(test_dataloader, model)
    return [train_err_rate, train_loss, test_err_rate, test_loss]



if __name__ == "__main__":
    train_mnist(data_dir="../data")
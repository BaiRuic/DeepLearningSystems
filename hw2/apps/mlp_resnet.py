import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    f = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim, bias=True),
            norm(dim=hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(in_features=hidden_dim, out_features=dim, bias=True),
            norm(dim=dim),
        )

    block = nn.Sequential(
                nn.Residual(f),
                nn.ReLU(),
            )

    return block
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    model_list = [
        nn.Linear(dim, hidden_dim, True),
        nn.ReLU(),]
    
    for _ in range(num_blocks):
        model_list.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))

    model_list.append(nn.Linear(hidden_dim, num_classes, True))

    return nn.Sequential(*model_list)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model.train() if opt else model.eval()

    total_loss, total_err = 0., 0
    for X, y in dataloader:
        X = X.reshape((X.shape[0], 28*28))
        y_pred = model(X)
        loss = nn.SoftmaxLoss()(y_pred, y)
        # compute error
        total_err += np.sum(np.argmax(y_pred.numpy(),axis=-1) != y.numpy())
        total_loss += loss.numpy()
        # backprop
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    
    return total_err  / len(dataloader.dataset), \
           total_loss / len(dataloader.ordering)
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_ds = ndl.data.MNISTDataset(image_filename=(data_dir + "/train-images-idx3-ubyte.gz"),
                                     label_filename=(data_dir + "/train-labels-idx1-ubyte.gz"))
    test_ds = ndl.data.MNISTDataset(image_filename=(data_dir + "/t10k-images-idx3-ubyte.gz"),
                                    label_filename=(data_dir + "/t10k-labels-idx1-ubyte.gz"))
    train_dloader = ndl.data.DataLoader(dataset=train_ds,
                                        batch_size=batch_size,
                                        shuffle=True)
    test_dloaser = ndl.data.DataLoader(dataset=test_ds,
                                       batch_size=batch_size)

    resnet = MLPResNet(dim=28*28, hidden_dim=hidden_dim)
    
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)

    log = ()
    for _ in range(epochs):
        log = (*epoch(train_dloader, resnet, opt),
               *epoch(test_dloaser , resnet))
    return log
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")

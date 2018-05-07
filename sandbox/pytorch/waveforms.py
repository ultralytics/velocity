import time
import copy
import torch
import torch.utils.data as data_utils
import numpy as np
import scipy.io
from plotly.offline import plot
import plotly.graph_objs as go

torch.set_printoptions(linewidth=320, precision=8)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


def normalize(x, axis=None):
    # normalize NN inputs and outputs by column (axis=0)
    if axis is None:
        mu, sigma = x.mean(), x.std()
    elif axis == 0:
        mu, sigma = x.mean(0), x.std(0)
    elif axis == 1:
        mu, sigma = x.mean(1).reshape(x.shape[0], 1), x.std(1).reshape(x.shape[0], 1)
    return (x - mu) / sigma, mu, sigma


def shuffledata(x, y):
    i = np.arange(x.shape[0])
    np.random.shuffle(i)
    return x[i], y[i]


def splitdata(x, y, train=0.7, validate=0.15, test=0.15, shuffle=False):
    n = x.shape[0]
    if shuffle:
        x, y = shuffledata(x, y)
    i = round(n * train)
    j = round(n * validate) + i
    k = round(n * test) + j
    return x[:i], y[:i], x[i:j], y[i:j], x[j:k], y[j:k]  # xy train, xy validate, xy test


# http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors
@profile
def runexample():
    cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    path = '/Users/glennjocher/Google Drive/DATA/'

    H = [76, 23, 7]
    model = torch.nn.Sequential(
        torch.nn.Linear(512, H[0]), torch.nn.Tanh(),
        torch.nn.Linear(H[0], H[1]), torch.nn.Tanh(),
        torch.nn.Linear(H[1], H[2]), torch.nn.Tanh(),
        torch.nn.Linear(H[2], 2))

    # H = [32]
    # H = [81, 13]
    # H = [76, 23, 7]
    # H = [128, 32, 8]
    # H = [169, 56, 18, 6]

    lr = 0.005
    eps = 0.001
    batch_size = 10000
    epochs = 50000
    validation_checks = 5000
    name = 'nn%s%glr%geps' % (H[:], lr, eps)

    tica = time.time()
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Running on %s\n%s' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))

    mat = scipy.io.loadmat(path + 'MLTOFdataset.mat')
    x = mat['Ib']
    y = mat['T'][:, 0:2] - mat['tbias'][:, 0:2]
    nb, D_in = x.shape
    D_out = y.shape[1]

    x, _, _ = normalize(x, 1)
    # x, xmu, xs = normalize(x, 0)  # normalize each input column
    y, ymu, ys = normalize(y, 0)  # normalize each output column
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    x, y, xv, yv, xt, yt = splitdata(x, y, train=0.70, validate=0.15, test=0.15, shuffle=False)
    labels = ['train', 'validate', 'test']

    # SubsetRandomSampler
    train_dataset = data_utils.TensorDataset(x, y)
    test_dataset = data_utils.TensorDataset(xt, yt)
    train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = data_utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    class LinearTanh(torch.nn.Module):
        def __init__(self, D_in, D_out):
            super(LinearTanh, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, D_out)
            self.linear2 = torch.nn.Tanh()

        def forward(self, x):
            y = self.linear1(x)
            y = self.linear2(y)
            return y

    print(model)
    if cuda:
        x, xv, xt = x.cuda(), xv.cuda(), xt.cuda()
        y, yv, yt = y.cuda(), yv.cuda(), yt.cuda()
        model = model.cuda()

    # criteria and optimizer
    criteria = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, amsgrad=True)

    ticb = time.time()
    L = np.full((epochs, 3), np.nan)
    best = (0, 1E6, model.state_dict())  # best (epoch, validation loss, model)
    for i in range(epochs):
        # for j, (xj, yj) in enumerate(train_loader):
        #    print(xj.shape,time.time() - tic)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criteria(y_pred, y)
        L[i, 0] = loss.item()  # / y.numel()  # train
        L[i, 1] = criteria(model(xv), yv).item()  # / yv.numel()  # validate

        if i > 2000:  # validation checks
            if L[i, 1] < best[1]:
                best = (i, L[i, 1], copy.deepcopy(model.state_dict()))
            if (i - best[0]) > validation_checks:
                print('\n%g validation checks exceeded at epoch %g.\n' % (validation_checks, i))
                break

        if i % 1000 == 0:  # print and save progress
            rv = (model(xv) - yv).std(0).detach().cpu().numpy() * ys  # validate residual
            scipy.io.savemat(path + name + '.mat', dict(best=best[0:2], L=L, name=name))
            print('%.3fs' % (time.time() - ticb), i, L[i], rv)
            ticb = time.time()

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(best[2], path + name + '.pt')
    model.load_state_dict(best[2])
    dt = time.time() - tica

    print('\nFinished %g epochs in %.3fs (%.3f epochs/s)\nBest results from epoch %g:' % (i, dt, i / dt, best[0]))
    for i, (xi, yi) in enumerate(((x, y), (xv, yv), (xt, yt))):
        r = (model(xi) - yi)
        print('%.5f %s %s' % ((r ** 2).mean(), r.std(0).detach().cpu().numpy()[:] * ys, labels[i]))

    data = []
    for i, s in enumerate(['train', 'validate', 'test']):
        data.append(go.Scatter(x=np.arange(epochs), y=L[:, i], mode='markers+lines', name=s))
    layout = go.Layout(
        xaxis=dict(type='log', autorange=True),
        yaxis=dict(type='log', autorange=True))
    plot(go.Figure(data=data, layout=layout))


runexample()

import time
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


# http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors
# @profile
def runexample():
    cuda = torch.cuda.is_available()
    torch.manual_seed(0)
    mat = scipy.io.loadmat('/Users/glennjocher/Google Drive/DATA/MLTOFdataset.mat')

    tic = time.time()
    # dtype = torch.float
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Running on ' + str(device))
    if cuda:
        print(torch.cuda.get_device_properties(0))

    x = mat['Ia']
    y = mat['T'][:, 0:2] - mat['tbias'][:, 0:2]
    nb, D_in = x.shape
    D_out = y.shape[1]
    batch_size = 10000
    # nb, D_in, H, D_out = 6400, 1024, 20, 2

    x, xmu, xs = normalize(x, 0)  # normalize each input column
    y, ymu, ys = normalize(y, 0)  # normalize each output column
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    # x = torch.randn(nb, D_in)
    # y = torch.randn(nb, D_out)

    # SubsetRandomSampler
    train_dataset = data_utils.TensorDataset(x, y)
    train_loader = data_utils.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # test_loader = data_utils.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    class LinearTanh(torch.nn.Module):
        def __init__(self, D_in, D_out):
            # In the constructor we instantiate two nn.Linear modules and assign them as member variables.
            super(LinearTanh, self).__init__()
            self.linear1 = torch.nn.Linear(D_in, D_out)
            self.linear2 = torch.nn.Tanh()

        def forward(self, x):
            # In the forward function we accept a Tensor of input data and we must return a Tensor of output data.
            # We can use Modules defined in the constructor as well as arbitrary operators on Tensors.
            y = self.linear1(x)
            y = self.linear2(y)
            return y

    # H = [20, 20]
    H = [76, 23, 7]
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H[0]), torch.nn.Tanh(),
        # torch.nn.Linear(H[0], H[1]), torch.nn.Tanh(),
        LinearTanh(H[0], H[1]),
        torch.nn.Linear(H[1], H[2]), torch.nn.Tanh(),
        torch.nn.Linear(H[2], D_out))
    if cuda:
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()

    # criterion and optimizer
    criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)

    epochs = 30
    L = np.zeros((epochs, 1))
    for i in range(epochs):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        L[i] = loss.item()
        if i % 10 == 0:
            residual = (y_pred - y).std(0).detach().cpu().numpy() * ys
            print(i, L[i] / y.numel(), residual)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    dt = time.time() - tic
    residual = (y_pred - y).detach().cpu().numpy()
    residual = (residual * ys).std(0)
    print('\nFinished %g epochs in %.3fs (%.3f epochs/s)' % (i, dt, i / dt))
    print('Loss = %g\nStd = %s' % (loss / y.numel(), residual[:]))

    # Plot
    a = go.Scatter(x=np.arange(epochs), y=L[:, 0], mode='markers+lines', name='position')
    # b = go.Scatter(x=np.arange(epochs), y=L[:, 1], mode='markers+lines', name='time')
    plot([a])


runexample()

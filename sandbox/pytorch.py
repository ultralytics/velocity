# @profile
def runexample():
    import time
    import torch
    import torchvision
    import numpy as np
    import scipy.io
    from plotly.offline import plot
    import plotly.graph_objs as go
    tic = time.time()

    torch.set_printoptions(linewidth=320, precision=8)
    np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
    # http://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-tensors

    dtype = torch.float
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # dtype = torch.device("cuda:0") # Uncomment this to run on GPU

    # nb is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    nb, D_in, H, D_out = 6400, 1024, 20, 2

    # Create random input and output data
    torch.manual_seed(0)
    params = dict(device=device, dtype=dtype)

    mat = scipy.io.loadmat('/Users/glennjocher/Documents/PyCharmProjects/Velocity/data/MLTOFdataset.mat')
    x = mat['Ia']
    y = mat['T'][:, 0:2] - mat['tbias'][:, 0:2]
    nb, D_in = x.shape
    D_out = y.shape[1]

    # normalize = torchvision.transforms.Normalize(mean=x.mean(), std=x.std())
    # x = (x-x.mean()) / x.std()  # normalize each input row

    def normalize(x, axis=None):
        if axis is None:
            mu, sigma = x.mean(), x.std()
        elif axis == 0:
            mu, sigma = x.mean(0), x.std(0)
        elif axis == 1:
            mu, sigma = x.mean(1).reshape(x.shape[0], 1), x.std(1).reshape(x.shape[0], 1)
        return (x - mu) / sigma, mu, sigma

    x, xmu, xs = normalize(x, 1)
    y, ymu, ys = normalize(y, 0)

    x = torch.Tensor(x)
    y = torch.Tensor(y)
    # x = torch.randn(nb, D_in)
    # y = torch.randn(nb, D_out)

    # # different activations
    # data=[]
    # x = torch.Tensor(np.linspace(-5,5,1000))
    # activations = ('ReLU','RReLU','Hardtanh','ReLU6','Sigmoid','Tanh','ELU','SELU','GLU','Hardshrink','LeakyReLU',
    #                'LogSigmoid','Softplus','Softshrink','PReLU','Softsign','Tanhshrink','Softmin','Softmax')
    # activations = ('Tanh','Sigmoid','LogSigmoid')
    # for i in activations:
    #     f=eval('torch.nn.' + i + '()')
    #     data.append(go.Scatter(x=x, y=f(x), mode='markers+lines', name=i))
    # plot(data)

    # Randomly initialize weights
    useoptim = True
    usemodel = True
    autograd = True
    b1 = torch.randn(H, **params, requires_grad=autograd)
    w1 = torch.randn(D_in, H, **params, requires_grad=autograd)
    w2 = torch.randn(H, D_out, **params, requires_grad=autograd)

    if usemodel or useoptim:
        learning_rate = 2e-3
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.Tanh(),
            torch.nn.Linear(H, H),
            torch.nn.Tanh(),
            torch.nn.Linear(H, D_out))
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    else:
        learning_rate = 1e-6

    lossfcn = torch.nn.MSELoss(size_average=True)

    epochs = 100
    L = np.zeros((epochs, 1))
    for i in range(epochs):
        if autograd:
            # Forward pass: compute predicted y using operations on Tensors; these
            # are exactly the same operations we used to compute the forward pass using
            # Tensors, but we do not need to keep references to intermediate values since
            # we are not implementing the backward pass by hand.
            if usemodel or useoptim:
                y_pred = model(x)
            else:
                y_pred = (x @ w1 + b1).clamp(min=0) @ w2

            # Compute and print loss using operations on Tensors.
            # Now loss is a Tensor of shape (1,)
            # loss.item() gets the a scalar value held in the loss.
            # loss = (y_pred - y).pow(2).sum()
            loss = lossfcn(y_pred, y)
            # residual = (y_pred - y).detach().numpy()
            L[i] = loss.item()  # residual.std(0) * ys
            print(i, L[i])

            # Zero the gradients before running the backward pass.
            if usemodel:
                model.zero_grad()
            elif useoptim:
                optimizer.zero_grad()

            # Use autograd to compute the backward pass. This call will compute the
            # gradient of loss with respect to all Tensors with requires_grad=True.
            # After this call w1.grad and w2.grad will be Tensors holding the gradient
            # of the loss with respect to w1 and w2 respectively.
            loss.backward()

            if useoptim:
                optimizer.step()
            else:
                # Manually update weights using gradient descent. Wrap in torch.no_grad()
                # because weights have requires_grad=True, but we don't need to track this in autograd.
                # An alternative way is to operate on weight.data and weight.grad.data.
                # Recall that tensor.data gives a tensor that shares the storage with tensor, but doesn't track history.
                # You can also use torch.optim.SGD to achieve this.
                with torch.no_grad():
                    if usemodel:
                        for param in model.parameters():
                            param -= learning_rate * param.grad
                    else:
                        b1 -= learning_rate * b1.grad
                        w1 -= learning_rate * w1.grad
                        w2 -= learning_rate * w2.grad

                        # Manually zero the gradients after updating weights
                        b1.grad.zero_()
                        w1.grad.zero_()
                        w2.grad.zero_()
        else:
            # Forward pass: compute predicted y
            h = x @ w1
            h_relu = h.clamp(min=0)
            y_pred = h_relu @ w2

            # Compute and print loss
            loss = ((y_pred - y) ** 2).sum().item()
            print(i, loss)

            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_y_pred = 2.0 * (y_pred - y)
            grad_w2 = h_relu.t() @ grad_y_pred
            grad_h_relu = grad_y_pred @ w2.t()
            grad_h = grad_h_relu.clone()
            grad_h[h < 0] = 0
            grad_w1 = x.t() @ grad_h

            # Update weights using gradient descent
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2

    print('Done in %gs' % (time.time() - tic))
    residual = ((y_pred - y).detach().numpy() * ys).std(0)
    print('std = %s' % residual[:])

    a = go.Scatter(x=np.arange(epochs), y=L[:, 0], mode='markers+lines', name='position')
    # b = go.Scatter(x=np.arange(epochs), y=L[:, 1], mode='markers+lines', name='time')
    plot([a])


runexample()

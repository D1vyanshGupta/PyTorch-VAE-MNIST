# import core Python utils
import os
import argparse

# import PyTorch utils
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

# import torchvision utils for getting the MNIST dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

# setup argument parser for parsing arguments
parser = argparse.ArgumentParser(description='VAE MNIST Implementation')

# option to input batch size
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')

# option to input number of epochs
parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: 10)')

# option to set the logging interval
parser.add_argument('--zdims', type=int, default=30, metavar='N', help='number of dimensions of the latent vector (default: 20)')

# option to disable cuda implementation
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')

# option to input seed value for random number generator
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# option to set the logging interval
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')


# get all the parsed arguments
args = parser.parse_args()

# check whether to enable cude or not
args.cuda = (not args.no_cuda) and torch.cuda.is_available()

# seed the random number generator
torch.manual_seed(args.seed)

# setup the device on which the model will be run
device = torch.device('cuda' if args.cuda else 'cpu')

# setup kwargs for implementation
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# load the training data (MNIST Data Set)
# download the MNIST data set, transform data points to tensors and shuffle them
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# load the testing data (MNIST Data Set)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

# define the VAE
class VAE(nn.Module):

    # define the class constructor
    def __init__(self):
    # invoke the superclass constructor
        super(VAE, self).__init__()

        # encoder implementation

        # 1st fully connected layer of the encoder
        # 784 input pixels, 400 outputs
        self.en_fc1 = nn.Linear(784, 400)

        # the 2nd fully connected layer returns:
        #   a vector of means
        #   a vector of log variances
        # these means and variances correspond to the parameters of the
        # probability distribution (gaussian) of the latent vector dimensions

        # outputs the vector of means
        self.en_fc2_mu = nn.Linear(400, args.zdims)

        # outputs the vector of log variances
        self.en_fc2_log_var = nn.Linear(400, args.zdims)

        # decoder implementation

        # 1st fully connected layer of the decoder
        # latent vector as input pixels, 400 outputs
        self.dc_fc1 = nn.Linear(args.zdims, 400)

        # 2nd fully connected layer of the decoder
        # from 400 input pixels to 784 output pixels
        self.dc_fc2 = nn.Linear(400, 784)

    # implement the forward pass for the encoder network
    def encode(self, x: Variable) -> (Variable, Variable):

        """Input vector x -> 1st fully connected layer of encoder -> ReLU activation -> (fully connected
        layer for means, fully connected layer for log variances)

        Parameters
        ----------
        x : [128, 784] matrix; 128 digits of 28x28 pixels each

        Returns
        -------

        (mu, logvar) : args.zdims mean units one for each latent dimension, args.zdims
            variance units one for each latent dimension

        """

        # implement forward pass through the 1st layer of the encoder
        h1 = self.en_fc1(x)

        # apply ReLU activation
        h2 = F.relu(h1)

        # calculate the vector of means & log variances
        mu = self.en_fc2_mu(h2)
        log_var = self.en_fc2_log_var(h2)

        return mu, log_var

    # the key idea of the Kingma & Welling Paper
    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        """THE REPARAMETERIZATION IDEA:

        For each training sample (batch of 128 digit images)

        - take the current learned mu (mean), stddev (standard deviation) for each of the args.zdims
          dimensions of the latent vector, and for each dimension draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KL Divergence term (see loss_function() below)
          the distribution will tend to unit Gaussian

        Parameters
        ----------
        mu : [128, args.zdims] mean matrix
        logvar : [128, args.zdims] variance matrix

        Returns
        -------

        During training random sample from the learned args.zdims-dimensional
        normal distribution; during inference its mean.

        """
        if self.training:
            # multiply log_var with 0.5 (to take square root of the variance)
            # and then exponentiate to obtain the standard deviation
            std_dev = torch.exp(0.5*logvar)

            # for the latent vectors of each of the 128 digit images,
            # for each dimension of the latent vector, we sample from the
            # standard normal distribution N(0, 1)

            # get 128 vectors, whose each dimension is sampled from N(0, 1)
            eps = torch.randn_like(std_dev)

            # shift the individual components of the latent vectors
            # by the appropriate mean and standard deviation
            return eps.mul(std_dev).add_(mu)

        else:
            # during inference we simply return the mu which has the
            # highest probability of occurence
            return mu

    # implement the forward pass for the decoder network
    def decode(self, z: Variable) -> Variable:

        # implement forward pass through the 1st layer of the decoder
        h1 = self.dc_fc1(z)

        # apply the ReLU activation
        h2 = F.relu(h1)

        # implement forward pass through the 2nd layer of the decoder
        h3 = self.dc_fc2(h2)

        # apply the sigmoid activation
        return torch.sigmoid(h3)

    # finally, implement the forward pass for the VAE
    def forward(self, x):

        # flatten the 28 x 28 pixels into a vector of length 784
        mu, logvar = self.encode(x.view(-1, 784))

        # implement the reparameterization trick
        z = self.reparameterize(mu, logvar)

        # return the decoded representation along with mu & logvar
        return self.decode(z), mu, logvar

# instantiate the VAE model
model = VAE().to(device)

# setup the optimizer for training
# learning rate of 0.001
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# define the loss function
# the loss consists of two terms:
#   (i) the reconstruction loss - which forces q(z|x) to be close to the prior p(z)
#   (ii) KL divergence - forces the probability distributions of the dimensions of the latent space to be close to N(0, 1)

def loss_function(recon_x, x, mu, logvar) -> Variable:
    # binary cross-entropy loss
    # how well do the input image and the reconstructed image compare
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL divergence loss
    # penalize if the learned distributions deviate from N(0, 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD

# implement training
def train(epoch):
    # toggle model to train mode
    model.train()

    # instantiate the training loss to 0
    train_loss = 0

    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of args.batch_size samples and has shape [128, 1, 28, 28]

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        # initialize all gradients to 0
        optimizer.zero_grad()

        # implement the forward pass through the VAE to get reconstructed batch alog with mu & logvar
        recon_batch, mu, logvar = model(data)

        # calculate the training loss for this batcg
        loss = loss_function(recon_batch, data, mu, logvar)

        # implement back propagation
        loss.backward()

        # add the loss for this training batch
        train_loss += loss.item()

        # implement gradient descent
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# implement testing
def test(epoch):

    # create the directory for storing the results
    if not os.path.exists(os.path.join(os.getcwd(), 'results')):
        os.mkdir(os.path.join(os.getcwd(), 'results'))

    # toggle model to test / inference mode
    model.eval()

    # instantiate test loss to 0
    test_loss = 0

    with torch.no_grad():
        # each data is of args.batch_size (default 128) samples
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            # implement the forward pass through the VAE to get reconstructed batch alog with mu & logvar
            recon_batch, mu, logvar = model(data)

            # calculate the test loss for this batch
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            if i == 0:
                # for the first 128 batch of the epoch, show the first 8 input digits
                # with right below them the reconstructed output digits
                n = min(data.size(0), 8)

                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.data.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, args.zdims).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
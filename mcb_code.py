################################################################################
# This is the neural network that takes a stack of unfiltered single view back
# projections as input.  All layers are convolutional.  We tested this network 
# against the networks contained in FMCB_code.py and FCL_code.py.
################################################################################

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy
import PIL
from PIL import Image
from torchvision import datasets, transforms
import math
import matplotlib
import time

matplotlib.use('Agg')  # The reason we do this is so that matplotlib doesn't crash due to our display variables.
import matplotlib.pyplot


# Refactor Plan:
# 1) Clean up the first block of main() (the one that handles input args)
# 2) Write a Load Dataset function.
# 3) Write a "Generate Graphs" function.
# 4) Use pycharm's magic powers to search for/extract functions
# 5) See if pycharm has any magic powers to make moving functions to other files reasonable

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        width = 64
        self.num_layers = 20
        ker_size = 3
        pad_size = 1
        stride_size = 1
        leak = 0.01
        # input layer. Note: Don't pad first layer: our input exploits a better padding method for us.
        self.conv_in = nn.Conv2d(16, width, kernel_size=5, stride=1)
        dev = math.sqrt(2/width)/ker_size
        nn.init.normal_(self.conv_in.weight, mean=0, std=dev)
        # hidden layers. Note: We need to make this a ModuleList so that to("cuda") sees all the layers.
        self.convh = nn.ModuleList()
        for i in range(self.num_layers):
            self.convh.append(nn.Conv2d(width, width, kernel_size=ker_size, stride=stride_size, padding=pad_size))
            nn.init.normal_(self.convh[i].weight, mean=0, std=dev)
        # Batchnorm layers.
        self.batchnorm = nn.ModuleList()
        for i in range(self.num_layers):
            self.batchnorm.append(nn.BatchNorm2d(width))

        # output layer
        self.conv_out = nn.Conv2d(width, 1, kernel_size=ker_size, stride=stride_size, padding=pad_size)
        nn.init.normal_(self.conv_out.weight, mean=0, std=dev)

    def forward(self, x):
        # input layer
        x = f.leaky_relu(self.conv_in(x))
        # hidden layers
        for layer in range(self.num_layers):
            x = f.leaky_relu(self.batchnorm[layer](self.convh[layer](x)))
        # output layer
        x = self.conv_out(x)
        return x

    def verbose_forward(self, x):
        # input layer
        self.eval()  # This is a diagnostic function. This is not a training function.
        x = f.leaky_relu(self.conv_in(x))
        num_chans = self.conv_in.weight.size()[0]
        for i in range(0, num_chans):
            # save output of channel, not normalized
            layer_out = Image.fromarray(((x[0, i, :, :].detach().numpy()) * 255).astype(numpy.uint8))
            layer_out.save('VerboseOut/inLayer/Channel' + str(i + 1) + '.png')


            # save output of channel, normalized for viewing
            x_norm = x / torch.max(x)
            layer_out = Image.fromarray(((x_norm[0, i, :, :].detach().numpy()) * 255).astype(numpy.uint8))
            layer_out.save('VerboseOut/inLayer/Channel' + str(i + 1) + '_normalized.png')

            # save speckled output of channel
            x_spec = torch.ne(torch.zeros([1, 1, x.size()[2], x.size()[3]]), x_norm)
            layer_out = Image.fromarray(((x_spec[0, i, :, :].detach().numpy()) * 255).astype(numpy.uint8))
            layer_out.save('VerboseOut/inLayer/Channel' + str(i + 1) + '_Speckled.png')

        # save weight

        # hidden layers
        for layer in range(self.num_layers):
            x = f.leaky_relu(self.batchnorm[layer](self.convh[layer](x)))
            num_chans = self.convh[layer].weight.size()[0]
            for i in range(0, num_chans):
                # save output of channel, not normalized
                layer_out = Image.fromarray(((x[0, i, :, :].detach().numpy()) * 255).astype(numpy.uint8))
                layer_out.save('VerboseOut/layer' + str(layer+1) + '/Channel' + str(i + 1) + '.png')


                # save output of channel, normalized for viewing
                x_norm = x / torch.max(x)
                layer_out = Image.fromarray(((x_norm[0, i, :, :].detach().numpy()) * 255).astype(numpy.uint8))
                layer_out.save('VerboseOut/layer' + str(layer+1) + '/Channel' + str(i + 1) + '_normalized.png')

                # save speckled output of channel
                x_spec = torch.ne(torch.zeros([1, 1, x.size()[2], x.size()[3]]), x_norm)
                layer_out = Image.fromarray(((x_spec[0, i, :, :].detach().numpy()) * 255).astype(numpy.uint8))
                layer_out.save('VerboseOut/layer' + str(layer+1) + '/Channel' + str(i + 1) + '_Speckled.png')

        # output layer
        x = self.conv_out(x)
        # save output of channel, not normalized
        layer_out = Image.fromarray(((x[0, 0, :, :].detach().numpy()) * 255).astype(numpy.uint8))
        layer_out.save('VerboseOut/outLayer/Channel' + str(1) + '.png')


        # save output of channel, normalized for viewing
        x_norm = x / torch.max(x)
        layer_out = Image.fromarray(((x_norm[0, 0, :, :].detach().numpy()) * 255).astype(numpy.uint8))
        layer_out.save('VerboseOut/outLayer/Channel' + str(1) + '_normalized.png')

        # save speckled output of channel
        x_spec = torch.ne(torch.zeros([1, 1, x.size()[2], x.size()[3]]), x_norm)
        layer_out = Image.fromarray(((x_spec[0, 0, :, :].detach().numpy()) * 255).astype(numpy.uint8))
        layer_out.save('VerboseOut/outLayer/Channel' + str(1) + '_Speckled.png')

        return x


def train(model, device, datas, labels, data_size, batch_size, in_pixels, out_pixels, optimizer, epoch):
    model.train()  # Turns on the BatchNorm Layers, etc. that are only used for training.
    train_nrmse = 0
    total_batches = int(data_size / batch_size)
    index = torch.randperm(data_size)
    log_interval = 1
    for i in range(0, total_batches):
        # Load in a training example. Note: input must be 4 dimensional because pyTorch only takes in batches.
        data = torch.zeros([batch_size, 16, in_pixels, in_pixels])
        label = torch.zeros([batch_size, 1, out_pixels, out_pixels])
        for j in range(0, batch_size):
            data[j] = datas[index[i * batch_size + j]]
            label[j] = labels[index[i * batch_size + j]]
        # Move it to the GPU, if enabled
        data = data.to(device)
        label = label.to(device)
        # Train on the example
        optimizer.zero_grad()
        output = model(data)
        loss = f.mse_loss(output, label)
        loss.backward()
        optimizer.step()
        # Generate Training Error
        zeroval = torch.zeros([batch_size, 1, out_pixels, out_pixels])
        zeroval = zeroval.to(device)
        normalize = f.mse_loss(label, zeroval).item()
        nmse = loss.item() / normalize
        this_nrmse = math.sqrt(nmse)
        train_nrmse += this_nrmse
        # Update the user on the progress
        if i % log_interval == 0:
            print('Train Epoch: {} [{}/{}] \t NRMSE: {:.6f}'.format(epoch, i, total_batches, this_nrmse))
    return train_nrmse / total_batches


def test(model, device, datas, labels, data_size, in_pixels, out_pixels):
    model.eval()  # Disables the BatchNorm Layers, etc. that are only used for training
    test_nrmse = 0
    with torch.no_grad():
        for i in range(0, data_size):
            # Load in a verification example. Note: input must be 4 dimensional because pyTorch only takes in batches.
            data = torch.zeros([1, 16, in_pixels, in_pixels])
            data[0] = datas[i]
            label = torch.zeros([1, 1, out_pixels, out_pixels])
            label[0] = labels[i]
            # Move it to the GPU, if enabled
            data = data.to(device)
            label = label.to(device)
            # Test on the example
            output = model(data)
            test_loss = f.mse_loss(output, label).item()
            zeroval = torch.zeros([1, 1, out_pixels, out_pixels])
            zeroval = zeroval.to(device)
            normalize = f.mse_loss(label, zeroval).item()
            nmse = test_loss / normalize
            test_nrmse += math.sqrt(nmse)

    test_nrmse /= data_size
    print('\nTest set: Average NRMSE: {:.4f},\n'.format(test_nrmse))
    return test_nrmse


def main():
    start_time = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    ### EPOCHS #################################################################
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150)')
    ############################################################################
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Load Dataset

    data_size = 10000
    verify_size = 500
    out_pixels = 64
    padding_size = 2
    in_pixels = out_pixels + padding_size * 2
    datas = torch.zeros([data_size, 16, in_pixels, in_pixels])
    labels = torch.zeros([data_size, 1, out_pixels, out_pixels])
    ver_datas = torch.zeros([verify_size, 16, in_pixels, in_pixels])
    ver_labels = torch.zeros([verify_size, 1, out_pixels, out_pixels])

    for i in range(0, data_size):
        labels[i, 0, :, :] = torch.from_numpy(
            #numpy.array(PIL.Image.open('Dataset/out/Original' + str(i + 1) + '.png'), dtype='float')) / 255
            numpy.array(PIL.Image.open('train/orig/orig' + str(i + 1) + '.png'), dtype='float')) / 255

        for j in range(0, 16):
            # load data
            datas[i, j, :, :] = torch.from_numpy(
                #numpy.array(PIL.Image.open('Dataset/in/mcb' + str(i + 1) + 'channel' + str(j + 1) + '.png'),
                            #dtype='float')) / 255
                numpy.array(PIL.Image.open('train/mcb/mcb' + str(i + 1) + 'channel' + str(j + 1) + '.png'),
                            dtype='float')) / 255

    for i in range(0, verify_size):
        ver_labels[i, 0, :, :] = torch.from_numpy(
            #numpy.array(PIL.Image.open('verDataset/out/Original' + str(i + 1) + '.png'), dtype='float')) / 255
            numpy.array(PIL.Image.open('test/orig/orig' + str(i + 1) + '.png'), dtype='float')) / 255

        for j in range(0, 16):
            # load data
            ver_datas[i, j, :, :] = torch.from_numpy(
                #numpy.array(PIL.Image.open('verDataset/in/mcb' + str(i + 1) + 'channel' + str(j + 1) + '.png'),
                            #dtype='float')) / 255
                numpy.array(PIL.Image.open('test/mcb/mcb' + str(i + 1) + 'channel' + str(j + 1) + '.png'),
                            dtype='float')) / 255

    # Training Code begins here.

    # Generate the Network, and move everything to the training device (either CPU or GPU)
    model = Net().to(device)  # This is the line that is preventing you from moving layers to dictionaries
    labels = labels.to(device)
    datas = datas.to(device)
    ver_labels = ver_labels.to(device)
    ver_datas = ver_datas.to(device)
    # Pick an optimizer.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=0.01)  # Use ADAM
    # Setup metrics
    test_error = numpy.zeros(args.epochs)
    train_error = numpy.zeros(args.epochs)
    current_epoch = 1
    # Train, Test.
    for epoch in range(current_epoch, args.epochs + current_epoch):
        if epoch == 37:
            optimizer = optim.Adam(model.parameters(), lr=args.lr / 10,
                                   eps=0.01)  # Cut Learning Rate down after a long time.
        if epoch == 75:
            optimizer = optim.Adam(model.parameters(), lr= args.lr / 100, eps = 0.01)
        if epoch == 112:
            optimizer = optim.Adam(model.parameters(), lr=args.lr/ 1000, eps = 0.01)
        train_error[epoch - 1] = train(model, device, datas, labels, data_size, args.batch_size, in_pixels, out_pixels,
                                       optimizer, epoch)
        test_error[epoch - 1] = test(model, device, ver_datas, ver_labels, verify_size, in_pixels, out_pixels)
        state = {'epoch': epoch, 'state_dict()': model.state_dict()}
        torch.save(state, "Nets/NN.th")

    # Testing Metrics down below.

    # Make an example reconstruction.
    model = model.to("cpu")
    original_image = torch.zeros([1, 16, in_pixels, in_pixels])
    original_image = original_image.to("cpu")
    original_image[0] = ver_datas[0]
    reconstruction = model(original_image)
    original_label = torch.zeros([1, 1, out_pixels, out_pixels])
    original_label[0] = ver_labels[0]
    original_image = original_image.to("cpu")
    original_label = original_label.to("cpu")
    reconstruction = reconstruction.to("cpu")
    reconstruction = torch.clamp(reconstruction, 0, 1)
    writeable_original = Image.fromarray(((original_image[0, 0, :, :].numpy()) * 255).astype(numpy.uint8))
    writeable_reconst = Image.fromarray(((reconstruction[0, 0, :, :].detach().numpy()) * 255).astype(numpy.uint8))
    writeable_label = Image.fromarray(((original_label[0, 0, :, :].detach().numpy()) * 255).astype(numpy.uint8))
    writeable_label.save('train_label.png')
    writeable_original.save('train_original.png')
    writeable_reconst.save('train_reconst.png')

    original_image = torch.zeros([1, 16, in_pixels, in_pixels])
    original_image = original_image.to("cpu")
    original_image[0] = datas[0]
    reconstruction = model(original_image)
    original_label = torch.zeros([1, 1, out_pixels, out_pixels])
    original_label[0] = labels[0]
    original_image = original_image.to("cpu")
    original_label = original_label.to("cpu")
    reconstruction = reconstruction.to("cpu")
    reconstruction = torch.clamp(reconstruction, 0, 1)
    writeable_original = Image.fromarray(((original_image[0, 0, :, :].numpy()) * 255).astype(numpy.uint8))
    writeable_reconst = Image.fromarray(((reconstruction[0, 0, :, :].detach().numpy()) * 255).astype(numpy.uint8))
    writeable_label = Image.fromarray(((original_label[0, 0, :, :].detach().numpy()) * 255).astype(numpy.uint8))
    writeable_label.save('ver_label.png')
    writeable_original.save('ver_original.png')
    writeable_reconst.save('ver_reconst.png')

    # Make a plot of the NRMSE
    print(test_error)
    matplotlib.pyplot.plot(test_error)
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Average NRMSE')
    matplotlib.pyplot.title('NRMSE, Verification Examples, by epochs')
    matplotlib.pyplot.savefig('VerificationError.png')

    print(train_error)
    matplotlib.pyplot.plot(train_error)
    matplotlib.pyplot.xlabel('Epoch')
    matplotlib.pyplot.ylabel('Average NRMSE')
    matplotlib.pyplot.title('NRMSE, Training Examples, by epochs')
    matplotlib.pyplot.savefig('TrainError.png')

    print(str(time.time() - start_time) + " seconds")
    feed_data = torch.zeros([1, 16, in_pixels, in_pixels])
    feed_data[0] = datas[0]
    #model.verbose_forward(feed_data)


if __name__ == '__main__':
    main()




import os,sys
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt
import random
import numpy as np
from functools import reduce
from scipy import ndimage
from math import sqrt,floor,ceil
from batch_maker import StimMaker

# load saved learning parameters
def load_checkpoint(model, optimizer=None, losslogger=None, fname='checkpoint.pt'):
    # note that input model and optimizer should be pre-defined!!
    start_epoch = 0
    if os.path.isfile(fname):
        print('=> Loading checkpoint' + fname)
        checkpoint = torch.load(fname)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if losslogger is not None:
            losslogger = checkpoint['losslogger']
        print('=> loaded checkpoint ' + fname + ' epoch %3i'%start_epoch)
    else:
        print('=> no checkpoint found at' + fname)
    return model, optimizer, start_epoch, losslogger

# plot test results: crowding/uncrowding verson
def plot_test_results(results, norm='unknown',save_path='./result_figure'):
    # plot test results, plot 2 figures: crowd/uncrowd verion, plain performance plot
    # results   : Error rate (%) n_subjects x n_configs x 2 (shape - [:,:,0], vernier - [:,:,1])
    # save_path : file path and file name preface

    n_subjects,n_configs,n_targets = results.shape 

    # change result to accuracy
    results = 100-results

    # barPlot_pos = np.array([0,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,22,23,24,26,27,28])
    barPlot_pos = np.array([0,2,3,4,6,7,8,10,11,12,14,15,16,18,19,20,21,22])
    sel_conds   = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,16,19])

    test_names = [chr(i) for i in range(ord('a'),ord('a')+len(sel_conds))]

    # test_types  = ['ver','1sq','3sq','9sq','1st','3st','9st',\
    #                '1hg','3hg','9hg',\
    #                '1cc','3cc','9cc', \
    #                '3sqst','3sqbl','3stbl',
    #                '9msqst','9msqbl','9mstbl',\
    #                '9asqst','9asqbl','9astbl']

    # plain accuracy plot
    
    fig1 = plt.figure(figsize=(10,10))
    plt.bar(barPlot_pos, results[:,sel_conds,0].mean(axis=0),yerr=results[:,sel_conds,0].std(axis=0)/sqrt(n_subjects))
    plt.ylabel('Accuracy (%)')
    plt.xticks(barPlot_pos, test_names)
    plt.title('Shape results: '+norm)
    plt.ylim((0,100))
    fig2 = plt.figure(figsize=(10,10))
    plt.bar(barPlot_pos,results[:,sel_conds,1].mean(axis=0),yerr=results[:,sel_conds,1].std(axis=0)/sqrt(n_subjects))
    plt.ylabel('Accuracy (%)')
    plt.xticks(barPlot_pos, test_names)
    plt.title('Vernier results: '+norm)
    plt.ylim((0,100))
    fig1.savefig(save_path+'_shape_pc.png',transparent=True,bbox_inches='tight', pad_inches=0)
    fig2.savefig(save_path+'_vernier_pc.png',transparent=True,bbox_inches='tight', pad_inches=0)
    plt.close('all')

    # crowding vs uncrowding plot
    single_config = (np.tile(results[:, 1: 2,:],(1,3,1)),
                     np.tile(results[:, 4: 5,:],(1,3,1)),
                     np.tile(results[:, 7: 8,:],(1,3,1)),
                     np.tile(results[:,10:11,:],(1,3,1)),
                     np.tile(results[:, 1: 2,:],(1,2,1)),
                     np.tile(results[:, 4: 5,:],(1,1,1)),
                     np.tile(results[:, 1: 2,:],(1,2,1)),
                     np.tile(results[:, 4: 5,:],(1,1,1)),
                     np.tile(results[:, 1: 2,:],(1,2,1)),
                     np.tile(results[:, 4: 5,:],(1,1,1)),)
    diff_results  = results[:,1:,:] - np.concatenate(single_config,axis=1)
    fig1 = plt.figure(figsize=(10,10))
    plt.bar(barPlot_pos[1:], diff_results[:,sel_conds[1:]-1,0].mean(axis=0),yerr=diff_results[:,sel_conds[1:]-1,0].std(axis=0)/sqrt(n_subjects))
    plt.ylabel('Accuracy Differece')
    plt.xticks(barPlot_pos[1:], test_names[1:])
    plt.title('Shape results: '+norm)
    plt.ylim((-30,30))
    fig2 = plt.figure(figsize=(10,10))
    plt.bar(barPlot_pos[1:], diff_results[:,sel_conds[1:]-1,1].mean(axis=0),yerr=diff_results[:,sel_conds[1:]-1,1].std(axis=0)/sqrt(n_subjects))
    plt.ylabel('Accuracy Differece')
    plt.xticks(barPlot_pos[1:], test_names[1:])
    plt.title(r'$Vernier \ results: \ {} \ vernier \ alone \ pc: {:6.3} \pm {:6.3}$'.format(norm,results[:,0,1].mean(),results[:,0,1].std()/sqrt(n_subjects)))
    plt.ylim((-15,15))    
    fig1.savefig(save_path+'_shape_ad.png',transparent=True,bbox_inches='tight', pad_inches=0)
    fig2.savefig(save_path+'_vernier_ad.png',transparent=True,bbox_inches='tight', pad_inches=0)
    plt.close('all')


# visualize channel
def vis_channels(in_tensor, save_path):
    b,c,w,h = in_tensor.shape
    img = in_tensor[0,:,:,:].cpu().detach().numpy()

    # Subplot all channels' outputs in the 1st image in_tensor[0,:,:,:]
    nRow = round(sqrt(c))
    nCol = ceil(c/nRow)

    fig = plt.figure(figsize=(nCol*10,nRow*10))
    ax  = [[]]*c
    for i in range(c):
        ax[i] = fig.add_subplot(nRow,nCol,i+1)
        ax[i].imshow(img[i,:,:].squeeze(),vmin=0,vmax=1)

    plt.savefig(save_path,transparent=True,bbox_inches='tight', pad_inches=0)


# Generate train and test sets
def make_dataset(btch_size=50, shapeMatrix=[], imgSize=[120,120], shapeSize=18, 
                 type='train', device='cpu', fix_ver=False, make_shape_label_patterns=None, 
                 type_size_list=[[1,6,7],1,5]):

    barWidth = 1

    rufus = StimMaker(imgSize, shapeSize, barWidth)

    def transform_arrays(images, labels, device):
        images, labels = torch.Tensor(images).to(device), torch.Tensor(labels).to(device=device, dtype=torch.int64)
        images = images.permute(0, 3, 1, 2)  # Change to pytorch tensor dimension: batch x channel x height x width
        # normalize data
        mean, std = images.mean(), images.std()
        images.sub_(mean).div_(std)
        # # Change target to one hot vector: labels -> n x 2
        # # But CrossEntropyLoss doesn't expect the one-hot encoded vector, therefore no need to run below code..
        # batch_labels = torch.zeros(batch_labels.size(0), batch_labels.max().int()+1).scatter_(1, batch_labels.view(-1,1), 1).long()
        return images, labels

    if type is 'train':
        # use this for training set
        ratios = [0, 0, 1, 0]  # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside shape
        batch_images, batch_labels = rufus.generate_Batch(btch_size, ratios, noiseLevel=0.1, make_shape_label_patterns=make_shape_label_patterns)
        if make_shape_label_patterns is None:
            vernier_labels = batch_labels
            shape_labels   = []
        else:
            vernier_labels = batch_labels[0]
            shape_labels   = torch.Tensor(batch_labels[1]).to(device=device, dtype=torch.int64)

        batch_images, vernier_labels = transform_arrays(batch_images, vernier_labels, device)
    
    elif 'twinShapeConfig' in type:
        # training with equal-shape configurations
        # type_shape_list: [0]: shape types, [1]: maximum row, [2]: maximum col
        ratios = [[0,0,1,0]]*btch_size  # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside shape
        rows, cols, ind = np.random.randint(1,type_size_list[1]+1,btch_size),np.random.randint(1,type_size_list[2]+1,btch_size),np.random.randint(0,len(type_size_list[0]),btch_size)
        shapeMatrix = [np.ones((r,c))*type_size_list[0][i] for r,c,i in zip(rows, cols, ind)]

        batch_images   = np.ndarray(shape=(btch_size, imgSize[0], imgSize[1],3), dtype=np.float32)
        vernier_labels = np.zeros(btch_size, dtype=np.float32)
        
        # When specifying shapeMatrix, Stim_maker.generate_Batch makes whole batch with same configuration
        # Without specifying shapeMatrix, Stim_maker.generate_Batch creates random configurations by its own
        for b,(r,s) in enumerate(zip(ratios,shapeMatrix)):
            b_img, b_lab = rufus.generate_Batch(1, r, noiseLevel=0.1, shapeMatrix=s.tolist())
            batch_images[b,:,:,:] = b_img
            vernier_labels[b]     = b_lab

        # Transform arrays to a Pytorch friendly tensors and proper device ('cuda' or 'cpu')
        batch_images, vernier_labels = transform_arrays(batch_images, vernier_labels, device)

        # Specify shape labels torch tensor
        shape_labels = torch.Tensor(ind).to(device, dtype=torch.int64)

    elif type is 'test':
        # Test sets include: only vernier, 1,3,5,7,15,21 shapes (square, star, other shapes), trained shapes (some examples)
        ratios = [0, 0, 0, 1]  # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside shape
        batch_images, vernier_labels,shape_labels = ([],[],[])

        for shape in shapeMatrix:         
            imgs, labs = rufus.generate_Batch(btch_size, ratios, noiseLevel=0.1, shapeMatrix=shape)
            ss         = np.array(shape).reshape(-1) if len(shape) is not 0 else [1]
            ss         = ss[floor(len(ss)/2)] if ss[floor(len(ss)/2)] is not 0 else 1
            shape_labs = torch.ones(btch_size, device=device, dtype=torch.int64) * type_size_list[0].index(ss)

            imgs, labs = transform_arrays(imgs, labs, device)

            batch_images.append(imgs)
            vernier_labels.append(labs)   
            shape_labels.append(shape_labs) 

    return batch_images, vernier_labels, shape_labels

# Give the size of a layer given input size
def get_output_size(model, input_size=(3, 227, 227), device='CUDA'):
    with torch.no_grad():

        output_sizes = []
        x = torch.zeros(input_size).unsqueeze_(dim=0).to(device)

        for layer in list(model.features):
            x = layer(x)
            output_sizes.append(x.size()[1:])

        x = x.view(x.size(0), reduce(lambda a, b: a * b, x.size()[1:]))
        for layer in list(model.classifier):
            x = layer(x)
            output_sizes.append(x.size()[1:])
    
    return output_sizes


# Shape encoding neurons 
class MyShapeKernel(nn.Module):

    def __init__(self, input_size, kernel_size=3,n_hidden=64):

        super(MyShapeKernel, self).__init__()
        
        input_len = reduce(lambda a, b: a * b, input_size[1:])

        self.shape_kernel  = nn.Sequential(
            nn.Conv2d(input_size[0], 1, kernel_size=kernel_size, padding=1))
        self.shape_encoder = nn.Sequential(
            nn.BatchNorm1d(input_len),
            nn.Linear(input_len, n_hidden),
            nn.ELU(inplace=True),
            nn.Linear(n_hidden,2),
            nn.Softmax())

    def forward(self, x):
        x = self.shape_kernel(x)
        x = x.view(x.size(0),-1)
        x = self.shape_encoder(x)
        return x


# AlexNet network
class AlexNet(nn.Module):

    def __init__(self):

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000))

        self.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, ns):

        ys = []

        for n, layer in enumerate(self.features):
            x = layer(x)
            if n in ns:
                ys.append(x)

        x = x.view(x.size(0), reduce(lambda a, b: a * b, x.size()[1:]))
        for n, layer in enumerate(self.classifier):
            x = layer(x)
            if n + len(self.features) in ns:
                ys.append(x)

        return ys


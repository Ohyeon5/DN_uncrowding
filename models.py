# models of DN_normalizations
#
# Author: Oh-hyeon Choung 


# import relevant packages
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from oh_specs import *



# Define normalization type
def define_norm(norm_type,n_channel,n_group=None):
	# Referred to https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html
	if norm_type is 'bn':  # Batch normalization
		return nn.BatchNorm2d(n_channel)
	elif norm_type is 'gn':# Group normalization
		if n_group is None: n_group=2 # default group num is 2
		return nn.GroupNorm(n_group,n_channel)
	elif norm_type is 'in':# instance normalization
		return nn.GroupNorm(n_channel,n_channel)
	elif norm_type is 'ln':# layer normalization
		return nn.GroupNorm(1,n_channel)
	elif norm_type is 'None':
		bypass = lambda a: a
		return bypass
	else:
		return ValueError('Normalization type - '+norm_type+' is not defined yet')

# Define models 1: feedforward model
class feedforwardCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='bn',n_group=None, device='cpu'):

		super(feedforwardCNN, self).__init__()

		# specify conv params
		ini_k = [7,5,3]
		ini_s = [4,3,1]
		ini_p = [2,1,1]

		# Calculate flatten len
		def outsize(i,k,s,p):
			return floor((i+2*p-k+s)/s)

		fin_unit_len    = [32,0,0]
		fin_unit_len[1] = outsize(outsize(outsize(input_size[0],ini_k[0],ini_s[0],ini_p[0]),ini_k[1],ini_s[1],ini_p[1]),ini_k[2],ini_s[2],ini_p[2])
		fin_unit_len[2] = outsize(outsize(outsize(input_size[1],ini_k[0],ini_s[0],ini_p[0]),ini_k[1],ini_s[1],ini_p[1]),ini_k[2],ini_s[2],ini_p[2])
		flatten_len     = reduce(lambda a,b: a*b, fin_unit_len)

		self.device = device

		if 'in' in norm:
			norms = norm.split('_')
			if len(norms) is 1:
				norm = [norm]*3
			else:
				norm = ['None']*3
				for mm in norms[1]:
					norm[int(mm)] = 'in'
		else:
			norm = [norm]*3

		print('Norms are {}'.format(norm))

		self.conv0 = nn.Conv2d( 3,  8, kernel_size=ini_k[0], stride=ini_s[0], padding=ini_p[0])
		self.norm0 = define_norm(norm[0],  8, n_group)
		self.conv1 = nn.Conv2d( 8, 16, kernel_size=ini_k[1], stride=ini_s[1], padding=ini_p[1])
		self.norm1 = define_norm(norm[1], 16, n_group)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=ini_k[2], stride=ini_s[2], padding=ini_p[2])
		self.norm2 = define_norm(norm[2], 32, n_group)

		print(self.norm0)

		self.early_decoder = nn.Sequential(
		      nn.BatchNorm1d(flatten_len),
		      nn.Linear(flatten_len,hidden_dim),
		      nn.ReLU(inplace=True))
		self.shape_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,n_shapes),
		      nn.Softmax())
		self.vernier_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,2),
		      nn.Softmax())	
    
	def forward(self, x):

		x = self.conv0(x)
		x = self.norm0(x) 
		x = F.relu(x)
		x = self.conv1(x)
		x = self.norm1(x) 
		x = F.relu(x)
		x = self.conv2(x)
		x = self.norm2(x)
		x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier

	# entropy version
	def forward_ent(self, x):

		x = self.conv0(x)
		x = self.norm0(x)
		x = F.relu(x)
		x = self.conv1(x)
		x = instanceNorm_ent( x,self.norm1,device=self.device)
		x = F.relu(x)
		x = self.conv2(x) 
		x = instanceNorm_ent( x,self.norm2,device=self.device)
		x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier


# Define models 2: feedforward model + unfold plain sliding window -> norm='uspn'
# unfoldSlideNorm_fb
class unfoldSlidePlainNormCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='uspn',slide_size=3,n_group=None,timesteps=4,device='cpu'):

		super(unfoldSlidePlainNormCNN, self).__init__()

		# specify conv params
		ini_k = [7,5,3]
		ini_s = [4,3,1]
		ini_p = [2,1,1]

		self.slide_size = slide_size
		self.timesteps  = timesteps
		self.device     = device

		# Calculate output size of each layer
		def outsize(i,k,s,p):
			return floor((i+2*p-k+s)/s)

		self.unit_size = [[8,0,0],[16,0,0],[32,0,0]] # C,H,W
		self.unit_size[0] = [ 8,outsize(       input_size[0],ini_k[0],ini_s[0],ini_p[0]),outsize(       input_size[1],ini_k[0],ini_s[0],ini_p[0])]
		self.unit_size[1] = [16,outsize(self.unit_size[0][1],ini_k[1],ini_s[1],ini_p[1]),outsize(self.unit_size[0][2],ini_k[1],ini_s[1],ini_p[1])]
		self.unit_size[2] = [32,outsize(self.unit_size[1][1],ini_k[2],ini_s[2],ini_p[2]),outsize(self.unit_size[1][2],ini_k[2],ini_s[2],ini_p[2])]
		flatten_len  = reduce(lambda a,b: a*b, self.unit_size[2])

		self.conv0 = nn.Conv2d( 3,  8, kernel_size=ini_k[0], stride=ini_s[0], padding=ini_p[0])
		self.conv1 = nn.Conv2d( 8, 16, kernel_size=ini_k[1], stride=ini_s[1], padding=ini_p[1])
		self.conv2 = nn.Conv2d(16, 32, kernel_size=ini_k[2], stride=ini_s[2], padding=ini_p[2])

		self.norm0_gamma = nn.Parameter(torch.ones ((1, 8,self.unit_size[0][1],self.unit_size[0][2])))
		self.norm0_beta  = nn.Parameter(torch.zeros((1, 8,self.unit_size[0][1],self.unit_size[0][2])))
		self.norm1_gamma = nn.Parameter(torch.ones ((1,16,self.unit_size[1][1],self.unit_size[1][2])))
		self.norm1_beta  = nn.Parameter(torch.zeros((1,16,self.unit_size[1][1],self.unit_size[1][2])))
		self.norm2_gamma = nn.Parameter(torch.ones ((1,32,self.unit_size[2][1],self.unit_size[2][2])))
		self.norm2_beta  = nn.Parameter(torch.zeros((1,32,self.unit_size[2][1],self.unit_size[2][2])))

		self.early_decoder = nn.Sequential(
		      nn.BatchNorm1d(flatten_len),
		      nn.Linear(flatten_len,hidden_dim),
		      nn.ReLU(inplace=True))
		self.shape_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,n_shapes),
		      nn.Softmax())
		self.vernier_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,2),
		      nn.Softmax())	
    
	def forward(self, x):

		x = self.conv0(x)
		input_x = x

		for i in range(self.timesteps):
			
			if i is 0:
				norm0_gammaH = torch.ones ((1, 8,self.unit_size[0][1],self.unit_size[0][2])).to(self.device)
				norm0_betaH  = torch.zeros((1, 8,self.unit_size[0][1],self.unit_size[0][2])).to(self.device)
				norm1_betaH  = torch.zeros((1,16,self.unit_size[1][1],self.unit_size[1][2])).to(self.device)
				norm1_gammaH = torch.ones ((1,16,self.unit_size[1][1],self.unit_size[1][2])).to(self.device)
			else:
				upsample_4norm0 = nn.Upsample(size=self.unit_size[0][1:],mode='bilinear',align_corners=True)
				upsample_4norm1 = nn.Upsample(size=self.unit_size[1][1:],mode='bilinear',align_corners=True)

				norm0_gammaH = upsample_4norm0(self.norm1_gamma)[:,::2,:,:]
				norm0_betaH  = upsample_4norm0(self.norm1_beta) [:,::2,:,:]
				norm1_gammaH = upsample_4norm1(self.norm2_gamma)[:,::2,:,:]
				norm1_betaH  = upsample_4norm1(self.norm2_beta) [:,::2,:,:]

			x = unfoldSlideNorm_fb(input_x,self.norm0_gamma,self.norm0_beta,norm0_gammaH,norm0_betaH,ss=self.slide_size)
			x = F.relu(x)
			x = self.conv1(x)
			x = unfoldSlideNorm_fb(      x,self.norm1_gamma,self.norm1_beta,norm1_gammaH,norm1_betaH,ss=self.slide_size)
			x = F.relu(x)
			x = self.conv2(x) 
			x = unfoldSlideNorm   (      x,self.norm2_gamma,self.norm2_beta,ss=self.slide_size)
			x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier

# Unfold sliding window
def unfoldSlideNorm(x,gamma,beta,ss=3,eps=1e-5):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,H,W]
	# ss: slide size for slideNorm

	# Unfold: output tensor of shape (N,Cx(pi_kernelSize),L)
	# pi_kernelSize - all the elements in the block (ex. )
	b,c,h,w = x.shape
	unfold  = F.unfold(x,kernel_size=ss)
	_,pi,l  = unfold.shape
	mean    = F.upsample(unfold.reshape(b,c,pi//c,l).mean(dim=2).reshape(b,c,(h-ss+1),(w-ss+1)),size=(h,w),mode='nearest')
	var     = F.upsample(unfold.reshape(b,c,pi//c,l).var (dim=2).reshape(b,c,(h-ss+1),(w-ss+1)),size=(h,w),mode='nearest')
	x_new   = (x-mean)/torch.sqrt(var+eps)*gamma + beta

	return x_new

def unfoldSlideNorm_fb(x,gamma,beta,gammaH,betaH,ss=3,eps=1e-5):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,H,W]
	# gamma, beta: scale and offset from higher level featrues, learnable param, with shape [1,C,H,W]
	# ss: slide size for slideNorm 
	b,c,h,w = x.shape
	unfold  = F.unfold(x,kernel_size=ss)
	_,pi,l  = unfold.shape
	mean    = F.upsample(unfold.reshape(b,c,pi//c,l).mean(dim=2).reshape(b,c,(h-ss+1),(w-ss+1)),size=(h,w),mode='nearest')
	var     = F.upsample(unfold.reshape(b,c,pi//c,l).var (dim=2).reshape(b,c,(h-ss+1),(w-ss+1)),size=(h,w),mode='nearest')
	x_new   = (x-mean)/torch.sqrt(var+eps)*gamma*gammaH + beta + betaH

	return x_new


# Define models 3: feedforward model + instance normalization + feedback: norm type 'in_fb'
# instanceNormfbCNN
class instanceNormfbCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='in_fb',slide_size=5,n_group=None,timesteps=4,device='cpu'):

		super(instanceNormfbCNN, self).__init__()

		# specify conv params
		ini_k = [7,5,3]
		ini_s = [4,3,1]
		ini_p = [2,1,1]

		self.slide_size = slide_size
		self.timesteps  = timesteps
		self.device     = device

		# Calculate output size of each layer
		def outsize(i,k,s,p):
			return floor((i+2*p-k+s)/s)

		self.unit_size = [[8,0,0],[16,0,0],[32,0,0]] # C,H,W
		self.unit_size[0] = [ 8,outsize(       input_size[0],ini_k[0],ini_s[0],ini_p[0]),outsize(       input_size[1],ini_k[0],ini_s[0],ini_p[0])]
		self.unit_size[1] = [16,outsize(self.unit_size[0][1],ini_k[1],ini_s[1],ini_p[1]),outsize(self.unit_size[0][2],ini_k[1],ini_s[1],ini_p[1])]
		self.unit_size[2] = [32,outsize(self.unit_size[1][1],ini_k[2],ini_s[2],ini_p[2]),outsize(self.unit_size[1][2],ini_k[2],ini_s[2],ini_p[2])]
		flatten_len  = reduce(lambda a,b: a*b, self.unit_size[2])

		self.conv0 = nn.Conv2d( 3,  8, kernel_size=ini_k[0], stride=ini_s[0], padding=ini_p[0])
		self.conv1 = nn.Conv2d( 8, 16, kernel_size=ini_k[1], stride=ini_s[1], padding=ini_p[1])
		self.conv2 = nn.Conv2d(16, 32, kernel_size=ini_k[2], stride=ini_s[2], padding=ini_p[2])

		self.norm0 = nn.GroupNorm( 8, 8)
		self.norm1 = nn.GroupNorm(16,16)
		self.norm2 = nn.GroupNorm(32,32)

		self.norm0fb_gamma = nn.Parameter(torch.ones (self.norm0.weight.view(1,-1,1,1).shape))
		self.norm0fb_beta  = nn.Parameter(torch.zeros(self.norm0.bias.view(1,-1,1,1).shape))
		self.norm1fb_gamma = nn.Parameter(torch.ones (self.norm1.weight.view(1,-1,1,1).shape))
		self.norm1fb_beta  = nn.Parameter(torch.zeros(self.norm1.bias.view(1,-1,1,1).shape))

		self.early_decoder = nn.Sequential(
		      nn.BatchNorm1d(flatten_len),
		      nn.Linear(flatten_len,hidden_dim),
		      nn.ReLU(inplace=True))
		self.shape_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,n_shapes),
		      nn.Softmax())
		self.vernier_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,2),
		      nn.Softmax())	
    
	def forward(self, x):

		x = self.conv0(x)
		input_x = x

		for i in range(self.timesteps):

			if i is 0: 
				mean_conv1 = None
				var_conv1  = None
				mean_conv2 = None
				var_conv2  = None
			else:
				mean_conv1 = mean_conv1[:,::2,:,:]
				var_conv1  = var_conv1 [:,::2,:,:]
				mean_conv2 = mean_conv2[:,::2,:,:]
				var_conv2  = var_conv2 [:,::2,:,:]

			x, _, __ = instanceNorm_fb(input_x,self.norm0, self.norm0fb_gamma, self.norm0fb_beta, mean_conv1, var_conv1, timesteps=i)
			x = F.relu(x)
			x = self.conv1(x)
			x, mean_conv1, var_conv1 = instanceNorm_fb( x,self.norm1, self.norm1fb_gamma, self.norm1fb_beta, mean_conv2, var_conv2, timesteps=i)
			x = F.relu(x)
			x = self.conv2(x) 
			x, mean_conv2, var_conv2 = instanceNorm_fb( x,self.norm2, timesteps=0)
			x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier

def instanceNorm_fb(x, norm, gammafb=None, betafb=None, meanH=None, varH=None, timesteps=0, eps=1e-5):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,1,1]

	b,c,h,w = x.shape
	meanL = x.mean(dim=[2,3], keepdim=True)
	varL  = x.var(dim=[2,3], keepdim=True)
	if timesteps is 0:
		# Feedforward instance normalization
		x_new = norm(x)

	else:
		# Feedback normalization based on the next conv layer's activations
		x_new = (x-meanH)/torch.sqrt(varH+eps)*gammafb + betafb

	return x_new, meanL, varL


# Define models 4: feedforward + patch instance normalization + feedback: norm = 'inp_fb'(only r,b in feedback sweep) or 'inp_fb_plain' 
# patchInstanceNormfbCNN
class patchInstanceNormfbCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='inp_fb',slide_size=5,n_group=None,timesteps=4,device='cpu'):

		super(patchInstanceNormfbCNN, self).__init__()

		# specify conv params
		ini_k = [7,5,3]
		ini_s = [4,3,1]
		ini_p = [2,1,1]

		self.slide_size = slide_size
		self.timesteps  = timesteps
		self.device     = device
		self.norm       = norm

		# Calculate output size of each layer
		def outsize(i,k,s,p):
			return floor((i+2*p-k+s)/s)

		self.unit_size = [[8,0,0],[16,0,0],[32,0,0]] # C,H,W
		self.unit_size[0] = [ 8,outsize(       input_size[0],ini_k[0],ini_s[0],ini_p[0]),outsize(       input_size[1],ini_k[0],ini_s[0],ini_p[0])]
		self.unit_size[1] = [16,outsize(self.unit_size[0][1],ini_k[1],ini_s[1],ini_p[1]),outsize(self.unit_size[0][2],ini_k[1],ini_s[1],ini_p[1])]
		self.unit_size[2] = [32,outsize(self.unit_size[1][1],ini_k[2],ini_s[2],ini_p[2]),outsize(self.unit_size[1][2],ini_k[2],ini_s[2],ini_p[2])]
		flatten_len  = reduce(lambda a,b: a*b, self.unit_size[2])

		self.conv0 = nn.Conv2d( 3,  8, kernel_size=ini_k[0], stride=ini_s[0], padding=ini_p[0])
		self.conv1 = nn.Conv2d( 8, 16, kernel_size=ini_k[1], stride=ini_s[1], padding=ini_p[1])
		self.conv2 = nn.Conv2d(16, 32, kernel_size=ini_k[2], stride=ini_s[2], padding=ini_p[2])

		self.norm0fb_gamma = nn.Parameter(torch.ones ((1, 8,self.unit_size[0][1],self.unit_size[0][2])))
		self.norm0fb_beta  = nn.Parameter(torch.zeros((1, 8,self.unit_size[0][1],self.unit_size[0][2])))
		self.norm1fb_gamma = nn.Parameter(torch.ones ((1,16,self.unit_size[1][1],self.unit_size[1][2])))
		self.norm1fb_beta  = nn.Parameter(torch.zeros((1,16,self.unit_size[1][1],self.unit_size[1][2])))
		self.norm2fb_gamma = nn.Parameter(torch.ones ((1,32,self.unit_size[2][1],self.unit_size[2][2])))
		self.norm2fb_beta  = nn.Parameter(torch.zeros((1,32,self.unit_size[2][1],self.unit_size[2][2])))

		self.early_decoder = nn.Sequential(
		      nn.BatchNorm1d(flatten_len),
		      nn.Linear(flatten_len,hidden_dim),
		      nn.ReLU(inplace=True))
		self.shape_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,n_shapes),
		      nn.Softmax())
		self.vernier_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,2),
		      nn.Softmax())	
    
	def forward(self, x):

		x = self.conv0(x)
		input_x = x

		for i in range(self.timesteps):
			
			if i is 0:
				mean_conv1 = None
				var_conv1  = None
				mean_conv2 = None
				var_conv2  = None
			else:

				upsample_4norm0 = nn.Upsample(size=self.unit_size[0][1:],mode='nearest')
				upsample_4norm1 = nn.Upsample(size=self.unit_size[1][1:],mode='nearest')

				mean_conv1 = upsample_4norm0(mean_conv1)[:,::2,:,:]
				var_conv1  = upsample_4norm0(var_conv1) [:,::2,:,:]
				mean_conv2 = upsample_4norm1(mean_conv2)[:,::2,:,:]
				var_conv2  = upsample_4norm1(var_conv2) [:,::2,:,:]

			x, _, __         = patchInstanceNorm_fb(input_x,self.norm0fb_gamma,self.norm0fb_beta,mean_conv1,var_conv1,ss=self.slide_size,timesteps=i, norm=self.norm)
			x = F.relu(x)
			x = self.conv1(x)
			x,mean_conv1,var_conv1 = patchInstanceNorm_fb(x,self.norm1fb_gamma,self.norm1fb_beta,mean_conv2,var_conv2,ss=self.slide_size,timesteps=i, norm=self.norm)
			x = F.relu(x)
			x = self.conv2(x) 
			x,mean_conv2,var_conv2 = patchInstanceNorm_fb(x,self.norm2fb_gamma,self.norm2fb_beta,ss=self.slide_size,timesteps=0, norm=self.norm, last_flg=True)
			x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier

# patch instance normalization
def patchInstanceNorm_fb(x,gamma,beta,meanH=None,varH=None,ss=5,eps=1e-5,timesteps=0, norm='pin_fb', last_flg=False):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,H-ss,W-ss]
	# gamma, beta: scale and offset from higher level featrues, learnable param, with shape [1,C,H,W]
	# ss: slide size for slideNorm 
	# last_flg: The last convolution layer flag
	b,c,h,w = x.shape
	unfold  = F.unfold(x,kernel_size=ss)
	_,pi,l  = unfold.shape
	meanL   = F.upsample(unfold.reshape(b,c,pi//c,l).mean(dim=2).reshape(b,c,(h-ss+1),(w-ss+1)),size=(h,w),mode='nearest')
	varL    = F.upsample(unfold.reshape(b,c,pi//c,l).var (dim=2).reshape(b,c,(h-ss+1),(w-ss+1)),size=(h,w),mode='nearest')

	if timesteps is 0 :
		meanH = meanL
		varH  = varL
		if ('plain' not in norm) or last_flg: 
			x_new   = (x-meanH)/torch.sqrt(varH+eps)*gamma + beta
		else:	
			# first sweep without the learnable shifting and scaling factors
			x_new = (x-meanH)/torch.sqrt(varH+eps)
	
	else:
		# Only learnable parameters in feedback sequence
		x_new   = (x-meanH)/torch.sqrt(varH+eps)*gamma + beta

	return x_new, meanL, varL


# Define models 4: feedforward model + instance normalization + entropy: norm type 'ine'
# instanceNormfbCNN
class instanceNormEntropyCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='ine',slide_size=5,n_group=None,timesteps=4,device='cpu'):

		super(instanceNormEntropyCNN, self).__init__()

		# specify conv params
		ini_k = [7,5,3]
		ini_s = [4,3,1]
		ini_p = [2,1,1]

		self.slide_size = slide_size
		self.timesteps  = timesteps
		self.device     = device

		# Calculate output size of each layer
		def outsize(i,k,s,p):
			return floor((i+2*p-k+s)/s)

		self.unit_size = [[8,0,0],[16,0,0],[32,0,0]] # C,H,W
		self.unit_size[0] = [ 8,outsize(       input_size[0],ini_k[0],ini_s[0],ini_p[0]),outsize(       input_size[1],ini_k[0],ini_s[0],ini_p[0])]
		self.unit_size[1] = [16,outsize(self.unit_size[0][1],ini_k[1],ini_s[1],ini_p[1]),outsize(self.unit_size[0][2],ini_k[1],ini_s[1],ini_p[1])]
		self.unit_size[2] = [32,outsize(self.unit_size[1][1],ini_k[2],ini_s[2],ini_p[2]),outsize(self.unit_size[1][2],ini_k[2],ini_s[2],ini_p[2])]
		flatten_len  = reduce(lambda a,b: a*b, self.unit_size[2])

		self.conv0 = nn.Conv2d( 3,  8, kernel_size=ini_k[0], stride=ini_s[0], padding=ini_p[0])
		self.conv1 = nn.Conv2d( 8, 16, kernel_size=ini_k[1], stride=ini_s[1], padding=ini_p[1])
		self.conv2 = nn.Conv2d(16, 32, kernel_size=ini_k[2], stride=ini_s[2], padding=ini_p[2])

		self.norm0 = nn.GroupNorm( 8, 8)
		self.norm1 = nn.GroupNorm(16,16)
		self.norm2 = nn.GroupNorm(32,32)

		self.early_decoder = nn.Sequential(
		      nn.BatchNorm1d(flatten_len),
		      nn.Linear(flatten_len,hidden_dim),
		      nn.ReLU(inplace=True))
		self.shape_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,n_shapes),
		      nn.Softmax())
		self.vernier_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,2),
		      nn.Softmax())	
    
	def forward(self, x):

		x = self.conv0(x)
		x = self.norm0(x)
		x = F.relu(x)
		x = self.conv1(x)
		x = instanceNorm_ent( x,self.norm1,device=self.device)
		x = F.relu(x)
		x = self.conv2(x) 
		x = instanceNorm_ent( x,self.norm2,device=self.device)
		x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier

def instanceNorm_ent(x, norm, kernel_size=5,eps=1e-5,device='cpu'):
	# x: input features with shape [B,C,H,W]

	x_copy  = x.clone().detach()
	ent     = channel_entropy(x_copy,kernel_size,device)	 # output entropy is in shape [B,C]
	# large entropy -> less suppression
	x_new = (x-x.mean(dim=[2,3],keepdim=True))/torch.sqrt(x.var(dim=[2,3],keepdim=True)+eps)*norm.weight.reshape(1,-1,1,1)*ent + norm.bias.reshape(1,-1,1,1)

	return x_new

def channel_entropy(x, kernel_size=5,device='cpu'):
	# x: input features with shape [B,C,H,W]
	# entropy: output should be size [C] : return mean of entropy values
	# let's calculate with scipy built-in function.... 
	b,c,h,w = x.shape
	unfold  = F.unfold(x,kernel_size=kernel_size)
	_,pi,l  = unfold.shape
	unfold  = unfold.reshape(b,c,pi//c,l)

	entropy = torch.zeros(b,c).to(device)
	for i0 in range(b):
		for i1 in range(c):
			ent = torch.zeros(l)
			for i3 in range(l):
				pk      = unfold[i0,i1,:,i3].histc(bins=25)/(kernel_size**2)
				ent[i3] = (-pk*torch.log(pk.clamp(min=1e-5, max=1 - 1e-5))).sum()
			entropy[i0,i1] = ent.mean()
	return entropy.unsqueeze(-1).unsqueeze(-1)


# Define models 5: feedforward model + instance normalization + backpropagation inhibition: norm type 'bpin'
# instanceNormfbCNN
class backProbINCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='ine',slide_size=5,n_group=None,timesteps=4,device='cpu'):

		super(backProbINCNN, self).__init__()

		# specify conv params
		ini_k = [7,5,3]
		ini_s = [4,3,1]
		ini_p = [2,1,1]

		self.slide_size = slide_size
		self.timesteps  = timesteps
		self.device     = device

		# Calculate output size of each layer
		def outsize(i,k,s,p):
			return floor((i+2*p-k+s)/s)

		self.unit_size = [[8,0,0],[16,0,0],[32,0,0]] # C,H,W
		self.unit_size[0] = [ 8,outsize(       input_size[0],ini_k[0],ini_s[0],ini_p[0]),outsize(       input_size[1],ini_k[0],ini_s[0],ini_p[0])]
		self.unit_size[1] = [16,outsize(self.unit_size[0][1],ini_k[1],ini_s[1],ini_p[1]),outsize(self.unit_size[0][2],ini_k[1],ini_s[1],ini_p[1])]
		self.unit_size[2] = [32,outsize(self.unit_size[1][1],ini_k[2],ini_s[2],ini_p[2]),outsize(self.unit_size[1][2],ini_k[2],ini_s[2],ini_p[2])]
		flatten_len  = reduce(lambda a,b: a*b, self.unit_size[2])

		self.conv0 = nn.Conv2d( 3,  8, kernel_size=ini_k[0], stride=ini_s[0], padding=ini_p[0])
		self.conv1 = nn.Conv2d( 8, 16, kernel_size=ini_k[1], stride=ini_s[1], padding=ini_p[1])
		self.conv2 = nn.Conv2d(16, 32, kernel_size=ini_k[2], stride=ini_s[2], padding=ini_p[2])

		self.norm0 = nn.GroupNorm( 8, 8)
		self.norm1 = nn.GroupNorm(16,16)
		self.norm2 = nn.GroupNorm(32,32)

		self.early_decoder = nn.Sequential(
		      nn.BatchNorm1d(flatten_len),
		      nn.Linear(flatten_len,hidden_dim),
		      nn.ReLU(inplace=True))
		self.shape_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,n_shapes),
		      nn.Softmax())
		self.vernier_decoder = nn.Sequential(
		      nn.Linear(hidden_dim,2),
		      nn.Softmax())	
    
	def forward(self, x):

		x = self.conv0(x)
		x = self.norm0(x)
		x = F.relu(x)
		x = self.conv1(x)
		x = instanceNorm_ent( x,self.norm1,device=self.device)
		x = F.relu(x)
		x = self.conv2(x) 
		x = instanceNorm_ent( x,self.norm2,device=self.device)
		x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier

def backprobIN(x, norm, kernel_size=5,eps=1e-5,device='cpu'):
	# x: input features with shape [B,C,H,W]

	x_copy  = x.clone().detach()
	ent     = channel_entropy(x_copy,kernel_size,device)	 # output entropy is in shape [B,C]
	# large entropy -> less suppression
	x_new = (x-x.mean(dim=[2,3],keepdim=True))/torch.sqrt(x.var(dim=[2,3],keepdim=True)+eps)*norm.weight.reshape(1,-1,1,1)*ent + norm.bias.reshape(1,-1,1,1)

	return x_new

def channel_entropy(x, kernel_size=5,device='cpu'):
	# x: input features with shape [B,C,H,W]
	# entropy: output should be size [C] : return mean of entropy values
	# let's calculate with scipy built-in function.... 
	b,c,h,w = x.shape
	unfold  = F.unfold(x,kernel_size=kernel_size)
	_,pi,l  = unfold.shape
	unfold  = unfold.reshape(b,c,pi//c,l)

	entropy = torch.zeros(b,c).to(device)
	for i0 in range(b):
		for i1 in range(c):
			ent = torch.zeros(l)
			for i3 in range(l):
				pk      = unfold[i0,i1,:,i3].histc(bins=25)/(kernel_size**2)
				ent[i3] = (-pk*torch.log(pk.clamp(min=1e-5, max=1 - 1e-5))).sum()
			entropy[i0,i1] = ent.mean()
	return entropy.unsqueeze(-1).unsqueeze(-1)









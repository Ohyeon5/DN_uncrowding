# Deprived models

# import relevant packages
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from oh_specs import *



# This is again wrongggggg!!! Mean part is fine, but variance part is so wronnnggggg 
# AvgPooling normalization 1: Plain normalizations (Everything has to be tensor!!!!)
def avgpoolNorm(x,gamma,beta,ss=3,eps=1e-5):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,H,W]
	# ss: slide size for slideNorm 

	b,c,h,w = x.shape
	mean    = F.upsample(F.avg_pool2d(x,kernel_size=ss,stride=1),size=(h,w),mode='nearest')
	var     = torch.pow(x-mean,2).sum(dim=[0,2,3],keepdim=True)
	x_new   = (x-mean)/torch.sqrt(var+eps)*gamma + beta

	return x_new

def avgpoolNorm_fb(x,gamma,beta,gammaH,betaH,ss=3,eps=1e-5):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,H,W]
	# gamma, beta: scale and offset from higher level featrues, learnable param, with shape [1,C,H,W]
	# ss: slide size for slideNorm 

	b,c,h,w = x.shape
	mean    = F.upsample(F.avg_pool2d(x,kernel_size=ss,stride=1),size=(h,w),mode='nearest')
	var     = torch.pow(x-mean,2).sum(dim=[0,2,3],keepdim=True)
	x_new   = (x-mean)/torch.sqrt(var+eps)*gamma*gammaH + beta + betaH
  
	return x_new


class avgpoolPlainNormCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='avgpn',slide_size=3,n_group=None,timesteps=4,device='cpu'):

		super(avgpoolPlainNormCNN, self).__init__()

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

			x = avgpoolNorm_fb(input_x,self.norm0_gamma,self.norm0_beta,norm0_gammaH,norm0_betaH,ss=self.slide_size)
			x = F.relu(x)
			x = self.conv1(x)
			x = avgpoolNorm_fb(      x,self.norm1_gamma,self.norm1_beta,norm1_gammaH,norm1_betaH,ss=self.slide_size)
			x = F.relu(x)
			x = self.conv2(x)
			x = avgpoolNorm   (      x,self.norm2_gamma,self.norm2_beta,ss=self.slide_size)
			x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier


# These Sliding window things are so wrong.....
# Sliding window normalizations 1: Plain normalizations (Everything has to be tensor!!!!)
def slideNorm(x,gamma,beta,ss=3,eps=1e-5):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,H-ss,W-ss]
	# ss: slide size for slideNorm 

	b,c,h,w = x.shape
	for yy in range(h-ss):
		for xx in range(w-ss):
			x_old = x[:,:,yy:yy+ss,xx:xx+ss].clone()
			mean  = torch.mean(x_old,dim=[2,3],keepdim=True)
			var   = torch.std (x_old,dim=[2,3],keepdim=True)
			x_new = (x_old-mean) / torch.sqrt(var+eps) * gamma[:,:,yy:yy+1,xx:xx+1] + beta[:,:,yy:yy+1,xx:xx+1]
			
			x[:,:,yy:yy+ss,xx:xx+ss] = x_new
	return x

def slideNorm_fb(x,gamma,beta,gammaH,betaH,ss=3,eps=1e-5):
	# x: input features with shape [B,C,H,W]
	# gamma, beta: scale and offset, learnable param, with shape [1,C,H-ss,W-ss]
	# gamma, beta: scale and offset from higher level featrues, learnable param, with shape [1,C,H-ss,W-ss]
	# ss: slide size for slideNorm 

	b,c,h,w = x.shape
	for yy in range(h-ss):
		for xx in range(w-ss):
			x_old  = x[:,:,yy:yy+ss,xx:xx+ss].clone()
			mean   = torch.mean(x_old,dim=[2,3],keepdim=True)
			var    = torch.std (x_old,dim=[2,3],keepdim=True)
			x_new  = (x_old-mean) / torch.sqrt(var+eps) * gamma[:,:,yy:yy+1,xx:xx+1]*gammaH[:,:,yy:yy+1,xx:xx+1] + beta[:,:,yy:yy+1,xx:xx+1]+betaH[:,:,yy:yy+1,xx:xx+1]
			
			x[:,:,yy:yy+ss,xx:xx+ss] = x_new
  
	return x

# Define models 2: feedforward model + plain fb #norm='spn'
class slidingPlainCNN(nn.Module):

	def __init__(self, hidden_dim=32, n_shapes=5,input_size=[224,224],norm='spn',slide_size=3,n_group=None,timesteps=4,device='cpu'):

		super(slidingPlainCNN, self).__init__()

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

		self.norm0_gamma = nn.Parameter(torch.ones ((1, 8,self.unit_size[0][1]-self.slide_size,self.unit_size[0][2]-self.slide_size)))
		self.norm0_beta  = nn.Parameter(torch.zeros((1, 8,self.unit_size[0][1]-self.slide_size,self.unit_size[0][2]-self.slide_size)))
		self.norm1_gamma = nn.Parameter(torch.ones ((1,16,self.unit_size[1][1]-self.slide_size,self.unit_size[1][2]-self.slide_size)))
		self.norm1_beta  = nn.Parameter(torch.zeros((1,16,self.unit_size[1][1]-self.slide_size,self.unit_size[1][2]-self.slide_size)))
		self.norm2_gamma = nn.Parameter(torch.ones ((1,32,self.unit_size[2][1]-self.slide_size,self.unit_size[2][2]-self.slide_size)))
		self.norm2_beta  = nn.Parameter(torch.zeros((1,32,self.unit_size[2][1]-self.slide_size,self.unit_size[2][2]-self.slide_size)))

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
				norm0_gammaH = torch.ones ((1, 8,self.unit_size[0][1]-self.slide_size,self.unit_size[0][2]-self.slide_size)).to(self.device)
				norm0_betaH  = torch.zeros((1, 8,self.unit_size[0][1]-self.slide_size,self.unit_size[0][2]-self.slide_size)).to(self.device)
				norm1_gammaH = torch.ones ((1,16,self.unit_size[1][1]-self.slide_size,self.unit_size[1][2]-self.slide_size)).to(self.device)
				norm1_betaH  = torch.zeros((1,16,self.unit_size[1][1]-self.slide_size,self.unit_size[1][2]-self.slide_size)).to(self.device)
			else:
				upsample_4norm0 = nn.Upsample(size=self.unit_size[0][1:],mode='bilinear',align_corners=True)
				upsample_4norm1 = nn.Upsample(size=self.unit_size[1][1:],mode='bilinear',align_corners=True)

				norm0_gammaH = upsample_4norm0(self.norm1_gamma)[:,::2,:,:]
				norm0_betaH  = upsample_4norm0(self.norm1_beta) [:,::2,:,:]
				norm1_gammaH = upsample_4norm1(self.norm2_gamma)[:,::2,:,:]
				norm1_betaH  = upsample_4norm1(self.norm2_beta) [:,::2,:,:]

			x = slideNorm_fb(input_x,self.norm0_gamma,self.norm0_beta,norm0_gammaH,norm0_betaH,ss=self.slide_size)
			x = F.relu(x)
			x = self.conv1(x)
			x = slideNorm_fb(      x,self.norm1_gamma,self.norm1_beta,norm1_gammaH,norm1_betaH,ss=self.slide_size)
			x = F.relu(x)
			x = self.conv2(x)
			x = slideNorm   (      x,self.norm2_gamma,self.norm2_beta,ss=self.slide_size)
			x = F.relu(x)

		x = torch.flatten(x, start_dim=1) # flatten tensor from channel dimension (torch tensor: b c w h)
		x = self.early_decoder(x)
		output_shape   = self.shape_decoder(x)
		output_vernier = self.vernier_decoder(x)      

		return output_shape, output_vernier
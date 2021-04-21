# Run file
import os, sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from math import isnan
from oh_specs import *
from models import *


def test(parser, norm='in'):

	# test trained kernels with surround suppression kernel

	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	btch_size   = 2
	n_blocks    = 10
	use_gpu     = parser.use_gpu

	n_hidden    = parser.n_hidden
	timesteps   = parser.timesteps
	learn_rate  = parser.learn_rate	
	train_type  = parser.train_type

	shape_types = parser.shape_types
	n_shapes 	= len(shape_types)
	input_size  = parser.input_size
	maxRow 		= parser.configRC[0]
	maxCol 		= parser.configRC[1]

	shapeMatrix = [[],[1],[1]*3,[[1]*3]*3,[6],[6]*3,[[6]*3]*3,\
	                  [3],[3]*3,[[3]*3]*3,\
	                  [2],[2]*3,[[2]*3]*3,\
	                  [6,1,6],[0,1,0],[0,6,0], \
	                  [[1,6,1],[6,1,6],[1,6,1]],[[1,0,1],[0,1,0],[1,0,1]],[[6,0,6],[0,6,0],[6,0,6]], \
	                  [[6,1,6]]*3,[[0,1,0]]*3,[[0,6,0]]*3]
	test_types  = ['ver','1sq','3sq','9sq','1st','3st','9st',\
	               '1hg','3hg','9hg',\
	               '1cc','3cc','9cc', \
	               '3sqst','3sqbl','3stbl',
	               '9msqst','9msqbl','9mstbl',\
	               '9asqst','9asqbl','9astbl']
	n_configs   = len(shapeMatrix)

	device = torch.device('cpu')

	if use_gpu and torch.cuda.is_available():
		torch.cuda.empty_cache()
		device = torch.device('cuda')

	# save model checkpoints 
	model_path  = './model_checkpoints/train_' + train_type + str(maxRow) + str(maxCol) + '/hidden_' + str(n_hidden) + '/n_shapes_' + str(n_shapes) + '/timesteps_'+str(timesteps) + '/lr_' + str(learn_rate) + '/norm_type_' + norm
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	# Only test fully trained networks
	saved_ckpt  = os.listdir(model_path)
	subjects    = [int(sc.split('_')[1].split('.')[0]) for sc in saved_ckpt if len(sc.split('_'))==2]
	print('Test model in model-path {} with {} \n\tsubjects are {}'.format(model_path, device, subjects))

	results     = np.zeros((len(subjects),n_configs,2))

	# test the trained networks
	for i,subj in enumerate(subjects):
		
		# initialize the model
		if 'uspn' in norm:
			print('Initialize '+norm+' model')
			model = unfoldSlidePlainNormCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
		elif 'in_fb' in norm:
			print('Initialize '+norm+' model')
			model = instanceNormfbCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
		elif 'inp_fb' in norm:
			print('Initialize '+norm+' model')
			model = patchInstanceNormfbCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
		elif 'ine' in norm:
			print('Initialize '+norm+' model')
			model = instanceNormEntropyCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
		else:
			print('Initialize ffCNN')
			model = feedforwardCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,device=device)

		losslogger      = OrderedDict([('train_errs', []),('val_errs', [])]) # save all errors 
		this_model_path = model_path +  '/subject_'+str(subj)+'.pt'
		
		if os.path.exists(this_model_path):
			model, _, __, losslogger = load_checkpoint(model, losslogger=losslogger, fname=this_model_path)

		if os.path.exists(os.path.join(model_path,'data_for_plots','all_test_errors_data.npy')):
			print('File exists')
			results = np.load(os.path.join(model_path,'data_for_plots','all_test_errors_data.npy'))
			continue


		if use_gpu and torch.cuda.is_available():
			device = torch.device('cuda')
			model  = model.to('cuda')

		n_test_errs = [[0]*2  for __ in range(n_configs)]

		for b in range(n_blocks):
			test_i, test_v, test_s = make_dataset(btch_size=btch_size, imgSize=input_size, device=device, type='test', shapeMatrix=shapeMatrix, type_size_list=[shape_types,maxRow,maxCol])

			for s in range(n_configs):
				if i==0 and b==0:
					vischannel_shapeType = s 
				else:
					vischannel_shapeType = None

				with torch.no_grad(): # testing: no gradient needed
					if norm is 'in_12_e':
						out_s, out_v = model.forward_ent(test_i[s])
					else:
						out_s, out_v = model(test_i[s])

					n_test_errs[s][0] += (out_s.argmax(1) != test_s[s]).sum()
					n_test_errs[s][1] += (out_v.argmax(1) != test_v[s]).sum()

					# plt.figure()
					# plt.imshow(test_i[s][0,0,:,:].cpu().numpy(),cmap='gray')
					# plt.savefig('./figs/shapeType'+str(s)+'_input.png',transparent=True,bbox_inches='tight', pad_inches=0)
					# plt.close()

		n_test_errs = [[100 * float(err) / (btch_size * n_blocks) for err in shape_errs] for shape_errs in n_test_errs]
		print('subject {}\'s errors are {}'.format(subj, n_test_errs))
		results[i,:,:] = np.array(n_test_errs)
		# plt.close('all')

	if not os.path.exists(os.path.join(model_path,'data_for_plots')):
		os.makedirs(os.path.join(model_path,'data_for_plots'))
	np.save(os.path.join(model_path,'data_for_plots','all_test_errors_data') , results)
	if not os.path.exists(os.path.join(model_path,'..','..','..','figs')):
		os.makedirs(os.path.join(model_path,'..','..','..','figs'))

	fig = plt.figure(figsize=(30,10))
	plt.plot(losslogger['train_errs'][:,0], label='shape')
	plt.plot(losslogger['train_errs'][:,1], label='vernier')
	plt.ylim((0,60))
	plt.legend()
	plt.title('Norm type: '+norm)
	fig.savefig(os.path.join(model_path,'..','..','..','figs','Train_errs_'+norm+'_timesteps_'+str(timesteps)+'.png'))

	plot_test_results(results, norm=norm, save_path=os.path.join(model_path,'..','..','..','figs','Test_'+norm+'_timesteps_'+str(timesteps)) )

	plt.close('all')

	return	


def train(parser, subj=0, norm='bn'):
	# Train the feedforward network with different normalization schemes

	# Training parameters
	n_epochs   = parser.n_epochs
	learn_rate = parser.learn_rate
	use_gpu    = parser.use_gpu
	n_hidden   = parser.n_hidden
	timesteps  = parser.timesteps
	btch_size  = parser.btch_size
	n_batches  = parser.n_batches

	train_type = parser.train_type

	device = torch.device('cpu')

	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	shape_types = parser.shape_types
	n_shapes 	= len(shape_types)
	input_size  = parser.input_size
	maxRow 		= parser.configRC[0]
	maxCol 		= parser.configRC[1]

	device = torch.device('cpu')

	if use_gpu and torch.cuda.is_available():
		device = torch.device('cuda')
		torch.cuda.empty_cache()

	# initialize the model
	if 'uspn' in norm:
		print('Initialize '+norm+' model')
		model = unfoldSlidePlainNormCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
	elif 'in_fb' in norm:
		print('Initialize '+norm+' model')
		model = instanceNormfbCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
	elif 'inp_fb' in norm:
		print('Initialize '+norm+' model')
		model = patchInstanceNormfbCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
	elif 'ine' in norm:
		print('Initialize '+norm+' model')
		model = instanceNormEntropyCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,timesteps=timesteps,device=device)
	else:
		print('Initialize ff CNN')
		model = feedforwardCNN(hidden_dim=n_hidden,n_shapes=n_shapes,input_size=input_size,norm=norm,device=device)

	model.to(device)
	optims     = torch.optim.Adam(model.parameters(), lr=learn_rate)
	crit       = nn.CrossEntropyLoss().to(device)
	losslogger = OrderedDict([('train_errs', np.zeros((2000,2))),('val_errs', [])]) # save all errors 
	start_epoch= 0

	# save model checkpoints 
	model_path  = './model_checkpoints/train_' + train_type + str(maxRow) + str(maxCol) + '/hidden_' + str(n_hidden) + '/n_shapes_' + str(n_shapes) + '/timesteps_'+str(timesteps) + '/lr_' + str(learn_rate) + '/norm_type_' + norm
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	this_model_path = model_path +  '/subject_'+str(subj)+'.pt'
	if os.path.exists(this_model_path):
		model, optims, start_epoch, losslogger = load_checkpoint(model, optims, losslogger, fname=this_model_path)
		if len(losslogger['train_errs']) < n_epochs:
			new_logger = np.zeros((2000,2))
			new_logger[:len(losslogger['train_errs']),:] = losslogger['train_errs']
			losslogger['train_errs'] = new_logger

	print('\n\nTraining subject # {} model {} in {} ...'.format(subj, norm,device))

	for e in range(start_epoch, n_epochs):

		e_start = time.time()
		n_train_errs = [0,0]

		for b in range(0, n_batches):

			train_i, train_v, train_s = make_dataset(btch_size=btch_size, device=device, type=train_type, imgSize=input_size, type_size_list=[shape_types,maxRow,maxCol])

			if device == 'CUDA':
			  torch.cuda.empty_cache()

			model.train()
			out_s, out_v = model(train_i)

			# loss: shape cross-entropy loss + vernier cross-emtropy loss
			loss = crit(out_v,train_v) + crit(out_s,train_s)
			n_train_errs[0] += (out_s.argmax(1) != train_s).sum()
			n_train_errs[1] += (out_v.argmax(1) != train_v).sum()

			if isnan(loss):
				print('Loss is Nan, quite procedure to search better initial weights')
				# rename current saved ckpt 
				this_model_path = model_path +  '/subject_'+str(subj)+'.pt'
				if os.path.exists(this_model_path):
					os.rename(this_model_path, os.path.join(model_path,'subject_'+str(subj)+'_epoch'+str(e)))
				return

			# Update the weights 
			model.zero_grad()
			loss.backward()
			optims.step()

		n_train_errs = [100 * float(err) / (btch_size * n_batches) for err in n_train_errs]
		losslogger['train_errs'][e,0] = n_train_errs[0]
		losslogger['train_errs'][e,1] = n_train_errs[1]


		# Print training and testing errors at each epochs
		print('\n\nSubject ' + str(subj) + ' - Epoch ' + str(e) + '\n')
		print( '  Train shape error: %5.2f %% - train vernier error: %5.2f %% - loss: %5.2f'% (n_train_errs[0], n_train_errs[1],loss))

		# Save the networks every 10 epochs
		e_end = time.time()
		if (e + 1) % 10 == 0:
			this_model_path = model_path + '/subject_'+str(subj)+'.pt'
			state = {'epoch'     : e+1,
					'state_dict': model.state_dict(),
					'optimizer' : optims.state_dict(),
					'losslogger': losslogger}
			torch.save(state, this_model_path)
			e_save = time.time()
			print('\nEpoch %2i  took %3d min %4.2f sec, and %3d min %4.2f sec to save cp' 
			    % (e,divmod(e_end-e_start,60)[0],divmod(e_end-e_start,60)[1],divmod(e_save-e_end,60)[0],divmod(e_save-e_end,60)[1]))
	return

def get_parser():
	parser = argparse.ArgumentParser(
	    prog='Vernier discrimination task with flexible DN networks',
	    usage='python main.py',
	    description='This module conducts vernier discrimination task by ene2end training and testing flexible DN networks',
	    add_help=True
	)

	parser.add_argument('-g', '--use_gpu', action='store_true', help='Using GPUs')
	parser.add_argument('-t', '--test', action='store_true', help='Test')
	parser.add_argument('-i', '--train', action='store_true', help='Train')
	parser.add_argument('-a', '--all_train_test', action='store_true', help='Train and test')
	parser.add_argument('-e', '--n_epochs', type=int, default=int(1000), help='Number of epochs')
	parser.add_argument('-s', '--n_subjs', type=int, default=5, help='Number of runs')
	parser.add_argument('-b', '--btch_size', type=int, default=32, help='Batch size')
	parser.add_argument('-n', '--n_batches', type=int, default=32*4, help='Batch size')
	parser.add_argument('-l', '--learn_rate', type=float, default=1e-4, help='Learning rate') 
	parser.add_argument('--shape_types', nargs='*', default=[1,2,3,6,7], help='shape types, defined in batch_maker')
	parser.add_argument('--input_size', nargs='*', default=[120,120], help='input shape size')
	parser.add_argument('--configRC', nargs='*', default=[3,3], help='configuration [maxRow, maxCol]')	
	parser.add_argument('--norm_types', nargs='*', default=['uspn'], help='normalization types')
	parser.add_argument('--n_hidden', type=int, default=32, help='num of hidden neurons')
	parser.add_argument('--timesteps', type=int, default=2, help='recurrent timesteps') 
	parser.add_argument('--train_type', type=str, default='twinShapeConfig', help='training dataset type') 	

	return parser

if __name__ == '__main__':
	parser = get_parser().parse_args()
	if parser.train:
		print('TRAIN norm_types... {}'.format(parser.norm_types))
		for norm in parser.norm_types:
			for subj in range(0,parser.n_subjs):
				train(parser,norm=norm, subj=subj)
	if parser.test:
		print('TEST ...')		
		for norm in parser.norm_types:
			test(parser,norm=norm)
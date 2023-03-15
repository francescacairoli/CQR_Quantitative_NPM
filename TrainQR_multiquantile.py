import os 
import pickle
import numpy as np

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from tqdm import tqdm

from QR import *

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class TrainQR():
	'''
	Class containing all the necessary methods to train a quantile regressor (QR) at level quantiles
	'''
	def __init__(self, model_name, dataset, idx = None, cal_hist_size = 50, test_hist_size = 2000, quantiles = [0.05, 0.95], opt = "Adam", n_hidden = 50, xavier_flag = False, scheduler_flag = False, drop_out_rate = 0.1):
		super(TrainQR, self).__init__()
		
		self.model_name = model_name
		self.dataset = dataset
		
		self.alpha = 0.1
		if idx:
			self.idx = idx
			self.models_path = "Models/"+self.model_name+"/ID_"+idx
			self.results_path = "Results/"+self.model_name+"/ID_"+idx
		else:
			rnd_idx = str(np.random.randint(0,100000))
			self.idx = rnd_idx
			self.models_path = "Models/"+self.model_name+"/ID_"+rnd_idx
			self.results_path = "Results/"+self.model_name+"/ID_"+rnd_idx
		os.makedirs(self.models_path, exist_ok=True)
		os.makedirs(self.results_path, exist_ok=True)

		self.cal_hist_size = cal_hist_size
		self.test_hist_size = test_hist_size

		self.valid_set_dim = 100

		self.quantiles = quantiles
		self.nb_quantiles = len(quantiles)

		self.opt = opt
		self.xavier_flag = xavier_flag
		self.n_hidden = n_hidden
		self.scheduler_flag = scheduler_flag
		self.drop_out_rate = drop_out_rate

	def pinball_loss(self, f_star, y, q_idx):

		n = len(y)
		loss = 0
		diff = y-f_star[:, q_idx]
		
		for i in range(n):
			if diff[i]>0:
				loss += self.quantiles[q_idx]*diff[i]
			else:
				loss += (self.quantiles[q_idx]-1)*diff[i]

		return loss/n


	def initialize(self):

		self.dataset.load_data()

		self.qr_model = QR(input_size = int(self.dataset.x_dim), output_size = self.nb_quantiles, hidden_size = self.n_hidden, xavier_flag = self.xavier_flag, drop_out_rate = self.drop_out_rate)

		if cuda:
			self.qr_model.cuda()


	def train(self, n_epochs, batch_size, lr):

		self.n_epochs = n_epochs
		self.batch_size = batch_size
		self.lr = lr

		

		if self.opt == "Adam":
			optimizer = torch.optim.Adam(self.qr_model.parameters(), lr=lr)
		else:
			optimizer = torch.optim.RMSprop(self.qr_model.parameters(), lr=lr)
		scheduler = ExponentialLR(optimizer, gamma=0.9)

		self.net_path = self.results_path+"/qr_{}epochs.pt".format(n_epochs)

		losses = []
		val_losses = []

		bat_per_epo = self.dataset.n_training_points // batch_size
		
		Xt_val = Variable(FloatTensor(np.repeat(self.dataset.X_cal, self.cal_hist_size, axis = 0)[:(self.valid_set_dim*self.cal_hist_size)]))
		Tt_val = Variable(FloatTensor(self.dataset.R_cal[:(self.valid_set_dim*self.cal_hist_size)]))
				
		for epoch in tqdm(range(n_epochs)):
						
			if (epoch+1) % 25 == 0:
				print("Epoch= {},\t loss = {:2.4f}".format(epoch+1, losses[-1]))

			tmp_val_loss = []
			tmp_loss = []

			for i in range(bat_per_epo):
				# Select a minibatch
				state, rob, sign, b_ix = self.dataset.generate_mini_batches(batch_size)
				Xt = Variable(FloatTensor(state))
				Tt = Variable(FloatTensor(rob))
				
				# initialization of the gradients
				optimizer.zero_grad()
				
				# Forward propagation: compute the output
				hypothesis = self.qr_model(Xt)

				# Computation of the loss
				loss = 0
				for q in range(self.nb_quantiles):
					loss += self.pinball_loss(hypothesis, Tt, q) # <= compute the loss function
				loss = loss/self.nb_quantiles
				
				val_loss = 0
				for qv in range(self.nb_quantiles):
					val_loss += self.pinball_loss(self.qr_model(Xt_val), Tt_val, qv)
				val_loss = val_loss/self.nb_quantiles

				# Backward propagation
				loss.backward() # <= compute the gradients
				
				# Update parameters (weights and biais)
				optimizer.step()

				# Print some performance to monitor the training
				tmp_loss.append(loss.item()) 
				tmp_val_loss.append(val_loss.item())

			if self.scheduler_flag:
				scheduler.step()	
			losses.append(np.mean(tmp_loss))
			val_losses.append(np.mean(tmp_val_loss))

		fig_loss = plt.figure()
		plt.plot(np.arange(n_epochs), losses, label="train", color="blue")
		plt.plot(np.arange(n_epochs), val_losses, label="valid", color="green")
		plt.title("QR loss")
		plt.legend()
		plt.tight_layout()
		fig_loss.savefig(self.models_path+"/qr_losses.png")
		plt.close()


	def save_model(self):
		self.net_path = self.models_path+"/qr_{}epochs.pt".format(self.n_epochs)
		torch.save(self.qr_model, self.net_path)


	def load_model(self, n_epochs):
		self.net_path = self.models_path+"/qr_{}epochs.pt".format(n_epochs)
		self.qr_model = torch.load(self.net_path)
		self.qr_model.eval()
		if cuda:
			self.qr_model.cuda()
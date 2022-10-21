import copy
import torch
import numpy as np
import scipy.special
import scipy.spatial
from numpy.random import rand
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
from torch.autograd import Variable

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class CQR():

	def __init__(self, Xc, Yc, trained_qr_model, test_hist_size = 2000, cal_hist_size = 50, quantiles = [0.05, 0.95], comb_flag = False):
		self.Xc = Xc 
		self.Yc = Yc
		self.trained_qr_model = trained_qr_model
		
		self.q = len(Yc) # number of points in the calibration set
		self.test_hist_size = test_hist_size
		self.cal_hist_size = cal_hist_size
		self.quantiles = quantiles
		self.epsilon = 2*quantiles[0]
		self.M = len(quantiles) # number of quantiles
		self.col_list = ['yellow', 'orange', 'red', 'orange', 'yellow']
		self.comb_flag = comb_flag

	def get_pred_interval(self, inputs):

		if not self.comb_flag:
			return self.trained_qr_model(Variable(FloatTensor(inputs))).cpu().detach().numpy()
		else:
			interval1 = self.trained_qr_model[0](Variable(FloatTensor(inputs))).cpu().detach().numpy() 
			interval2 = self.trained_qr_model[1](Variable(FloatTensor(inputs))).cpu().detach().numpy() 
			pis = []
			for i in range(inputs.shape[0]):
				# LB = min of the lbs 
				# UB = min of the ubs
				pis.append([min(interval1[i][0],interval2[i][0]), min(interval1[i][-1],interval2[i][-1])]) 

			return np.array(pis) 


	def get_calibr_nonconformity_scores(self, y, pred_interval, sorting = True):

		n = pred_interval.shape[0]
		m = len(y)
		ncm = np.empty(m)

		c = 0		
		for i in range(n):
			for j in range(self.cal_hist_size):
			
				ncm[c] = max(pred_interval[i,0]-y[c], y[c]-pred_interval[i,-1]) # pred_interval[i,0] = q_lo(x), pred_interval[i,1] = q_hi(x)
				c += 1	
		if sorting:
			ncm = np.sort(ncm)[::-1] # descending order
		return ncm


	def get_scores_threshold(self):
		self.calibr_pred = self.get_pred_interval(self.Xc)

		# nonconformity scores on the calibration set
		self.calibr_scores = self.get_calibr_nonconformity_scores(self.Yc, self.calibr_pred)

		Q = (1-self.epsilon)*(1+1/self.q)
		self.tau = np.quantile(self.calibr_scores, Q)

		print("self.tau: ", self.tau)


	def get_cpi(self, inputs, pi_flag = False):

		pi = self.get_pred_interval(inputs)
		self.get_scores_threshold()

		n_quant = pi.shape[1]
		cpi = pi[:,0]-self.tau
		for j in range(1,n_quant-1):
			cpi = np.vstack((cpi, pi[:,j]))
		cpi = np.vstack((cpi, pi[:,-1]+self.tau))
		if pi_flag:
			return cpi.T, pi
		else:
			return cpi.T


	def get_coverage_efficiency(self, y_test, test_pred_interval):

		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		c = 0
		for i in range(n_points):
			for j in range(self.test_hist_size):
				if y_test_hist[i,j] >= test_pred_interval[i,0] and y_test_hist[i, j] <= test_pred_interval[i,-1]:
					c += 1
		coverage = c/(n_points*self.test_hist_size)

		efficiency = np.mean(np.abs(test_pred_interval[:,-1]-test_pred_interval[:,0]))

		return coverage, efficiency

	def get_coverage_efficiency_coupled(self, y_test, test_pred_interval1, test_pred_interval2):

		n_points = len(y_test)//self.test_hist_size
		y_test_hist = np.reshape(y_test, (n_points, self.test_hist_size))
		c = 0
		for i in range(n_points):
			for j in range(self.test_hist_size):
				# if it lies in at least one of the two intervals, i.e. in the union
				if (y_test_hist[i,j] >= test_pred_interval1[i,0] and y_test_hist[i, j] <= test_pred_interval1[i,-1]) or (y_test_hist[i,j] >= test_pred_interval2[i,0] and y_test_hist[i, j] <= test_pred_interval2[i,-1]):
					c += 1
		coverage = c/(n_points*self.test_hist_size)

		efficiency = np.mean(self.measure_efficiency_union(test_pred_interval1,test_pred_interval2))

		return coverage, efficiency

	def measure_efficiency_union(self, I1, I2):
		L1 = I1[:,1]-I1[:,0]
		L2 = I2[:,1]-I2[:,0]
		
		union_len = []
		for i in range(len(I1)):
			if I1[i,0] < I2[i,0]: 
				len_intersection = I1[i,1]-I2[i,0]
				if len_intersection > 0:
					union_len.append(L1[i]+L2[i]-len_intersection)
				else:	 
					union_len.append(L1[i]+L2[i])
			else:
				len_intersection = I2[i,1]-I1[i,0]
				if len_intersection > 0:
					union_len.append(L1[i]+L2[i]-len_intersection)
				else:	 
					union_len.append(L1[i]+L2[i])
		return np.array(union_len)

	def compute_accuracy_and_uncertainty(self, test_pred_interval, L_test):

		n_points = len(L_test)#//self.test_hist_size

		#L_test_hist = np.reshape(L_test, (n_points, self.test_hist_size))

		correct = 0
		wrong = 0
		uncertain = 0
		fp = 0

		for i in range(n_points):
			
			if L_test[i,2]: # sign +1
				if test_pred_interval[i,0] >= 0 and test_pred_interval[i,-1] > 0:
					correct += 1
				elif test_pred_interval[i,0] <= 0 and test_pred_interval[i,-1] >= 0:
					uncertain += 1
				else:
					wrong +=1
			elif L_test[i,1]: # sign 0
				if test_pred_interval[i,0] <= 0 and test_pred_interval[i,-1] >= 0:
					correct += 1
				else:
					wrong +=1
					if test_pred_interval[i,0] > 0:
						fp+= 1
			else: # sign -1
				if test_pred_interval[i,-1] <= 0 and test_pred_interval[i,0] < 0:
					correct += 1
				elif test_pred_interval[i,-1] >= 0 and test_pred_interval[i,0] <= 0:
					uncertain += 1
				else:
					wrong +=1
					fp += 1

		return correct/n_points, uncertain/n_points, wrong/n_points, fp/n_points


	def plot_results(self, y_test, test_pred_interval, title_string, plot_path, extra_info = ''):

		n_quant = test_pred_interval.shape[1]

		n_points_to_plot = test_pred_interval.shape[0]
		xline = np.arange(n_points_to_plot)
		xline_rep = np.repeat(xline, self.test_hist_size)
		
		fig = plt.figure(figsize=(20,4))
		plt.scatter(xline_rep, y_test[:n_points_to_plot*self.test_hist_size], c='b', s=1, alpha = 0.1)
		if n_quant > 2:
			for i in range(1,n_quant-1):
				plt.plot(xline, test_pred_interval[:n_points_to_plot,i], 'r', alpha = 0.4)
		plt.plot(xline, np.zeros(n_points_to_plot), '-.', color='b', alpha = 0.2)
		plt.fill_between(xline, test_pred_interval[:n_points_to_plot,0], test_pred_interval[:n_points_to_plot,-1], color = 'r', alpha = 0.1)
		plt.title(title_string)
		plt.tight_layout()
		fig.savefig(plot_path+"/"+title_string+extra_info+".png")
		plt.close()
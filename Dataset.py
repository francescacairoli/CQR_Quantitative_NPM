import pickle
import numpy as np


class Dataset():
	'''
	Class containing all the data pre-processing steps for the train, calibration and test data and to create mini-batches of training data
	'''
	def __init__(self, property_idx, comb_flag, trainset_fn, testset_fn, calibrset_fn, alpha = 0.1, n_train_states = 2000, n_cal_states = 1000, n_test_states = 100, hist_size = 50, test_hist_size = 2000, multiplier = 1):
		super(Dataset, self).__init__()

		self.trainset_fn = trainset_fn
		self.testset_fn = testset_fn
		self.calibrset_fn = calibrset_fn
		self.n_train_states = n_train_states
		self.n_cal_states = n_cal_states
		self.n_test_states = n_test_states
		self.hist_size = hist_size
		self.test_hist_size = test_hist_size
		self.alpha = alpha
		self.multiplier = multiplier
		self.property_idx = property_idx
		self.comb_flag = comb_flag
		
	def load_data(self):
		
		self.load_train_data()
		eqr_width = self.load_test_data()
		self.load_calibration_data()

		return eqr_width
		
	def load_train_data(self):

		file = open(self.trainset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.X_train = data["x_scaled"]
		
		if self.property_idx == -1: #-1 denotes the property wrt all variables
			self.R_train = self.multiplier*data["rob"]
		else:
			if self.comb_flag:
				self.R_train = self.multiplier*data["couple_robs"][self.property_idx]
			else:
				self.R_train = self.multiplier*data["single_robs"][self.property_idx]
			
		self.x_dim = self.X_train.shape[1]
		self.n_training_points = self.X_train.shape[0]
		
		R_train_hist = np.reshape(self.R_train, (self.n_train_states, self.hist_size))
		
		L_train = np.zeros((self.n_train_states,3)) #{-1,0,1}
		C_train = np.empty(self.n_train_states)
		for i in range(self.n_train_states):
			lower_yi = np.quantile(R_train_hist[i], self.alpha/2)
			upper_yi = np.quantile(R_train_hist[i], 1-self.alpha/2)

			if lower_yi >= 0:
				L_train[i, 2] = 1
				C_train[i] = 2
			elif upper_yi <= 0:
				L_train[i, 0] = 1
				C_train[i] = 0
			else:
				L_train[i, 1] = 1
				C_train[i] = 1

		self.L_train = L_train
		self.C_train = C_train
			
		
	def load_test_data(self):

		file = open(self.testset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.X_test = data["x_scaled"]

		if self.property_idx == -1: #-1 denotes the property wrt all variables
			self.R_test = self.multiplier*data["rob"]
		else:
			if self.comb_flag:
				self.R_test = self.multiplier*data["couple_robs"][self.property_idx]
			else:
				self.R_test = self.multiplier*data["single_robs"][self.property_idx]		
		self.n_test_points = self.X_test.shape[0]
		
		R_test_hist = np.reshape(self.R_test, (self.n_test_states, self.test_hist_size))
		
		L_test = np.zeros((self.n_test_states,3)) #{-1,0,1}
		C_test = np.empty(self.n_test_states)
		
		widths = np.empty(self.n_test_states)
		for i in range(self.n_test_states):
			lower_yi = np.quantile(R_test_hist[i], self.alpha/2)
			upper_yi = np.quantile(R_test_hist[i], 1-self.alpha/2)

			widths[i] = upper_yi-lower_yi
			if lower_yi >= 0:
				L_test[i, 2] = 1
				C_test[i] = 2
			elif upper_yi <= 0:
				L_test[i, 0] = 1
				C_test[i] = 0
			else:
				L_test[i, 1] = 1
				C_test[i] = 1
		eqr_width = np.mean(widths)
		self.L_test = L_test
		self.C_test = C_test

		return eqr_width

	def load_calibration_data(self):

		file = open(self.calibrset_fn, 'rb')
		data = pickle.load(file)
		file.close()

		self.X_cal = data["x_scaled"]
		
		if self.property_idx == -1: #-1 denotes the property wrt all variables
			self.R_cal = self.multiplier*data["rob"]
		else:
			if self.comb_flag:
				self.R_cal = self.multiplier*data["couple_robs"][self.property_idx]
			else:
				self.R_cal = self.multiplier*data["single_robs"][self.property_idx]
		self.n_cal_points = self.X_cal.shape[0]

		R_cal_hist = np.reshape(self.R_cal, (self.n_cal_states, self.hist_size))
		
		L_cal = np.zeros((self.n_cal_states,3)) #{-1,0,1}
		C_cal = np.empty(self.n_cal_states)
		for i in range(self.n_cal_states):
			lower_yi = np.quantile(R_cal_hist[i], self.alpha/2)
			upper_yi = np.quantile(R_cal_hist[i], 1-self.alpha/2)

			if lower_yi >= 0:
				L_cal[i, 2] = 1
				C_cal[i] = 2 
			elif upper_yi <= 0:
				L_cal[i, 0] = 1
				C_cal[i] = 0
			else:
				L_cal[i, 1] = 1
				C_cal[i] = 1
		self.L_cal = L_cal
		self.C_cal = C_cal
		
	def generate_mini_batches(self, n_samples):
		
		n_trajs = len(self.R_train)//len(self.X_train)
		X_rep = np.repeat(self.X_train, n_trajs, axis = 0)
		L_rep = np.repeat(self.L_train, n_trajs, axis = 0)
		C_rep = np.repeat(self.C_train, n_trajs, axis = 0)
		ix = np.random.randint(0, len(self.R_train), n_samples)
		Xb = X_rep[ix]
		Rb = self.R_train[ix]
		Lb = L_rep[ix]
		Cb = C_rep[ix]

		return Xb, Rb, Cb, ix

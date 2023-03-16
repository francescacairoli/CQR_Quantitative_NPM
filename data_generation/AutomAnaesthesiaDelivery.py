import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from scipy.integrate import odeint, solve_ivp
from scipy.stats import norm, expon, bernoulli, multivariate_normal
import math
from tqdm import tqdm
import pickle
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries
from simple_pid import PID

class AutomatedAnaesthesiaDelivery(object):
	'''
	AUTOMATED ANAESTHESIA DEVILVERY BENCHMARK (ARCH-COMP18 SM)
	Linear 3-compartments stochastic system, discrete-time

	'''
	def __init__(self):
		self.model_name = "AAD"
		self.system_matrix = np.array([[0.8192, 0.03412, 0.01265], [0.01646, 0.9822, 0.0001], [0.0009, 0.00002,0.9989]])
		self.input_matrix = np.array([0.01883, 0.0002,0.00001])
		self.dt = 20 #seconds
		self.input_values = [3.5, 7]
		self.state_dim = 3
		self.dist_mean = np.zeros(3)
		self.dist_cov = 0.001*np.eye(3)
		self.time_horizon = 60 #minutes
		self.safe_ranges = np.array([[1, 6], [0, 10], [0, 10]])
		self.state_labels = ["$x_1$", "$x_2$", "$x_3$"]


	def initialize_settings(self):
		self.timeline = np.arange(start=0, stop=self.time_horizon, step=self.dt/60)
		self.n_steps = len(self.timeline)
		self.target_values = np.mean(self.safe_ranges[0])
		
		
	def dynamics(self, x):
		w = multivariate_normal.rvs(mean=self.dist_mean, cov=self.dist_cov)
		u = self.controller(x[0])
		next_x = self.system_matrix.dot(x) + u*self.input_matrix + w

		return next_x


	def controller(self, x1):

		if x1 < 3.5:
			return self.input_values[1]
		else:
			return self.input_values[0] 



	
	def sample_rnd_states(self, n_samples):

		states = np.empty((n_samples, self.state_dim))
		for i in range(n_samples):
			#print("Point {}/{}".format(i+1,n_samples))
			
			states[i] = self.safe_ranges[:,0]+(self.safe_ranges[:,1]-self.safe_ranges[:,0])*np.random.rand(self.state_dim)

		return states


	def gen_trajectories(self, states, n_trajs_per_state):

		n_samples = len(states)
		
		trajs = np.empty((n_samples*n_trajs_per_state, self.n_steps, self.state_dim))
		
		c = 0
		for i in range(n_samples):
			print("Point {}/{}".format(i+1, n_samples))
			for j in range(n_trajs_per_state):
				trajs[c,0] = states[i]

				for t in range(1, self.n_steps):
					trajs[c, t] = self.dynamics(trajs[c, t-1])
				c += 1
				
		return trajs


	def set_goal(self, formula):
		self.phi = formula


	def plot_trajectories(self, trajs):

		n_points = len(trajs)
		
		for i in range(n_points):
			fig, axs = plt.subplots(self.state_dim,1, figsize=(12, 9))
			for s in range(self.state_dim):
				axs[s].plot(self.timeline, trajs[i, :, s])
				axs[s].set_ylabel(self.state_labels[s])
				axs[s].plot(self.timeline, self.safe_ranges[s,0]*np.ones(self.n_steps), '--', c='r')
				axs[s].plot(self.timeline, self.safe_ranges[s,1]*np.ones(self.n_steps), '--', c='r')
			axs[-1].set_xlabel("time")
			plt.tight_layout()
			plt.savefig(self.model_name+"/trajs_{}.png".format(i))
			plt.close()

	def compute_robustness(self, trajs):
		n_states = len(trajs)
		robs = np.empty(n_states)
		 
		for i in tqdm(range(n_states)):
			#print("rob ", i+1, "/", n_states)
			time_series_i = TimeSeries(['X1', 'X2', 'X3'], self.timeline, trajs[i].T)
			robs[i] = stlRobustSemantics(time_series_i, 0, self.phi)

		return robs

if __name__=='__main__':

	ad_model = AutomatedAnaesthesiaDelivery()
	ad_model.initialize_settings()
	nb_points = 200
	nb_trajs_per_state = 500
	dataset_type = 'test'


	states = ad_model.sample_rnd_states(nb_points)
	trajs = ad_model.gen_trajectories(states, nb_trajs_per_state)

	#ad_model.plot_trajectories(trajs)

	if dataset_type == 'train':
		xmax = np.max(np.max(trajs, axis = 0), axis = 0)
		xmin = np.min(np.min(trajs, axis = 0), axis = 0)
	else:
		trainset_fn = '../Datasets/EHT2_train_set_{}x{}points.pickle'.format(nb_points, nb_trajs_per_state)

		with open(trainset_fn, 'rb') as handle:
			train_data = pickle.load(handle)
		handle.close()
		xmin, xmax = train_data["x_minmax"]

	trajs_scaled = -1+2*(trajs-xmin)/(xmax-xmin)
	states_scaled = -1+2*(states-xmin)/(xmax-xmin)

	scaled_safety_region = -1+2*(ad_model.safe_ranges.T-xmin)/(xmax-xmin)

	goal_formula_scaled = '( ( ( G_[0,{}]  ( (X1 <= {}) & (X1 >= {}) ) ) & ( G_[0,{}]  ( (X2 <= {}) & (X2 >= {}) ) ) ) & ( G_[0,{}]  ( (X3 <= {}) & (X3 >= {}) ) )     ) '.format(ad_model.time_horizon, scaled_safety_region[1,0], scaled_safety_region[0,0], ad_model.time_horizon, scaled_safety_region[1,1], scaled_safety_region[0,1], ad_model.time_horizon, scaled_safety_region[1,2], scaled_safety_region[0,2])
	ad_model.set_goal(goal_formula_scaled)

	robs = ad_model.compute_robustness(trajs_scaled)

	print("Percentage of positive: ", np.sum((robs >= 0))/len(robs))

	
	dataset_dict = {"rob": robs, "x_scaled": states_scaled, "trajs_scaled": trajs_scaled, "x_minmax": (xmin,xmax)}

	filename = '../Datasets/AAD_'+dataset_type+'_set_{}x{}points.pickle'.format(nb_points, nb_trajs_per_state)
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)
	
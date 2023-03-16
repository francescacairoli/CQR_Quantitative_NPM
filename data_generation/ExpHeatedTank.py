import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from scipy.integrate import odeint, solve_ivp
from scipy.stats import norm, expon, bernoulli
import math
import pickle
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from pcheck.series.TimeSeries import TimeSeries

class ExpHeatedTank(object):
	'''
	HEATED TANK BENCHMARK (ARCH-COMP18 SM)

	Tank of liquid whose level is influenced by two inflow pumps and one outflow valve. 
	Deterministic flow dynamics (constant flows) and deterministic switching controller = hybrid system.

	Stochastically distributed (exponential) failures of pumps and valve make the overall system a Stochastic Hybrid System (SHS)

	Task: transport heat from a heat source
	Safety: avoid dryout, overflow and overheat

	'''
	def __init__(self):
		
		self.model_name = "EHT2"
		self.state_dim = 2
		self.q = 0.6 # constant flow
		self.H_dryout = -3 #L=4
		self.H_overflow = 3 #L=10
		self.H_high = 1 #L=8
		self.H_low = -1 #L=6

		self.T_overheating = 100
		self.t_end = 48
		self.t_init = 0
		self.beta_P1 = 438/2 # avg time to failure
		self.beta_P2 = 350/2 
		self.beta_V = 640/2
		self.T_in = 15 # infow temperature
		self.E_in = 1 # heat source parameter
		self.control_config = {"Normal":0, "Increase":1, "Decrease":-1}
		self.working_config = ["Off","On"]
		self.failure_config = ["StuckOff","StuckOn"]
		self.utils_config = {"On": 1, "Off": 0, "StuckOn": 1, "StuckOff": 0}
		self.state_labels = ["height", "temperature"]


	def initial_settings(self):
		self.C = self.control_config["Normal"]
		self.P1_config = self.working_config[1]
		self.P2_config = self.working_config[0]
		self.V_config = self.working_config[1]
		self.set_util_indicators()
		self.n_steps = (self.t_end-self.t_init)*4
		self.tspan = [self.t_init, self.t_end]#np.linspace(self.t_init, self.t_end, self.n_steps)
		self.tgrid = np.linspace(self.t_init, self.t_end, self.n_steps)
		self.lambda_P1 = 1/self.beta_P1 # scale param
		self.lambda_P2 = 1/self.beta_P2
		self.lambda_V = 1/self.beta_V

	def set_goal(self, formula):
		self.phi = formula
	
	def sample_initial_states(self, nb_points):

		x_H = np.random.uniform(self.H_dryout, self.H_overflow, nb_points)
		x_T = np.random.uniform(self.T_in, self.T_overheating, nb_points)

		x = np.vstack([x_H, x_T]).T

		return x


	def sample_utils_state(self):

		self.P1_config = self.working_config[np.random.randint(0, 2)]
		self.P2_config = self.working_config[np.random.randint(0, 2)]
		self.V_config = self.working_config[np.random.randint(0, 2)]

		self.set_util_indicators()

	def initialize_utils_state(self):
		self.P1_config = self.working_config[1]
		self.P2_config = self.working_config[0]
		self.V_config = self.working_config[1]
		self.set_util_indicators()


	def differential_equations(self, t, x):
		# x = (x_H, x_T)
		dxdt = np.zeros(self.state_dim)

		self.switch_controller(x)
		self.failure_events(t)
		
		dxdt[0] = (self.P1_indicator+self.P2_indicator-self.V_indicator)*self.q
		dxdt[1] = ( (self.P1_indicator+self.P2_indicator)*(self.T_in-x[1])*self.q+self.E_in )/(x[0]+7)

		return dxdt


	def switch_controller(self, x):

		if x[0] >= self.H_high:
			self.C = self.control_config["Decrease"]

		if x[0] <= self.H_low:
			self.C = self.control_config["Increase"]

		if self.C == 1: # Increase
			if self.P1_config not in self.failure_config:
				self.P1_config = "On"
			if self.P2_config not in self.failure_config:
				self.P2_config = "On"
			if self.V_config not in self.failure_config:
				self.V_config = "Off"

		if self.C == -1: # Decrease
			if self.P1_config not in self.failure_config:
				self.P1_config = "Off"
			if self.P2_config not in self.failure_config:
				self.P2_config = "Off"
			if self.V_config not in self.failure_config:
				self.V_config = "On"

		self.set_util_indicators()


	def set_util_indicators(self):

		self.P1_indicator = self.utils_config[self.P1_config]
		self.P2_indicator = self.utils_config[self.P2_config]
		self.V_indicator = self.utils_config[self.V_config]


	def sample_failures_times(self):

		t_P1 = np.random.exponential(scale = self.beta_P1, size = 2)
		t_P2 = np.random.exponential(scale = self.beta_P2, size = 2)
		t_V = np.random.exponential(scale = self.beta_V, size = 2)
		#print("FAILURE TIME = ", [t_P1, t_P2, t_V])
		return [t_P1, t_P2, t_V]



	def failure_events(self, curr_t):

		nb_failures = 2
		nb_utils = 3
		
		# events: 0: stuckon
		#		  1: stuckoff
		if curr_t > self.failure_times[0][0] and self.P1_config not in self.failure_config:
			#print("--- Failure happened: P1 StuckOn", P1_surv_prob_t)
			self.P1_config = "StuckOn"
		if curr_t > self.failure_times[1][0] and self.P2_config not in self.failure_config:
			#print("--- Failure happened: P2 StuckOn", P2_surv_prob_t)
			self.P2_config = "StuckOn"
		if curr_t > self.failure_times[2][0] and self.V_config not in self.failure_config:
			#print("--- Failure happened: V StuckOn", V_surv_prob_t)
			self.V_config = "StuckOn"
		if curr_t > self.failure_times[0][1] and self.P1_config not in self.failure_config:
			#print("--- Failure happened: P1 StuckOff", P1_surv_prob_t)
			self.P1_config = "StuckOff"
		if curr_t > self.failure_times[1][1] and self.P2_config not in self.failure_config:
			#print("--- Failure happened: P2 StuckOff", P2_surv_prob_t)
			self.P2_config = "StuckOff"
		if curr_t > self.failure_times[2][1] and self.V_config not in self.failure_config:
			#print("--- Failure happened: V StuckOff", V_surv_prob_t)
			self.V_config = "StuckOff"

		self.set_util_indicators()			


	def generate_trajectories(self, states, n_trajs_per_state):

		nb_samples = len(states)

		trajs = np.zeros((nb_samples*n_trajs_per_state, self.n_steps, self.state_dim))
		
		c = 0
		for i in range(nb_samples):
			print("Point {}/{}".format(i+1, nb_samples))
			x0 = states[i]
			
			for j in range(n_trajs_per_state):
				self.failure_times = self.sample_failures_times()
				#self.sample_utils_state()
				self.initialize_utils_state()
				sol = solve_ivp(self.differential_equations, self.tspan, x0, method = 'RK45',  t_eval = self.tgrid) 
				trajs[c] = sol.y.T
				c += 1

		return trajs


	def plot_trajectories(self, trajs):

		n_points = len(trajs)
		
		for i in range(n_points):
			fig, axs = plt.subplots(self.state_dim,1, figsize=(12, 9))
			for s in range(self.state_dim):
				axs[s].plot(self.tgrid, trajs[i, :, s])
				axs[s].set_ylabel(self.state_labels[s])
			axs[0].plot(self.tgrid, self.H_dryout*np.ones(self.n_steps), '--', c='r')
			axs[0].plot(self.tgrid, self.H_overflow*np.ones(self.n_steps), '--', c='r')
			axs[1].plot(self.tgrid, self.T_overheating*np.ones(self.n_steps), '--', c='r')
			axs[1].set_xlabel("time")
			plt.tight_layout()
			plt.savefig(self.model_name+"/trajs_{}.png".format(i))
			plt.close()


	def plot_scaled_trajectories(self, trajs, h_lb, h_ub, t_ub):

		n_points = len(trajs)
		
		for i in range(n_points):
			fig, axs = plt.subplots(self.state_dim,1, figsize=(12, 9))
			for s in range(self.state_dim):
				axs[s].plot(self.tgrid, trajs[i, :, s])
				axs[s].set_ylabel(self.state_labels[s])
			axs[0].plot(self.tgrid, h_lb*np.ones(self.n_steps), '--', c='r')
			axs[0].plot(self.tgrid, h_ub*np.ones(self.n_steps), '--', c='r')
			axs[1].plot(self.tgrid, t_ub*np.ones(self.n_steps), '--', c='r')
			axs[1].set_xlabel("time")
			plt.tight_layout()
			plt.savefig(self.model_name+"/scaled_trajs_{}.png".format(i))
			plt.close()


	def compute_robustness(self, trajs):
		n_states = len(trajs)
		robs = np.empty(n_states)
		 
		for i in range(n_states):
			time_series_i = TimeSeries(['H','T'], self.tgrid, trajs[i].T)
			robs[i] = stlRobustSemantics(time_series_i, 0, self.phi)

		return robs

if __name__=='__main__':

	
	ht_model = ExpHeatedTank()
	ht_model.initial_settings()
	nb_points = 2000
	nb_trajs_per_state = 50
	dataset_type = 'train'

	states = ht_model.sample_initial_states(nb_points)
	trajs = ht_model.generate_trajectories(states, nb_trajs_per_state)
	
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

	scaled_dryout = -1+2*(ht_model.H_dryout-xmin[0])/(xmax[0]-xmin[0])
	scaled_overflow = -1+2*(ht_model.H_overflow-xmin[0])/(xmax[0]-xmin[0])
	scaled_overheat = -1+2*(ht_model.T_overheating-xmin[1])/(xmax[1]-xmin[1])
	
	goal_formula_scaled = '( G_[0,{}] ( ( (H <= {}) & (H >= {}) ) & (T <= {}) ) )'.format(ht_model.t_end, scaled_overflow, scaled_dryout, scaled_overheat)
	ht_model.set_goal(goal_formula_scaled)
	#ht_model.plot_scaled_trajectories(trajs_scaled, scaled_dryout, scaled_overflow, scaled_overheat)
	
	robs = ht_model.compute_robustness(trajs_scaled)
	print(robs)
	print("Percentage of positive: ", np.sum((robs >= 0))/len(robs))

	
	dataset_dict = {"rob": robs, "x_scaled": states_scaled, "trajs_scaled": trajs_scaled, "x_minmax": (xmin,xmax)}

	filename = '../Datasets/EHT2_'+dataset_type+'_set_{}x{}points.pickle'.format(nb_points, nb_trajs_per_state)
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)
	
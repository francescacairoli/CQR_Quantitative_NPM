import math
import time
import copy
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import room_heating_utils as utils
from scipy.integrate import odeint, solve_ivp
from pcheck.series.TimeSeries import TimeSeries
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from scipy.stats import norm, expon, bernoulli, multivariate_normal 



class MultiRoomHeating(object):
	'''
	Multi-Room Heating System (DTSHS from Abate et al.)
	h = number of rooms
	xa = ambient temperature (season dependent)
	bi = avg heat transfer from room i to ambient
	aij = avg heat transfer from room i to room j
		aij = 0 iff room i and j are non-adjacent
	ci = heat rate supplied to room i by hetaer i
	nu = distrurbance std
	dt = temporal interval
	N = nb of temporal steps
	'''
	def __init__(self, nb_rooms):
		self.model_name = "MRH"
		self.h = nb_rooms
		self.state_dim = 2*nb_rooms
		self.nu = 0.25
		self.dt = 0.25 #min 
		self.N = 100
		self.xa = 6 # season dependent
		self.var_names = [f'X{k}' for k in range(nb_rooms)]

	def rooms_layout(self, params):

		np.fill_diagonal(params["A"], 0)
		self.Sigma = copy.deepcopy(params["A"])
		self.Gamma = params["B"]*self.xa
		np.fill_diagonal(self.Sigma, -params["B"] - np.sum(params["A"],axis=1))
		self.C = params["C"]

		self.ranges = params["ranges"]
		self.safe_ranges = params["safe_ranges"]

		self.sigmoid_fnc = lambda x: [(x[i]**(params["steepnesses"][i]))/(params["thresholds"][i]**(params["steepnesses"][i])+x[i]**(params["steepnesses"][i])) for i in range(self.h)]

	def initialize_settings(self, params):
		
		self.final_time = self.N*self.dt
		self.timeline = np.arange(start=0, stop=self.final_time, step=self.dt)
		self.rooms_layout(params)
		
	def continuous_dynamics(self, x, q):

		# qi = 1 if heater i is ON
		# qi = 0 if heater i is OFF

		x_mean = x+np.dot(self.Sigma, x)+self.Gamma+self.C*q
		x_cov = (self.nu**2)*np.eye(self.h)

		next_x = multivariate_normal.rvs(mean=x_mean, cov=x_cov)
		
		return next_x


	def discrete_dynamics(self, x, q):

		S = self.sigmoid_fnc(x)
		#sampling from [S,1-S] for [OFF,ON]
		next_q = [np.random.choice(2,p=[S[i], 1-S[i]]) for i in range(self.h)]

		return next_q

	def dynamics(self, state):
		#state = [x,q]
		next_x = self.continuous_dynamics(state[:self.h],state[self.h:])
		next_q = self.discrete_dynamics(state[:self.h],state[self.h:])
		
		return np.concatenate((next_x, next_q))

	def sample_rnd_states(self, n_samples):

		states_list = []
		for i in range(n_samples):
			#print("Point {}/{}".format(i+1,n_samples))
			x_rnd = self.ranges[:,0]+(self.ranges[:,1]-self.ranges[:,0])*np.random.rand(self.h)
			q_rnd = np.random.choice(2, size=self.h)

			rnd_state = np.hstack((x_rnd,q_rnd))
			states_list.append(rnd_state)

		return states_list


	def gen_trajectories(self, states, n_trajs_per_state):

		n_samples = len(states)
		self.n_samples = n_samples
		self.n_trajs_per_state = n_trajs_per_state
		trajs = np.empty((n_samples*n_trajs_per_state, self.N, 2*self.h))
		
		c = 0
		for i in tqdm(range(n_samples)):
			for j in range(n_trajs_per_state):
				trajs[c,0] = states[i]

				for t in range(1, self.N):
					trajs[c, t] = self.dynamics(trajs[c, t-1])
				c += 1
				
		return trajs


	def set_goal(self, formula):
		self.phi = formula


	def plot_trajectories(self, trajs, ds_name = ""):

		trajs_reshaped = np.reshape(trajs, (self.n_samples, self.n_trajs_per_state, self.N, self.state_dim))
		self.room_labels = [f'room {i}' for i in range(self.h)]
		self.heater_labels = [f'heater {i}' for i in range(self.h)]
		colors = ['r','b','g','y','k','c','m','tab:purple']
		for i in range(self.n_samples):
			fig, axs = plt.subplots(self.h,2, figsize=(12, 9))
			for s in range(self.h):
				for j in range(1):#self.n_trajs_per_state
					axs[s][0].plot(self.timeline, trajs_reshaped[i, j, :, s], c=colors[s])
					axs[s][0].set_ylabel(self.room_labels[s])
					axs[s][0].plot(self.timeline, self.safe_ranges[s,0]*np.ones(self.N), '--', c=colors[s])
					axs[s][0].plot(self.timeline, self.safe_ranges[s,1]*np.ones(self.N), '--', c=colors[s])
					axs[s][1].step(self.timeline, trajs_reshaped[i, j, :, s+self.h], c=colors[s])
					axs[s][1].set_ylabel(self.heater_labels[s])
				
			axs[-1][0].set_xlabel("time")
			axs[-1][1].set_xlabel("time")
			plt.tight_layout()
			plt.savefig(self.model_name+"/"+ds_name+f"_trajs_{i}.png")
			plt.close()

	def compute_robustness(self, trajs):
		n_states = len(trajs)
		robs = np.empty(n_states)
		
		
		for i in tqdm(range(n_states)):
			time_series_i = TimeSeries(self.var_names, self.timeline, trajs[i,:,:self.h].T)
			robs[i] = stlRobustSemantics(time_series_i, 0, self.phi)

		return robs



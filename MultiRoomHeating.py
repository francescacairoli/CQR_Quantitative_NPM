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

if __name__=='__main__':

	nb_rooms = 2

	datasets = ['train', 'calibration','test']
	nb_points = [1000*nb_rooms, 500*nb_rooms, 100*nb_rooms]
	nb_trajs_per_state = [50, 50, 500]
	nb_points = [1000*nb_rooms, 500*nb_rooms, 100*nb_rooms]
	nb_trajs_per_state = [50, 50, 500]
	for ds in range(len(datasets)):

		model = MultiRoomHeating(nb_rooms)
		params = utils.get_parameters(nb_rooms)
		model.initialize_settings(params)

		print(nb_points[ds])
		states = model.sample_rnd_states(nb_points[ds])
		start_time = time.time()
		trajs = model.gen_trajectories(states, nb_trajs_per_state[ds])
		end_time = time.time()-start_time
		print(trajs.shape)
		print(f'Time needed to generate {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
		#model.plot_trajectories(trajs, datasets[ds])

		xmax = np.max(np.max(trajs, axis = 0), axis = 0)
		xmin = np.min(np.min(trajs, axis = 0), axis = 0)
		print("--- xmin, xmax = ", xmin, xmax)
		
		trajs_scaled = -1+2*(trajs-xmin)/(xmax-xmin)
		states_scaled = -1+2*(states-xmin)/(xmax-xmin)

		scaled_safety_region = -1+2*(model.safe_ranges.T-xmin[:nb_rooms])/(xmax[:nb_rooms]-xmin[:nb_rooms])
		print("scaled_safety_region = ", scaled_safety_region)
		goal_formula_scaled = utils.get_safety_property(nb_rooms, model.final_time,scaled_safety_region.T)
		
		model.set_goal(goal_formula_scaled)
		start_time = time.time()
		robs = model.compute_robustness(trajs_scaled)
		end_time = time.time()-start_time
		print(f'Time need to label (all rooms) {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
			
		print("FULL SAFETY Percentage of positive: ", np.sum((robs >= 0))/len(robs))
		
		roomspec_props = [utils.get_roomspec_safety_property(i, model.final_time,scaled_safety_region.T) for i in range(nb_rooms)]
		roomspec_robs = []
		roomcomb_robs = []
		for i in range(nb_rooms):
			model.set_goal(roomspec_props[i])
			start_time = time.time()
			roomspec_robs.append(model.compute_robustness(trajs_scaled))
			end_time = time.time()-start_time
			print(f'Time need to label (single room) {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
			print(f"SAFETY of Room {i} Percentage of positive: ", np.sum((roomspec_robs[-1] >= 0))/len(roomspec_robs[-1]))
			for j in range(i+1,nb_rooms): # upper-triangular matrix to avoid repetitions
				comb_stl = '('+roomspec_props[i]+'&'+roomspec_props[j]+')'
				model.set_goal(comb_stl)
				roomcomb_robs.append(model.compute_robustness(trajs_scaled))
				print(f"SAFETY of Room {i} and Room {j} Percentage of positive: ", np.sum((roomcomb_robs[-1] >= 0))/len(roomcomb_robs[-1]))

			
		dataset_dict = {"single_robs": roomspec_robs, "couple_robs": roomcomb_robs, "rob": robs, "x_scaled": states_scaled, "trajs_scaled": trajs_scaled, "x_minmax": (xmin,xmax)}

		filename = f'Datasets/MRH{nb_rooms}_'+datasets[ds]+f'_set_{nb_points[ds]}x{nb_trajs_per_state[ds]}points.pickle'
		with open(filename, 'wb') as handle:
			pickle.dump(dataset_dict, handle)
		handle.close()
		print("Data stored in: ", filename)
		
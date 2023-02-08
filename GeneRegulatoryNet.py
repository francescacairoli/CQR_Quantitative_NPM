import math
import time
import copy
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import gene_regulation_utils as utils
from scipy.integrate import odeint, solve_ivp
from pcheck.series.TimeSeries import TimeSeries
from pcheck.semantics import stlBooleanSemantics, stlRobustSemantics
from scipy.stats import norm, expon, bernoulli, multivariate_normal 



class GeneRegulatoryNet(object):
	'''
	Gene Regulatory Net System (SHS from Wolf V. et al)
	'Stochastic hybrid models of gene regulatory networks'
	h = number of genes
	dt = temporal interval
	N = nb of temporal steps
	'''
	def __init__(self, nb_genes):
		self.model_name = "GRN"
		self.h = nb_genes
		self.state_dim = 2*nb_genes
		self.n_modes = 2**nb_genes
		self.dt = 0.5  
		self.N = 500
		self.nb_of_transitions = 2*nb_genes
		self.var_names = [f'X{k}' for k in range(nb_genes)]

	def get_R_matrix(self, x, q):
		
		R = np.zeros(self.h)
	
		for i in range(self.h):
			if q[i] == 0: #gene deactivate
				R[i] = self.a[i]-self.b[i]*x[i]
			else: #q[i] == 1 gene activated
				R[i] = self.c[i]-self.d[i]*x[i]
		return R

	def get_infinitesimal_generator(self, x, q):
		Q = np.zeros((self.h*2, self.h*2)) # 2 possibles gene states (active, inactive)
		for i in range(self.h):
			Q[2*i,2*i+1] = self.k_unbind[i] #activation event of gene i
			Q[2*i+1,2*i] = self.k_bind[i]*x[i-1]*q[i]#inhibition event of gene i 
			#(if gene i is already inactive this transition cannot take place)
		return Q

	def net_layout(self):

		params = utils.get_parameters(self.h)
		self.a = params['a']#np.ones(self.h)
		self.b = params['b']
		self.c = params['c']
		self.d = params['d']
		self.k_bind = params['binding_rates']
		self.k_unbind = params['unbinding_rates']
		self.ranges = np.array([1,params['x_ub']])*np.ones((self.h,2))
		self.safe_ranges = self.ranges

	def initialize_settings(self, params):
		
		self.final_time = self.N*self.dt
		self.timeline = np.arange(start=0, stop=self.final_time, step=self.dt)
		self.net_layout()
		
	def continuous_dynamics(self, x, q, final_time):
		# deterministic evolution
		n_steps = int(final_time//self.dt)
		x_traj = np.empty((n_steps, self.h))
		curr_time = 0
		for i in range(n_steps):
			R = self.get_R_matrix(x, q)
			x = x+R*self.dt
			x_traj[i] = x
			
		return x_traj


	def discrete_dynamics(self, x, q):
		# stochastic dynamics (jumping chain and exponential times)
		# similar to SSA simulation
		Q = self.get_infinitesimal_generator(x, q)
		rates = np.sum(Q, axis=1)
		total_rate = np.sum(Q)
		trans_index = np.random.choice(self.nb_of_transitions, p=rates / total_rate)
		delta_time = np.random.exponential(1 / (total_rate))
		q[trans_index//2]=1-trans_index%2 
		return q, delta_time

	def dynamics(self, state):
		#state = [x,q]
		#discrete_jump
		next_q, time_delta = self.discrete_dynamics(state[:self.h],state[self.h:])
		
		x_traj = self.continuous_dynamics(state[:self.h],state[self.h:],time_delta)
		q_traj = next_q*np.ones((len(x_traj), self.h))
		traj = np.concatenate((x_traj, q_traj),axis=1)
		return traj, time_delta

	def sample_rnd_states(self, n_samples):

		states_list = []
		for i in range(n_samples):
			#print("Point {}/{}".format(i+1,n_samples))
			x_rnd = self.ranges[:,0]+(self.ranges[:,1]-self.ranges[:,0])*np.random.rand(self.h)
			q_rnd = np.random.choice(2, size=self.h)

			rnd_state = np.hstack((x_rnd,q_rnd))
			states_list.append(rnd_state)

		return np.array(states_list)


	def gen_trajectories(self, states, n_trajs_per_state):

		n_samples = len(states)
		self.n_samples = n_samples
		self.n_trajs_per_state = n_trajs_per_state

		trajs = np.empty((n_samples*n_trajs_per_state, self.N, 2*self.h))
		
		c = 0
		for i in tqdm(range(n_samples)):
			for j in range(n_trajs_per_state):
				#traj_c = np.array([states[i]])
				curr_time = 0
				step = 0
				trajs[c,step] = states[i]
				step+=1
				while step < self.N:
					loc_traj, time_delta = self.dynamics(trajs[c,step-1])
					n_steps = int(time_delta//self.dt)
					if step+n_steps < self.N:
						trajs[c,step:step+n_steps] = loc_traj
					else:
						trajs[c,step:self.N] = loc_traj[:self.N-step]

					step += n_steps
					curr_time += n_steps*self.dt
				c += 1
				
		return trajs


	def set_goal(self, formula):
		self.phi = formula


	def plot_trajectories(self, trajs, ds_name = ""):

		trajs_reshaped = np.reshape(trajs, (self.n_samples, self.n_trajs_per_state, self.N, self.state_dim))
		self.protein_labels = [f'P{i}' for i in range(self.h)]
		self.gene_labels = [f'G{i}' for i in range(self.h)]
		colors = ['r','b','g','y','k','c','m','tab:purple']
		for i in range(self.n_samples):
			fig, axs = plt.subplots(self.h,2, figsize=(12, 9))
			for s in range(self.h):
				for j in range(10):#self.n_trajs_per_state
					times = np.arange(len(trajs_reshaped[i, j, :, s]))
					axs[s][0].plot(self.timeline, trajs_reshaped[i, j, :, s], c=colors[s])
					axs[s][0].set_ylabel(self.protein_labels[s])
					axs[s][0].plot(self.timeline, self.safe_ranges[s,1]*np.ones(len(times)), '--', c=colors[s])
					axs[s][1].step(self.timeline, trajs_reshaped[i, j, :, s+self.h], c=colors[s])
					axs[s][1].set_ylabel(self.gene_labels[s])
				
			axs[-1][0].set_xlabel("time")
			axs[-1][1].set_xlabel("time")
			plt.tight_layout()
			plt.savefig(self.model_name+f"{self.h}/plots/"+ds_name+f"_trajs_{i}.png")
			plt.close()

	def compute_robustness(self, trajs):
		n_states = len(trajs)
		robs = np.empty(n_states)
		
		
		for i in tqdm(range(n_states)):
			time_series_i = TimeSeries(self.var_names, self.timeline, trajs[i,:,:self.h].T)
			robs[i] = stlRobustSemantics(time_series_i, 0, self.phi)

		return robs



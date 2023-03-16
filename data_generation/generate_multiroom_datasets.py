import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pickle
import argparse
from utils import * # import-export methods
from MultiRoomHeating import *

parser = argparse.ArgumentParser()
parser.add_argument("--nb_rooms", default=2, type=int, help="Nb of rooms")
args = parser.parse_args()

datasets = ['train', 'calibration','test']
nb_points = [1000*args.nb_rooms, 500*args.nb_rooms, 100*args.nb_rooms]
nb_trajs_per_state = [50, 50, 500]
nb_points = [1000*args.nb_rooms, 500*args.nb_rooms, 100*args.nb_rooms]
nb_trajs_per_state = [50, 50, 500]

for ds in range(len(datasets)):

	
	model = MultiRoomHeating(args.nb_rooms)
	params = utils.get_parameters(args.nb_rooms)
	model.initialize_settings(params)

	states = model.sample_rnd_states(nb_points[ds])
	start_time = time.time()
	trajs = model.gen_trajectories(states, nb_trajs_per_state[ds])
	end_time = time.time()-start_time
	print(f'Time needed to generate {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')

	if datasets[ds] == 'train':
		xmax = np.max(np.max(trajs, axis = 0), axis = 0)
		xmin = np.min(np.min(trajs, axis = 0), axis = 0)
	else:
		trainset_fn = f'../Datasets/MRH{args.nb_rooms}_'+'train'+f'_set_{nb_points[ds]}x{nb_trajs_per_state[ds]}points.pickle'
		with open(trainset_fn, 'rb') as handle:
			train_data = pickle.load(handle)
		handle.close()
		xmin, xmax = train_data["x_minmax"]


	trajs_scaled = -1+2*(trajs-xmin)/(xmax-xmin)
	states_scaled = -1+2*(states-xmin)/(xmax-xmin)

	scaled_safety_region = -1+2*(model.safe_ranges.T-xmin[:args.nb_rooms])/(xmax[:args.nb_rooms]-xmin[:args.nb_rooms])
	goal_formula_scaled = utils.get_safety_property(args.nb_rooms, model.final_time,scaled_safety_region.T)
	
	model.set_goal(goal_formula_scsafetyaled)
	start_time = time.time()
	robs = model.compute_robustness(trajs_scaled)
	end_time = time.time()-start_time
	print(f'Time need to label (all rooms) {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
		
	print("FULL SAFETY Percentage of positive: ", np.sum((robs >= 0))/len(robs))
	
	roomspec_props = [utils.get_roomspec_safety_property(i, model.final_time,scaled_safety_region.T) for i in range(args.nb_rooms)]
	roomspec_robs = []
	roomcomb_robs = []
	for i in range(args.nb_rooms):
		model.set_goal(roomspec_props[i])
		start_time = time.time()
		roomspec_robs.append(model.compute_robustness(trajs_scaled))
		end_time = time.time()-start_time
		print(f'Time need to label (single room) {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
		print(f"SAFETY of Room {i} Percentage of positive: ", np.sum((roomspec_robs[-1] >= 0))/len(roomspec_robs[-1]))
		for j in range(i+1,args.nb_rooms): # upper-triangular matrix to avoid repetitions
			comb_stl = '('+roomspec_props[i]+'&'+roomspec_props[j]+')'
			model.set_goal(comb_stl)
			roomcomb_robs.append(model.compute_robustness(trajs_scaled))
			print(f"SAFETY of Room {i} and Room {j} Percentage of positive: ", np.sum((roomcomb_robs[-1] >= 0))/len(roomcomb_robs[-1]))

	filename = f'../Datasets/MRH{args.nb_rooms}_'+datasets[ds]+f'_set_{nb_points[ds]}x{nb_trajs_per_state[ds]}points.pickle'
	
	dataset_dict = {"single_robs": roomspec_robs, "couple_robs": roomcomb_robs, "rob": robs, "x_scaled": states_scaled, "trajs_scaled": trajs_scaled, "x_minmax": (xmin,xmax)}

	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)

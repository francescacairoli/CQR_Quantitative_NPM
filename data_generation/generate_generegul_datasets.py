import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from GeneRegulatoryNet import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nb_genes", default=2, type=int, help="Nb of genes")
args = parser.parse_args()

datasets = ['train', 'calibration','test']
nb_points = [1000*args.nb_genes, 500*args.nb_genes, 100*args.nb_genes]
nb_trajs_per_state = [50, 50, 500]
nb_points = [1000*args.nb_genes, 500*args.nb_genes, 100*args.nb_genes]
nb_trajs_per_state = [50, 50, 500]

for ds in range(len(datasets)):

	model = GeneRegulatoryNet(args.nb_genes)
	params = utils.get_parameters(args.nb_genes)
	model.initialize_settings(params)


	states = model.sample_rnd_states(nb_points[ds])
	start_time = time.time()
	trajs = model.gen_trajectories(states, nb_trajs_per_state[ds])
	end_time = time.time()-start_time
	print(f'Time needed to generate {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
	#model.plot_trajectories(trajs, datasets[ds])

	if datasets[ds] == 'train':
		xmax = np.max(np.max(trajs, axis = 0), axis = 0)
		xmin = np.min(np.min(trajs, axis = 0), axis = 0)
	else:
		trainset_fn = f'../Datasets/MRH{args.nb_genes}_'+'train'+f'_set_{nb_points[ds]}x{nb_trajs_per_state[ds]}points.pickle'
		with open(trainset_fn, 'rb') as handle:
			train_data = pickle.load(handle)
		handle.close()
		xmin, xmax = train_data["x_minmax"]
	
	trajs_scaled = -1+2*(trajs-xmin)/(xmax-xmin)
	states_scaled = -1+2*(states-xmin)/(xmax-xmin)

	scaled_safety_region = -1+2*(model.safe_ranges.T-xmin[:args.nb_genes])/(xmax[:args.nb_genes]-xmin[:args.nb_genes])
	print("scaled_safety_region = ", scaled_safety_region)
	goal_formula_scaled = utils.get_property(args.nb_genes, model.final_time,scaled_safety_region.T)
	

	model.set_goal(goal_formula_scaled)
	start_time = time.time()
	robs = model.compute_robustness(trajs_scaled)
	end_time = time.time()-start_time
	print(f'Time need to label (all genes) {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
		
	print("FULL SAFETY Percentage of positive: ", np.sum((robs >= 0))/len(robs))
	
	robs = []	
	genespec_props = [utils.get_genespec_property(i, model.final_time,scaled_safety_region.T) for i in range(args.nb_genes)]
	genespec_robs = []
	genecomb_robs = []
	for i in range(args.nb_genes):
		model.set_goal(genespec_props[i])
		start_time = time.time()
		genespec_robs.append(model.compute_robustness(trajs_scaled))
		end_time = time.time()-start_time
		print(f'Time need to label (single gene) {nb_points[ds]*nb_trajs_per_state[ds]} trajectories = {end_time}')
		print(f"SATISFACTION for Gene {i} Percentage of positive: ", np.sum((genespec_robs[-1] >= 0))/len(genespec_robs[-1]))
		for j in range(i+1,args.nb_genes): # upper-triangular matrix to avoid repetitions
			comb_stl = '('+genespec_props[i]+'&'+genespec_props[j]+')'
			model.set_goal(comb_stl)
			genecomb_robs.append(model.compute_robustness(trajs_scaled))
			print(f"SATISFACTION of Gene {i} and Gene {j} Percentage of positive: ", np.sum((genecomb_robs[-1] >= 0))/len(genecomb_robs[-1]))

		
	dataset_dict = {"single_robs": genespec_robs, "couple_robs": genecomb_robs, "rob": robs, "x_scaled": states_scaled, "x_minmax": (xmin,xmax)}#"trajs_scaled": trajs_scaled,
	
	filename = f'../Datasets/GRN{args.nb_genes}_'+datasets[ds]+f'_set_{nb_points[ds]}x{nb_trajs_per_state[ds]}points.pickle'
	with open(filename, 'wb') as handle:
		pickle.dump(dataset_dict, handle)
	handle.close()
	print("Data stored in: ", filename)

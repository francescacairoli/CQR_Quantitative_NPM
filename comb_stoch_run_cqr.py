import copy
import time
import argparse
from QR import * # NN architecture to learn quantiles
from CQR import *
from utils import * # import-export methods
from Dataset import *
from TrainQR_multiquantile import *


# Reminder: we save results in the folder of the first variable in the pair

# for the sake of reproducibility we fix the seeds
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dim", default=2, type=int, help="Dimension of the model")
parser.add_argument("--model_prefix", default="MRH", type=str, help="Prefix of the model name")
parser.add_argument("--n_epochs", default=500, type=int, help="Nb of training epochs for QR")
parser.add_argument("--n_hidden", default=20, type=int, help="Nb of hidden nodes per layer")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--qr_training_flag", default=True, type=eval, help="training flag")
parser.add_argument("--comb_calibr_flag", default=True, type=eval, help="do combined calibration")
parser.add_argument("--xavier_flag", default=False, type=eval, help="Xavier random weights initialization")
parser.add_argument("--scheduler_flag", default=False, type=eval, help="scheduler flag")
parser.add_argument("--opt", default="Adam", type=str, help="Optimizer")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Drop-out rate")
parser.add_argument("--alpha", default=0.1, type=float, help="quantiles significance level")
parser.add_argument("--comb_idx", default=0, type=int, help="Identifier of the combination of properteies to monitor")
args = parser.parse_args()
print(args)

#print(torch.cuda.device_count())

model_name = args.model_prefix+str(args.model_dim)

comb_pairs = []
for i in range(args.model_dim):
	for j in range(i+1, args.model_dim):
		comb_pairs.append((i,j))
print('Combination pairs = ', comb_pairs)

prop_idxs = comb_pairs[args.comb_idx]


print("MODEL = ", model_name, "Dim = ", args.model_dim)

trainset_fn, calibrset_fn, testset_fn, ds_details = import_filenames_w_dim(model_name, args.model_dim, extra='K')
n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size = ds_details

print("qr_training_flag = ", args.qr_training_flag)

#trainset_fnH, calibrset_fnH, testset_fnH, ds_detailsH = import_filenames_w_dim(model_name, args.model_dim, extra='H')
#n_train_statesH, n_cal_statesH, n_test_statesH, cal_hist_sizeH, test_hist_sizeH = ds_detailsH

quantiles = np.array([args.alpha/2, 0.5,  1-args.alpha/2]) # LB, MEDIAN, UB
nb_quantiles = len(quantiles)


print(f"Property idxs = {prop_idxs}")

idx_str1 = f'CQR_#{prop_idxs[0]}_Dropout{args.dropout_rate}_multiout_opt=_{args.n_hidden}hidden_{args.n_epochs}epochs_{nb_quantiles}quantiles_3layers_alpha{args.alpha}_lr{args.lr}'
idx_str2 = f'CQR_#{prop_idxs[1]}_Dropout{args.dropout_rate}_multiout_opt=_{args.n_hidden}hidden_{args.n_epochs}epochs_{nb_quantiles}quantiles_3layers_alpha{args.alpha}_lr{args.lr}'


dataset = Dataset(property_idx=args.comb_idx, comb_flag=True, trainset_fn=trainset_fn, testset_fn=testset_fn, 
			calibrset_fn=calibrset_fn, alpha=args.alpha, n_train_states=n_train_states, n_cal_states=n_cal_states, 
			n_test_states=n_test_states, hist_size=cal_hist_size, test_hist_size=test_hist_size)
dataset.load_data()

if args.comb_calibr_flag:

	# Train the QR
	qr1 = TrainQR(model_name, dataset, idx = idx_str1, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = args.opt, n_hidden = args.n_hidden, xavier_flag = args.xavier_flag, scheduler_flag = args.scheduler_flag, drop_out_rate = args.dropout_rate)
	qr1.load_model(args.n_epochs)
	qr2 = TrainQR(model_name, dataset, idx = idx_str2, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = args.opt, n_hidden = args.n_hidden, xavier_flag = args.xavier_flag, scheduler_flag = args.scheduler_flag, drop_out_rate = args.dropout_rate)
	qr2.load_model(args.n_epochs)

	print(f"--------Property idxs = {prop_idxs}")

	# Obtain CQR intervals given the trained QR
	cqr = CQR(dataset.X_cal, dataset.R_cal, (qr1.qr_model,qr2.qr_model), test_hist_size = test_hist_size, cal_hist_size = cal_hist_size, comb_flag= True)
	cpi_test = cqr.get_cpi(dataset.X_test, pi_flag = False)


	cqr.plot_results(dataset.R_test, cpi_test, "CQR_interval", qr1.results_path, extra_info=f'_comb{args.comb_idx}_pair={prop_idxs}')
	cpi_coverage, cpi_efficiency = cqr.get_coverage_efficiency(dataset.R_test, cpi_test)
	print("cpi_coverage = ", cpi_coverage, "cpi_efficiency = ", cpi_efficiency)
	cpi_correct, cpi_uncertain, cpi_wrong, cpi_fp = cqr.compute_accuracy_and_uncertainty(cpi_test, dataset.L_test)
	print("cpi_correct = ", cpi_correct, "cpi_uncertain = ", cpi_uncertain, "cpi_wrong = ", cpi_wrong, "cpi_fp = ", cpi_fp)

	#dataset1 = Dataset(property_idx=prop_idxs[0], comb_flag=False, trainset_fn=trainset_fnH, testset_fn=testset_fnH, 
	#			calibrset_fn=calibrset_fnH, alpha=args.alpha, n_train_states=n_train_statesH, n_cal_states=n_cal_statesH, 
	#			n_test_states=n_test_statesH, hist_size=cal_hist_sizeH, test_hist_size=test_hist_sizeH)
	#dataset1.load_data()
	dataset1 = Dataset(property_idx=prop_idxs[0], comb_flag=False, trainset_fn=trainset_fn, testset_fn=testset_fn, 
				calibrset_fn=calibrset_fn, alpha=args.alpha, n_train_states=n_train_states, n_cal_states=n_cal_states, 
				n_test_states=n_test_states, hist_size=cal_hist_size, test_hist_size=test_hist_size)
	dataset1.load_data()
	#dataset2 = Dataset(property_idx=prop_idxs[1], comb_flag=False, trainset_fn=trainset_fnH, testset_fn=testset_fnH, 
	#			calibrset_fn=calibrset_fnH, alpha=args.alpha, n_train_states=n_train_statesH, n_cal_states=n_cal_statesH, 
	#			n_test_states=n_test_statesH, hist_size=cal_hist_sizeH, test_hist_size=test_hist_sizeH)
	#dataset2.load_data()
	dataset2 = Dataset(property_idx=prop_idxs[1], comb_flag=False, trainset_fn=trainset_fn, testset_fn=testset_fn, 
				calibrset_fn=calibrset_fn, alpha=args.alpha, n_train_states=n_train_states, n_cal_states=n_cal_states, 
				n_test_states=n_test_states, hist_size=cal_hist_size, test_hist_size=test_hist_size)
	dataset2.load_data()
	print("-------------Statical guarantees of the union")
	cqr1 = CQR(dataset1.X_cal, dataset1.R_cal, qr1.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size, comb_flag= False)
	cpi1 = cqr1.get_cpi(dataset1.X_test)
	cqr2 = CQR(dataset2.X_cal, dataset2.R_cal, qr2.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size, comb_flag= False)
	cpi2 = cqr2.get_cpi(dataset2.X_test)

	union_coverage, union_efficiency = cqr.get_coverage_efficiency_coupled(dataset.R_test, cpi1, cpi2)
	print("union_cpi_coverage = ", union_coverage, "union_cpi_efficiency = ", union_efficiency)


	results_list = ["\n Quantiles = ", str(quantiles), "\n Id1 = ", idx_str1,"\n Id2 = ", idx_str2, "\n tau = ", str(cqr.tau),
	"\n cpi_coverage = ", str(cpi_coverage), "\n cpi_efficiency = ", str(cpi_efficiency),
	"\n cpi_correct = ", str(cpi_correct), "\n cpi_uncertain = ", str(cpi_uncertain), "\n cpi_wrong = ", str(cpi_wrong), "\n cpi_fp = ", str(cpi_fp),
	"\n union_cpi_coverage = ", str(union_coverage), "\n union_cpi_efficiency = ", str(union_efficiency)]

	save_results_to_file(results_list, qr1.results_path, extra_info=f'_comb{args.comb_idx}_pair={prop_idxs}')
	print(qr1.results_path)

else: # train the CQR over the combined property


	idx_str12 = f'CQR_#{prop_idxs[0]}{prop_idxs[1]}_Dropout{args.dropout_rate}_multiout_opt=_{args.n_hidden}hidden_{args.n_epochs}epochs_{nb_quantiles}quantiles_3layers_alpha{args.alpha}_lr{args.lr}'

	qr12 = TrainQR(model_name, dataset, idx = idx_str12, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = args.opt, n_hidden = args.n_hidden, xavier_flag = args.xavier_flag, scheduler_flag = args.scheduler_flag, drop_out_rate = args.dropout_rate)
	qr12.initialize()

	if args.qr_training_flag:
		start_time = time.time()
		qr12.train(args.n_epochs, args.batch_size, args.lr)
		end_time = time.time()-start_time
		qr12.save_model()
		print(f'Training time for {model_name}-#{prop_idxs} with {args.n_epochs} epochs = {end_time}')
	else:
		qr12.load_model(args.n_epochs)


	# Obtain CQR intervals given the trained QR
	cqr12 = CQR(dataset.X_cal, dataset.R_cal, qr12.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
	cpi_test, pi_test = cqr12.get_cpi(dataset.X_test, pi_flag = True)

	print("shape: ", cpi_test.shape, pi_test.shape)

	cqr12.plot_results(dataset.R_test, pi_test, "QR_interval", qr12.results_path)
	pi_coverage, pi_efficiency = cqr12.get_coverage_efficiency(dataset.R_test, pi_test)
	print("pi_coverage = ", pi_coverage, "pi_efficiency = ", pi_efficiency)
	pi_correct, pi_uncertain, pi_wrong, pi_fp = cqr12.compute_accuracy_and_uncertainty(pi_test, dataset.L_test)
	print("pi_correct = ", pi_correct, "pi_uncertain = ", pi_uncertain, "pi_wrong = ", pi_wrong, "pi_fp = ", pi_fp)

	cqr12.plot_results(dataset.R_test, cpi_test, "CQR_interval", qr12.results_path)
	cpi_coverage, cpi_efficiency = cqr12.get_coverage_efficiency(dataset.R_test, cpi_test)
	print("cpi_coverage = ", cpi_coverage, "cpi_efficiency = ", cpi_efficiency)
	cpi_correct, cpi_uncertain, cpi_wrong, cpi_fp = cqr12.compute_accuracy_and_uncertainty(cpi_test, dataset.L_test)
	print("cpi_correct = ", cpi_correct, "cpi_uncertain = ", cpi_uncertain, "cpi_wrong = ", cpi_wrong, "cpi_fp = ", cpi_fp)


	results_list = ["\n Quantiles = ", str(quantiles), "\n Id = ", idx_str12, "\n tau = ", str(cqr12.tau),
	"\n pi_coverage = ", str(pi_coverage), "\n pi_efficiency = ", str(pi_efficiency),
	"\n pi_correct = ", str(pi_correct), "\n pi_uncertain = ", str(pi_uncertain), "\n pi_wrong = ", str(pi_wrong),"\n pi_fp = ", str(pi_fp),
	"\n cpi_coverage = ", str(cpi_coverage), "\n cpi_efficiency = ", str(cpi_efficiency),
	"\n cpi_correct = ", str(cpi_correct), "\n cpi_uncertain = ", str(cpi_uncertain), "\n cpi_wrong = ", str(cpi_wrong),"\n cpi_fp = ", str(cpi_fp)]

	save_results_to_file(results_list, qr12.results_path)
	print(qr12.results_path)



import copy
import time
import argparse
from QR import * # NN architecture to learn quantiles
from CQR import *
from utils import * # import-export methods
from Dataset import *
from TrainQR_multiquantile import *

# for the sake of reproducibility we fix the seeds
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dim", default=2, type=int, help="Dimension of the model")
parser.add_argument("--model_prefix", default="MRH", type=str, help="Prefix of the model name")
parser.add_argument("--n_epochs", default=200, type=int, help="Nb of training epochs for QR")
parser.add_argument("--n_hidden", default=20, type=int, help="Nb of hidden nodes per layer")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--qr_training_flag", default=True, type=eval, help="training flag")
parser.add_argument("--xavier_flag", default=False, type=eval, help="Xavier random weights initialization")
parser.add_argument("--scheduler_flag", default=False, type=eval, help="scheduler flag")
parser.add_argument("--opt", default="Adam", type=str, help="Optimizer")
parser.add_argument("--dropout_rate", default=0.2, type=float, help="Drop-out rate")
parser.add_argument("--alpha", default=0.1, type=float, help="quantiles significance level")
parser.add_argument("--property_idx", default=0, type=int, help="Identifier of the property to monitor (-1 denotes that the property is wrt all variables)")
args = parser.parse_args()
print(args)

print(torch.cuda.device_count())

model_name = args.model_prefix+str(args.model_dim)

print("MODEL = ", model_name, "Dim = ", args.model_dim)

trainset_fn, calibrset_fn, testset_fn, ds_details = import_filenames_w_dim(model_name, args.model_dim)
n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size = ds_details


quantiles = np.array([args.alpha/2, 0.5,  1-args.alpha/2]) # LB, MEDIAN, UB
nb_quantiles = len(quantiles)

print(f"Quantiles = {quantiles}")
print(f"Property idx = {args.property_idx}")

print(f"n_epochs = {args.n_epochs}, lr = {args.lr}, batch_size = {args.batch_size}")

idx_str = f'CQR_#{args.property_idx}_Dropout{args.dropout_rate}_multiout_opt=_{args.n_hidden}hidden_{args.n_epochs}epochs_{nb_quantiles}quantiles_3layers_alpha{args.alpha}_lr{args.lr}'

print(f"Results folder = {idx_str}")

dataset = Dataset(property_idx=args.property_idx, comb_flag=False, trainset_fn=trainset_fn, testset_fn=testset_fn, 
			calibrset_fn=calibrset_fn, alpha=args.alpha, n_train_states=n_train_states, n_cal_states=n_cal_states, 
			n_test_states=n_test_states, hist_size=cal_hist_size, test_hist_size=test_hist_size)
dataset.load_data()

# Train the QR
qr = TrainQR(model_name, dataset, idx = idx_str, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = args.opt, n_hidden = args.n_hidden, xavier_flag = args.xavier_flag, scheduler_flag = args.scheduler_flag, drop_out_rate = args.dropout_rate)
qr.initialize()

if args.qr_training_flag:
	start_time = time.time()
	qr.train(args.n_epochs, args.batch_size, args.lr)
	end_time = time.time()-start_time
	qr.save_model()
	print(f'Training time for {model_name}-#{args.property_idx} with {args.n_epochs} epochs = {end_time}')
else:
	qr.load_model(args.n_epochs)


# Obtain CQR intervals given the trained QR
cqr = CQR(dataset.X_cal, dataset.R_cal, qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
cpi_test, pi_test = cqr.get_cpi(dataset.X_test, pi_flag = True)

print("shape: ", cpi_test.shape, pi_test.shape)

cqr.plot_results(dataset.R_test, pi_test, "QR_interval", qr.results_path)
pi_coverage, pi_efficiency = cqr.get_coverage_efficiency(dataset.R_test, pi_test)
print("pi_coverage = ", pi_coverage, "pi_efficiency = ", pi_efficiency)
pi_correct, pi_uncertain, pi_wrong = cqr.compute_accuracy_and_uncertainty(pi_test, dataset.L_test)
print("pi_correct = ", pi_correct, "pi_uncertain = ", pi_uncertain, "pi_wrong = ", pi_wrong)

cqr.plot_results(dataset.R_test, cpi_test, "CQR_interval", qr.results_path)
cpi_coverage, cpi_efficiency = cqr.get_coverage_efficiency(dataset.R_test, cpi_test)
print("cpi_coverage = ", cpi_coverage, "cpi_efficiency = ", cpi_efficiency)
cpi_correct, cpi_uncertain, cpi_wrong = cqr.compute_accuracy_and_uncertainty(cpi_test, dataset.L_test)
print("cpi_correct = ", cpi_correct, "cpi_uncertain = ", cpi_uncertain, "cpi_wrong = ", cpi_wrong)


results_list = ["\n Quantiles = ", str(quantiles), "\n Id = ", idx_str, "\n tau = ", str(cqr.tau),
"\n pi_coverage = ", str(pi_coverage), "\n pi_efficiency = ", str(pi_efficiency),
"\n pi_correct = ", str(pi_correct), "\n pi_uncertain = ", str(pi_uncertain), "\n pi_wrong = ", str(pi_wrong),
"\n cpi_coverage = ", str(cpi_coverage), "\n cpi_efficiency = ", str(cpi_efficiency),
"\n cpi_correct = ", str(cpi_correct), "\n cpi_uncertain = ", str(cpi_uncertain), "\n cpi_wrong = ", str(cpi_wrong)]

save_results_to_file(results_list, qr.results_path)
print(qr.results_path)
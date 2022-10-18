import copy
from QR import * # NN architecture to learn quantiles
from CQR import *
from utils import * # import-export methods
from Dataset import *
from TrainQR_multiquantile import *

# for the sake of reproducibility we fix the seeds
torch.manual_seed(0)
np.random.seed(0)

print(torch.cuda.device_count())

model_dim = 4
model_name = f"MRH{model_dim}"


print("MODEL = ", model_name, "Dim = ", model_dim)

trainset_fn, calibrset_fn, testset_fn, ds_details = import_filenames_w_dim(model_name, model_dim)
n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size = ds_details

n_epochs = 200

batch_size = 512
lr = 0.0005

qr_training_flag = True

xavier_flag = False
n_hidden = 20
drop_out_rate = 0.2
opt = "Adam"
scheduler_flag = False

print("qr_training_flag = ", qr_training_flag)
alpha = 0.1 # significance level for QR
quantiles = np.array([alpha/2, 0.5,  1-alpha/2]) # LB, MEDIAN, UB
nb_quantiles = len(quantiles)


property_idx = 3 # -1 denotes that the property is wrt all rooms

print(f"Quantiles = {quantiles}")
print(f"Property idx = {property_idx}")

print(f"n_epochs = {n_epochs}, lr = {lr}, batch_size = {batch_size}")

idx_str = f'CQR_#{property_idx}_Dropout{drop_out_rate}_multiout_opt=_{n_hidden}hidden_{n_epochs}epochs_{nb_quantiles}quantiles_3layers_alpha{alpha}_lr{lr}'

print(f"Results folder = {idx_str}")

dataset = Dataset(property_idx=property_idx, comb_flag=False, trainset_fn=trainset_fn, testset_fn=testset_fn, 
			calibrset_fn=calibrset_fn, alpha=alpha, n_train_states=n_train_states, n_cal_states=n_cal_states, 
			n_test_states=n_test_states, cal_hist_size=cal_hist_size, test_hist_size=test_hist_size)
dataset.load_data()

# Train the QR
qr = TrainQR(model_name, dataset, idx = idx_str, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = opt, n_hidden = n_hidden, xavier_flag = xavier_flag, scheduler_flag = scheduler_flag, drop_out_rate = drop_out_rate)
qr.initialize()

if qr_training_flag:
	qr.train(n_epochs, batch_size, lr)
	qr.save_model()
else:
	qr.load_model(n_epochs)


# Obtain CQR intervals given the trained QR
cqr = CQR(dataset.X_cal, dataset.R_cal, qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
cpi_test, pi_test = cqr.get_cpi(dataset.X_test, pi_flag = True)

print("shape: ", cpi_test.shape, pi_test.shape)

cqr.plot_results(dataset.R_test, pi_test, "QR_interval", qr.results_path, )
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
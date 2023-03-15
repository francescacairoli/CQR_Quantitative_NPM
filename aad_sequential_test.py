import sys
sys.path.append(".")
import pickle
import argparse
from utils import * # import-export methods
from data_generation.AutomAnaesthesiaDelivery import *

from QR import * # NN architecture to learn quantiles
from CQR import *
from Dataset import *
from TrainQR_multiquantile import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dim", default=2, type=int, help="Dimension of the model")
parser.add_argument("--model_prefix", default="AAD", type=str, help="Prefix of the model name")
parser.add_argument("--n_epochs", default=500, type=int, help="Nb of training epochs for QR")
parser.add_argument("--n_hidden", default=20, type=int, help="Nb of hidden nodes per layer")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--qr_training_flag", default=True, type=eval, help="training flag")
parser.add_argument("--xavier_flag", default=False, type=eval, help="Xavier random weights initialization")
parser.add_argument("--scheduler_flag", default=False, type=eval, help="scheduler flag")
parser.add_argument("--opt", default="Adam", type=str, help="Optimizer")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Drop-out rate")
parser.add_argument("--alpha", default=0.1, type=float, help="quantiles significance level")
parser.add_argument("--property_idx", default=-1, type=int, help="Identifier of the property to monitor (-1 denotes that the property is wrt all variables)")
parser.add_argument("--prop_str", default='G', type=str, help='G for globally property, F for eventually property')
parser.add_argument("--seed", default=0, type=int, help='set random seed')
args = parser.parse_args()

# for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)


nb_trajs_per_state = 500
n_steps = 20


if args.prop_str == 'F':
	args.model_prefix = args.model_prefix+args.prop_str
model_name = args.model_prefix+str(args.model_dim)

trainset_fn, calibrset_fn, testset_fn, ds_details = import_filenames_w_dim(model_name, args.model_dim)
n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size = ds_details

quantiles = np.array([args.alpha/2, 0.5,  1-args.alpha/2]) # LB, MEDIAN, UB
nb_quantiles = len(quantiles)

idx_str = f'CQR_#{args.property_idx}_Dropout{args.dropout_rate}_multiout_opt=_{args.n_hidden}hidden_{args.n_epochs}epochs_{nb_quantiles}quantiles_3layers_alpha{args.alpha}_lr{args.lr}'

print(f"Models folder = Models/{model_name}/ID_{idx_str}")
print(f"Results folder = Results/{model_name}/ID_{idx_str}")

# import data
dataset = Dataset(property_idx=args.property_idx, comb_flag=False, trainset_fn=trainset_fn, testset_fn=testset_fn, 
			calibrset_fn=calibrset_fn, alpha=args.alpha, n_train_states=n_train_states, n_cal_states=n_cal_states, 
			n_test_states=n_test_states, hist_size=cal_hist_size, test_hist_size=test_hist_size)
_ = dataset.load_data()

file = open(trainset_fn, 'rb')
data = pickle.load(file)
file.close()
xmin, xmax = data["x_minmax"]

model = AutomatedAnaesthesiaDelivery()
model.initialize_settings()

if args.prop_str == 'G':
	scaled_safety_region = -1+2*(model.safe_ranges.T-xmin)/(xmax-xmin)
	goal_formula_scaled = '( ( ( G_[0,{}]  ( (X1 <= {}) & (X1 >= {}) ) ) & ( G_[0,{}]  ( (X2 <= {}) & (X2 >= {}) ) ) ) & ( G_[0,{}]  ( (X3 <= {}) & (X3 >= {}) ) )     ) '.format(model.time_horizon, scaled_safety_region[1,0], scaled_safety_region[0,0], model.time_horizon, scaled_safety_region[1,1], scaled_safety_region[0,1], model.time_horizon, scaled_safety_region[1,2], scaled_safety_region[0,2])
else: #F
	goal_formula_scaled = '(F_[0,{}]  (X2 < X3) )'.format(model.time_horizon)

model.set_goal(goal_formula_scaled)

qr = TrainQR(model_name, dataset, idx = idx_str, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = args.opt, n_hidden = args.n_hidden, xavier_flag = args.xavier_flag, scheduler_flag = args.scheduler_flag, drop_out_rate = args.dropout_rate)
qr.initialize()
qr.load_model(args.n_epochs)
cqr = CQR(dataset.X_cal, dataset.R_cal, qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)


# randomly sample an initial state
state = model.sample_rnd_states(1) #sample an initial state

# unrull a single evolution from this initial state
running_traj = model.gen_trajectories(state, 1)

running_traj_scaled = -1+2*(running_traj-xmin)/(xmax-xmin)
running_rob = model.compute_robustness(running_traj_scaled)

list_robs = np.empty((n_steps,nb_trajs_per_state))
list_cpi = np.empty((n_steps,nb_quantiles))
list_pi = np.empty((n_steps,nb_quantiles))
yq = []
yq_out = []
xtime_rep = []
xline_rep_out = []
for t in range(n_steps):
	# for each state in the trajectory compute the empirical distribution of robustness values
	trajs = model.gen_trajectories([running_traj[0][t]], nb_trajs_per_state)

	trajs_scaled = -1+2*(trajs-xmin)/(xmax-xmin)
	state_scaled = -1+2*([running_traj[0][t]]-xmin)/(xmax-xmin)


	robs = model.compute_robustness(trajs_scaled)
	lower = np.quantile(robs, args.alpha/2)
	upper = np.quantile(robs, 1-args.alpha/2)
	for j in range(len(robs)):
		if robs[j] <= upper and robs[j] >= lower:
			yq.append(robs[j])
			xtime_rep.append(t)		
		else:
			yq_out.append(robs[j])
			xline_rep_out.append(t)					

	list_robs[t] = robs

	# apply CQR to each state in the trajectory
	cpi_test, pi_test = cqr.get_cpi(state_scaled, pi_flag = True)
	list_cpi[t] = cpi_test
	list_pi[t] = pi_test

# analyze the sequential performances
cov, eff = cqr.get_coverage_efficiency(list_robs.flatten(), list_pi)
print('pi sequential coverage = ',cov)
print('pi sequential efficiency = ',eff)

cov, eff = cqr.get_coverage_efficiency(list_robs.flatten(), list_cpi)
print('cpi sequential coverage = ',cov)
print('cpi sequential efficiency = ',eff)

xtime = np.arange(n_steps)

y_med = list_cpi[:,1]
dminus = y_med-list_cpi[:,0]
dplus = list_cpi[:,-1]-y_med

y_med_pi = list_pi[:,1]
dminus_pi = y_med_pi-list_pi[:,0]
dplus_pi = list_pi[:,-1]-y_med_pi

fig = plt.figure(figsize=(20,4))
plt.scatter(xline_rep_out, yq_out, c='peachpuff', s=6, alpha = 0.25)
plt.scatter(xtime_rep, yq,c='orange', s=8, alpha = 0.25, label='seq-test')
plt.errorbar(x=xtime+0.2, y=y_med_pi, yerr=[dminus_pi,dplus_pi], color = 'c',fmt='o',  capsize = 4, label='QR')
plt.errorbar(x=xtime+0.4, y=y_med, yerr=[dminus,dplus], color = 'darkviolet',fmt='o',  capsize = 4, label='CQR')
plt.plot(xtime[:n_steps], np.zeros(n_steps), '-.',c='k')
plt.title('sequential evaluation')
plt.xlabel('time')
plt.ylabel('robustness')
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig(qr.results_path+"/sequential_evaluation_"+args.prop_str+f"_seed={args.seed}.png")
plt.close()

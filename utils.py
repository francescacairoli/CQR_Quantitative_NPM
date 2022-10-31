def import_filenames_w_dim(model_name, dim):

	n_train_states = dim*1000
	n_cal_states = dim*500
	n_test_states = dim*100
	cal_hist_size = 50
	test_hist_size = 500

	trainset_fn = "Datasets/"+model_name+"_train_set_{}x{}points.pickle".format(n_train_states, cal_hist_size)
	testset_fn = "Datasets/"+model_name+"_test_set_{}x{}points.pickle".format(n_test_states, test_hist_size)
	calibrset_fn = "Datasets/"+model_name+"_calibration_set_{}x{}points.pickle".format(n_cal_states, cal_hist_size)

	return trainset_fn, calibrset_fn, testset_fn, (n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size)


def save_results_to_file(results_list, filepath, extra_info= ""):

	f = open(filepath+"/results"+extra_info+".txt", "w")
	for i in range(len(results_list)):
		f.write(results_list[i])
	f.close()
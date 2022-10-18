import numpy as np

def get_parameters(nb_rooms):

	
	#A, B, C, thresholds, steepnesses, ranges, safe_ranges = params
	if nb_rooms == 2:

		A = np.array([[0,0.0625], [0.0625, 0]])
		B = np.array([0.0375,0.025])
		C = np.array([0.65,0.6])
		thresholds = [19.5,21.25]
		steepnesses = [10,10] # 1 = flat, 10 = gradual, 100 = steep
		ranges = np.array([[17,22],[16,23]])
		#safe_ranges = np.array([[thresholds[0]-1, thresholds[0]+1], [thresholds[1]-1, thresholds[1]+1]])
		safe_ranges = ranges

	elif nb_rooms == 4:
		a = 0.0625
		A = np.array([[0,a,a,0],[a,0,0,a],[a,0,0,a],[0,a,a,0]])
		B = np.array([0.0375,0.025,0.0375,0.025])
		C = np.array([0.65,0.6,0.65,0.6])
		thresholds = [19.5,21.25, 20.5,22.25]
		steepnesses = [10,10,10,10] # 1 = flat, 10 = gradual, 100 = steep
		ranges = np.array([[17,22],[16,23],[16,23],[20,25]])
		#safe_ranges = np.array([[thresholds[i]-1, thresholds[i]+1] for i in range(nb_rooms)])
		safe_ranges = ranges
	else:
		A, B, C, thresholds, steepnesses, ranges, safe_ranges = 0,0,0,0,0,0,0


	params = {"A": A, "B": B, "C": C, "thresholds": thresholds, "steepnesses": steepnesses, "ranges": ranges, "safe_ranges": safe_ranges}

	return params

def get_roomspec_safety_property(room_idx, T, safe):

	prop = f'( G_[0,{T}]  ( (X{room_idx} <= {safe[room_idx,1]}) & (X{room_idx} >= {safe[room_idx,0]}) ) )'

	return prop

def get_safety_property(nb_rooms, T, safe):

	parts = []
	for i in range(nb_rooms):

		parts.append(f'( G_[0,{T}]  ( (X{i} <= {safe[i,1]}) & (X{i} >= {safe[i,0]}) ) )')

	if nb_rooms == 2:
		safety_property = '(' + parts[0]+ ' & ' + parts[1] + ')' 
	elif nb_rooms == 4:
		safety_property = '( (' + parts[0]+ ' & ' + parts[1] + ') & (' + parts[2]+ ' & ' + parts[3] + ') )'
	else:
		safety_property = ''

	return safety_property
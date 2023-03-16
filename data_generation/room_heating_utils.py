import numpy as np

def get_parameters(nb_rooms):

	a = 0.0625
	b1, b2 = 0.0375,0.025
	c1, c2 = 0.65,0.6
	steep = 10
	#A, B, C, thresholds, steepnesses, ranges, safe_ranges = params
	if nb_rooms == 2:

		A = np.array([[0,a], [a, 0]])
		B = np.array([b1, b2])
		C = np.array([c1, c2])
		thresholds = [19.5,21.25]
		steepnesses = [steep,steep] # 1 = flat, 10 = gradual, 100 = steep
		ranges = np.array([[17,22],[16,23]])
		#safe_ranges = np.array([[thresholds[0]-1, thresholds[0]+1], [thresholds[1]-1, thresholds[1]+1]])
		safe_ranges = ranges

	elif nb_rooms == 4:
		
		A = np.array([[0,a,a,0],[a,0,0,a],[a,0,0,a],[0,a,a,0]])
		B = np.array([b1, b2, b1, b2])
		C = np.array([c1, c2, c1, c2])
		thresholds = [19.5,21.25, 20.5,22.25]
		steepnesses = [steep,steep,steep,steep] # 1 = flat, 10 = gradual, 100 = steep
		ranges = np.array([[17,22],[16,23],[16,23],[20,25]])
		#safe_ranges = np.array([[thresholds[i]-1, thresholds[i]+1] for i in range(nb_rooms)])
		safe_ranges = ranges
	
	elif nb_rooms == 8:
		A = np.zeros((nb_rooms,nb_rooms))
		A[0,[1,2,4]] = a
		A[1,[0,3,6]] = a
		A[2,[0,3,5]] = a
		A[3,[1,2,7]] = a
		A[4,[0,5,6]] = a
		A[5,[2,4,7]] = a
		A[6,[1,4,7]] = a
		A[7,[3,5,6]] = a
		B = np.array([b1, b2, b1, b2, b1, b2, b1, b2])
		C = np.array([c1, c2, c1, c2, c1, c2, c1, c2])
		steepnesses = steep*np.ones(nb_rooms)
		thresholds = [19.5,21.25, 20.5,22.25, 22.25, 20.5,21,25,19.5]
		ranges = np.array([[17,22],[16,23],[16,23],[20,25],[20,25],[16,23],[16,23],[17,22]])
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
	elif nb_rooms == 8:
		safety_property = '( ( (' + parts[0]+ ' & ' + parts[1] + ') & (' + parts[2]+ ' & ' + parts[3] + ') ) & ( (' + parts[4]+ ' & ' + parts[5] + ') & (' + parts[6]+ ' & ' + parts[7] + ') ) )'
	else:
		safety_property = ''

	return safety_property
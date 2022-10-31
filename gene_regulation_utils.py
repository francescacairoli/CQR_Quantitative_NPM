import numpy as np

def get_parameters(nb_genes):
	x_ub = 300
	if nb_genes == 2:
		# production rates
		a = [0.0,0.0] # inactive
		c = 600*[1,0.4] # active
		# degradation rates
		b = [0.001,0.0004] # inactive
		d = [0.001,0.0004] # active
		binding_rates = [0.001,0.001]#[0.008,0.008]
		unbinding_rates = [0.01,0.01]#[0.02,0.01]
	elif nb_genes == 4:
		# production rates
		a = [0.0,0.0,0.0,0.0] # inactive
		c = 600*[1,0.8,0.6,0.4] # active
		# degradation rates
		b = [0.001,0.0008,0.006,0.0004] # inactive
		d = [0.001,0.0008,0.006,0.0004] # active
		binding_rates = [0.001,0.001,0.001,0.001]
		unbinding_rates = [0.01,0.01,0.01,0.01]
	elif nb_genes ==6:
		# production rates
		a = [0.0,0.0,0.0,0.0,0.0,0.0] # inactive
		c = 600*[1,0.9,0.8,0.7,0.6,0.5] # active
		# degradation rates
		b = [0.001,0.0009,0.0008,0.0007,0.006,0.0005] # inactive
		d = [0.001,0.0009,0.0008,0.0007,0.006,0.0005] # active
		binding_rates = [0.001,0.001,0.001,0.001,0.001,0.001]
		unbinding_rates = [0.01,0.01,0.01,0.01,0.01,0.01]
	else:
		a,b,c,d,binding_rates,unbinding_rates,x_ub = 0,0,0,0,0,0,0

	params = {'a': a,'b': b,'c': c,'d': d,
			'binding_rates': binding_rates,'unbinding_rates': unbinding_rates,'x_ub': x_ub}
	return params

def get_genespec_property(gene_idx, T, bound):
	prop = f'(G_[{T/2},{T}](X{gene_idx}<={bound[gene_idx][1]/2}))'
	return prop

def get_property(nb_genes, T, bound):

	parts = []
	for i in range(nb_genes):
		#parts.append(f'( F_[0,{T}](G_[0,{T}](X{i}<={bound[i][1]})) )')
		parts.append(get_genespec_property(i, T, bound))
		
	if nb_genes == 2:
		bound_property = '(' + parts[0]+ ' & ' + parts[1] + ')' 
	elif nb_genes == 4:
		bound_property = '( (' + parts[0]+ ' & ' + parts[1] + ') & (' + parts[2]+ ' & ' + parts[3] + ') )'
	elif nb_genes == 8:
		bound_property = '( ( (' + parts[0]+ ' & ' + parts[1] + ') & (' + parts[2]+ ' & ' + parts[3] + ') ) & ( (' + parts[4]+ ' & ' + parts[5] + ') & (' + parts[6]+ ' & ' + parts[7] + ') ) )'
	else:
		bound_property = ''

	return bound_property
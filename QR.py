import torch
import torch.nn as nn
import torch.nn.functional as F


class QR(nn.Module):

	def __init__(self, input_size = 2, hidden_size = 50, output_size = 2, xavier_flag = False, drop_out_rate = 0.1):
		super(QR, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.drop_out_rate = drop_out_rate

		self.fc_in = nn.Linear(input_size, hidden_size)
		self.fc_1 = nn.Linear(hidden_size, hidden_size)		
		self.fc_2 = nn.Linear(hidden_size, hidden_size)
		self.fc_3 = nn.Linear(hidden_size, hidden_size)
		self.fc_out = nn.Linear(hidden_size, output_size)
		
		if xavier_flag:
			nn.init.xavier_normal_(self.fc_in.weight)
			nn.init.xavier_normal_(self.fc_1.weight)
			nn.init.xavier_normal_(self.fc_2.weight)
			nn.init.xavier_normal_(self.fc_3.weight)
			nn.init.xavier_normal_(self.fc_out.weight)


		
	def forward(self, x):
		
		x = self.fc_in(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=self.drop_out_rate)(x)

		x = self.fc_1(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=self.drop_out_rate)(x)

		x = self.fc_2(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=self.drop_out_rate)(x)

		x = self.fc_3(x)
		x = nn.LeakyReLU()(x)
		x = nn.Dropout(p=self.drop_out_rate)(x)

		x = self.fc_out(x)

		return x
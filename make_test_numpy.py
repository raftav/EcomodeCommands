import numpy as np
import os

commands_dict={}

with open('command_mapping.txt','r') as fp:

	row=fp.readline()
	while row:
		cmd_index,cmd_string,cmd_label = row.split('\t')
		commands_dict[cmd_index]=cmd_label.replace('\n','')
		row = fp.readline()

with open('test_irec_files_list.txt') as f:
	test_files_list=f.readlines()

test_files_list=[x.strip() for x in test_files_list]

test_files_list=[x.split('\t')[0] for x in test_files_list]

for file in test_files_list:
	basename=os.path.basename(file).strip('.dat')
	print('converting file {}'.format(basename))
	
	features=np.genfromtxt(file,dtype=float)

	for key,value in commands_dict.items():
		if key in basename:
			label = float(value)

	
	np.save('TEST_FEAT_IREC_NUMPY/features_{}.npy'.format(basename),features,allow_pickle=False)
	np.save('TEST_FEAT_IREC_NUMPY/label_{}.npy'.format(basename),label,allow_pickle=False)

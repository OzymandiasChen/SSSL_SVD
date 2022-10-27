
# coding: utf-8
import os
import config
import numpy as np
import torch

'''
The file is used for frequently used fuctions, thus named as utilis.
	Functions:
		(1). soft2Hard(probOut): Transform probility to tag, the input must be on cpu
		(2). deviceGetter(): Current availabel device getter, 'gpu'/'cpu'.
	Description:
		(a). (1) is model related, could be used for 
		(b). (2) is device realted.
	Using:
		(x). ------------
'''

def soft2Hard(probOut):
	'''
	Transform probility to tag, the input must be on cpu
	Input:
		probOut: model output which could be interpreted as probilities
	Output:
		hardLabel: 0/1 hard label got by probOut and config.THRESHOLD
	'''
	with torch.no_grad():
		return (probOut >= config.THRESHOLD).float()

def deviceGetter():
	'''
	Current availabel device getter, 'gpu'/'cpu'.
	Output:
		device: 'gpu'/'cpu'
	'''
	if(config.GPU_FLAG == False):
		device = torch.device("cpu")
	else:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	return device



if __name__ == '__main__':
	pass
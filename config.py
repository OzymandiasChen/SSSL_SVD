# coding: utf-8


import os
import pathlib

#--------------------------------------------Path Settings---------------------#

PROJECT_PATH = pathlib.Path(__file__).parent.absolute()
JAMENDO_PATH = ''
UNLABEL_PATH = ''

#--------------------------------------------Data Loading Info------------------#
'''
label: sing/nosing: 1/0
FRAME-LEN_HOP-EN_NMEL: '1024_315_80'
'''
PITCH_SHIFTINF_RANGE = 2
TRAIN_STEP = 7
VALID_STEP = 5
TEST_STEP = 3
AUDIO_PROCESSSING_METHOD = '1024_315_80'
AUDIO_PROCESSSING_METHOD_LIST = ['1024_315_80', '2048_512_128']
if AUDIO_PROCESSSING_METHOD == '1024_315_80':
	PROCESSER_DATA_FOLDER_NAME = '1024_315_80'
	SR = 22050	# 1k := 1,000
	FRAME_LEN = 1024 # 0.046s/frame
	HOP_LENGTH = 315 # ~0.014s resolution
	N_MELS = 80
	CUTOFF = 8000  # fmax = 8kHz
	FMIN = 27.5
	FMAX = 8000
	WIN_SIZE = 115  # 1.6 sec
	# CNN_OVERLAP = 7  # Hopsize of 5 for training, 1 for inference
elif AUDIO_PROCESSSING_METHOD == '2048_512_128':
	PROCESSER_DATA_FOLDER_NAME = '2048_512_128'
	SR = 22050	# 1k := 1,000
	FRAME_LEN = 2048 # 0.096s/frame
	HOP_LENGTH = 512 # 0.023s resolution
	N_MELS = 128
	CUTOFF = 8000  # fmax = 8kHz, ... ...
	FMIN = 0
	FMAX = 8192
	WIN_SIZE = 25  # ~0.58s
	# CNN_OVERLAP = 7  # Hopsize of 5 for training, 1 for inference

ENERGY_THRESHOLD = 0.0001

#--------------------------------------------SchCNN Paramaters---------------------------------------------#
DROPOUTRATE = 0.2

#--------------------------------------------Training Settings---------------------------------------------#
THRESHOLD = 0.5  # threshold for binary classification
MODEL_NAME = 'SchCNN'
ZERO_MEAN = True 
OPTIM = 'Adam'	
LR =  0.00001
MOMENTUM = 0.9
MONITOR = 'acc'
SCHEDULER_FLAG = True	# only be useful when the optimizer is 'SGD'
PS_ARG_FLAG = False

TRAINSET_LIST = ['Jamendo']	
VALID_SET_NAME = 'Jamendo' 
TEST_SET_NAME = VALID_SET_NAME


EPOCH_NUM = 192
BATCH_SIZE = 64
BATCH_SIZE_UNLABEL = 64
VALID_BATCH_SIZE = 512
GPU_FLAG = True
FINALLY_SAVED_MODEL = ['lastModel.pkl', 'bestModel_acc.pkl', 'bestModel_loss.pkl'] 
EARLY_STOPPING_EPOCH = EPOCH_NUM + 100

#--------------------------------------------Testing Settigs---------------------------------------------#
# TESTED_MODEL = 'bestModel_acc.pkl'

#--------------------------------------------unlabel data Info---------------------------------------------#
# ['audio_Grammy', 'audio_Oscars_old', audio_GrammyNominees20102020']
# [61, 58, 185]

#--------------------------------------------Experiment Settings-----------------------------------------#
TRAIN_MODE = 'SSL' 

TRAIN_MODE_LIST = ['NaiveNN', 'KD', 'SSL', 'SSSL'] 
if(TRAIN_MODE == 'NaiveNN'):
	pass
elif(TRAIN_MODE == 'KD'): # to validate the similarity hypothesis.
	'''
	KD:= BCE_soft + BCE_hard
	lambda_KD * KD_T * KD_T * BCE_soft + (1 - lambda_KD) * BCE_hard
	'''
	DISTILLER_LIST = ['']
	KD_T = 8
	LAMBDA_KD = 0.5
elif(TRAIN_MODE == 'SSL'): # S2019
	'''
	BCE_hard + BCE_hard
	There is no concept of KD, T. only weight is neeeded.
	'''

	HARD_USING = False
	UNLABEL_FOLDER_LIST = [''] # G1
	DISTILLER_LIST = ['']
	LAMBDA_SSL = 0.7 # lambda_unlabel
elif(TRAIN_MODE == 'SSSL'):
	'''
	Both modes are organized by unlabeled data loss and well-labeled data loss.
	'soft':
		lambda_unlabel * SSL_T * SSL_T * BCE_soft(SSL_T) + (1 - lambda_unlabel) * KDLoss(KD_T, LAMBDA_KD)
	'multi':
		lambda_unlabel * KDLoss(SSL_T, LAMBDA_SSL_KD) + (1 - lambda_unlabel) * KDLoss(KD_T, LAMBDA_KD)
	'''
	UNLABEL_FOLDER_LIST = [''] 
	DISTILLER_LIST = ['']
	KD_T = 8
	LAMBDA_KD = 0.5
	KD_SSL_MODE = 'soft' # 'multi'/'soft' # too troublesome & not necessary.
	if(KD_SSL_MODE == 'soft'):
		# BCE_soft + KD
		SSL_T = 8 # T for unlabeled data.
		LAMBDA_UNLABEL = 0.5 # weight for unlabed data loss.
	elif(KD_SSL_MODE == 'multi'):
		# KD + KD
		SSL_T = 8 # T for unlabeled data.
		LAMBDA_UNLABEL = 0.5 # weight for unlabed data loss.
		LAMBDA_SSL_KD = 0.5 
else:
	pass


#--------------------------------------------Distiller Info ---------------------------------------------#
DISTILLER_INFO = {}
DISTILLER_INFO['g1'] = {'model': 'SchZM', 
						'name': 'g1.pkl',
						'description': 'generation: 1'}
UNLABELDATA_INFO[''] = {'name': '',
						'audio_track_num': ''}


if __name__ == '__main__':
	pass

import os
import config
import argparse
import numpy as np
import librosa
from dataloader.JamendoProcessor import JamendoProcessor
from models.SchCNN import SchCNN
from utils import soft2Hard, deviceGetter
import torch
import torch.nn as nn
import torch.utils.data as Data

'''
Funcs and classes: 
	(1). class Distiller(): distiler model getter oand output getter.
'''
class Distiller():
	'''
	The class is structed with following paramaters and functions.
	Functions:
		(1). __init_(self, phase): ditiller model path 
		(2). distillerGetter(self): get distiller model list.
		(3). getDistillerOutput(self, batch_x): get distiller output.
	Description:
		(a). (3) calls (2) to use.
	Using:
		(a). Distiller().getDistillerOutput(self, batch_x)
		(b). For distiller evaluation, python eval.py --expName 'distillers_eval'
	'''
	def __init__(self):
		'''
		ditiller model path 
		Input:
			----
		Output:
			----
		'''
		self.distillerModelsPath = os.path.join(config.PROJECT_PATH, 'teachers')

	def distillerGetter(self):
		'''
		get distiller model list.
		Input:
			----
		Output:
			distillerModelSet: list of models for the distillers.
		'''
		distillerModelSet = []
		for model_single in config.DISTILLER_LIST:
			distillerModelSet.append(torch.load(os.path.join(self.distillerModelsPath, config.DISTILLER_INFO[model_single]['name'])))
		return distillerModelSet

	def getDistillerOutput(self, batch_x, phase):
		'''
		get distiller output.
		Input:
			batch_x: input x in batch version
			phase: 'KD' | 'SSL_KD' | 'distillers_eval'
		Output:
			output: ditiller ouput in cpu version.
		'''
		if(phase == 'KD'):
			T = config.KD_T
		elif(phase == 'SSL_KD'):
			T = config.SSL_T
		elif(phase in ['SSL']):
			T = 1
		elif(phase == 'distillers_eval'):
			T = 1
		else:
			pass
		singleModelOutputList = []
		distillerModelSet = self.distillerGetter()
		device = deviceGetter()
		for singleModel in distillerModelSet:
			singleModel.eval()			
			singleModel = singleModel.to(device)
			batch_x = batch_x.to(device)
			with torch.no_grad():
				singleModelOutput = singleModel(batch_x, T)
			singleModelOutputList.append(singleModelOutput)
		output = torch.mean(torch.stack(singleModelOutputList), dim=0)
		return output.float()

if __name__ == '__main__':
	pass
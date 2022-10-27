
# coding: utf-8
import os
import sys
import argparse
import numpy as np
from shutil import copyfile
import config
from dataloader.JamendoProcessor import JamendoProcessor
from dataloader.RWCProcessor import RWCProcessor
from dataloader.MIR1KProcessor import MIR1KProcessor
from dataloader.MedleyDBProcessor import MedleyDBProcessor
from dataloader.UnlabelProcessor import UnlabelProcessor
from models.SchCNN import SchCNN
from models.Res_ZeroMean_StdPool import Res_ZeroMean_StdPool
from eval import Evaluator
from utils import deviceGetter, soft2Hard
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score
import losses 
from distiller import Distiller

'''
Funcs and classes: 
	(1). parse_args(): arguement parser for the python file running
	(2). class Trainer(): for training.
'''

def parse_args():
	'''
	arguement parser for the python file running
	argument:
		'--expName': experiment log folder name.
	'''
	description = 'Trainer arguement parser'
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument('--expName',help = '')
	args = parser.parse_args()
	return args

class Trainer():
	'''
	A trainer class. Note that evaluator from eval.py is called for evaluation on valid dataset.
	Functions:
		(1). __init__(self, expName): model, optimizer, scheduler ... ... settings
		(2). lossAccWritter(self, fo, loss, acc, stepIndex, epochIndex, phase): Writter for loss and acc in fo and on screen.
		(3). betterSaver(self, loss_valid, acc_valid, fo): Best model saver according to monitor info.
		(4). trainValidDataLoader(self): trainset daloader and X_valid, Y_valid getter.
		(5). one_pass_train_SL(self, train_loader, epochIndex, fo): One pass runnning operation while training, which will be called for each epoch.
		(6). train(self): Training.
	Description:
		(a). (6) calls (4) and (5) for the main training process
		(b). (2) and (3) are also called by (6), they are auxiliary functions.
	Using:
		(a). python Trainer().train()
	'''
	def __init__(self, expName):
		'''
		model, optimizer, scheduler ... ... settings
		Input: 
			expName: experiment log folder name.
		'''
		# model setting
		if(config.MODEL_NAME == 'SchCNN'):
			# self.model = torch.load(os.path.join(config.PROJECT_PATH, 'logs', '1221_S2018', 'lastModel.pkl'))
			self.model = SchCNN()
		elif(config.MODEL_NAME == 'Res_ZeroMean_StdPool'):
			self.model = Res_ZeroMean_StdPool()
		# optimizer setting
		if(config.OPTIM.lower() == 'adam'):
			self.optimizer = optim.Adam(self.model.parameters(), lr = config.LR)
		elif(config.OPTIM.lower() == 'sgd'):
			self.optimizer = optim.SGD(self.model.parameters(), lr = config.LR, momentum=config.MOMENTUM, nesterov=True)
			if(config.SCHEDULER_FLAG == True):
				if(config.MONITOR == 'loss'):
					self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose = True, factor=0.3, min_lr = 1e-6)
				elif(config.MONITOR == 'acc'):
					self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', verbose = True, factor=0.3, min_lr = 1e-6)
		# As for the criterions, one or some of them might be used, depending on the traininig mode.
		# 不应该这样命名, 那就这样吧
		if(config.TRAIN_MODE in ['NaiveNN']):
			self.criterion_BCE = nn.BCELoss()
		elif(config.TRAIN_MODE in ['KD']):
			self.criterion_KD = losses.KDLoss('KD')
		elif(config.TRAIN_MODE in ['SSL']):
			self.criterion_SSL = losses.SSLLoss_raw()
			# detach()
		elif(config.TRAIN_MODE in ['SSSL']):
			self.criterion_KDSSL = losses.KD_SSLLoss(config.KD_SSL_MODE)
		else:
			pass
		# train info writter and helper.	
		self.bestLoss = float("Inf")
		self.bestACC = -1
		self.bestEpoch = -1
		self.expName = expName
		self.logPath = os.path.join(config.PROJECT_PATH, 'logs', self.expName)
		if not os.path.exists(self.logPath):
			os.makedirs(self.logPath)
		self.writter = SummaryWriter(log_dir = os.path.join(self.logPath, 'tensorboard'))
		self.nonbetterCount = 0

	def lossAccWritter(self, fo, loss, acc, stepIndex, epochIndex, phase):
		'''
		Writter for loss and acc in fo and on screen.
		Input: 
			fo, loss, acc, stepIndex, epochIndex: info
			phase: writter pahse info, 'batch'/'train'
		Output:
			----
		'''
		if(phase.lower() == 'train'):
			print('[Train] ')
			fo.write('[Train] ')
		print('{}/{}: loss: {:.3f}, acc: {:.3f}'.format(stepIndex, epochIndex, loss, acc))
		fo.write('{}/{}: loss: {:.3f}, acc: {:.3f}\n'.format(stepIndex, epochIndex, loss, acc))

	def betterSaver(self, loss_valid, acc_valid, fo):
		'''
		Best model saver according to monitor info.
		Input:
			loss_valid, acc_valid: criterion for deciding whether to save to model.
			fo: info output file.
		Output:
			----
		'''
		if(config.MONITOR == 'acc'):
			if(acc_valid >= self.bestACC):
				self.bestACC = acc_valid
				self.bestLoss = loss_valid
				torch.save(self.model.cpu(), os.path.join(self.logPath, 'bestModel_acc.pkl'))
				print('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
				fo.write('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
				self.nonbetterCount = 0
			else:
				self.nonbetterCount = self.nonbetterCount + 1
		elif(config.MONITOR == 'loss'):
			if(loss_valid <= self.bestLoss):
				self.bestACC = acc_valid
				self.bestLoss = loss_valid
				torch.save(self.model.cpu(), os.path.join(self.logPath, 'bestModel_loss.pkl'))
				print('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
				fo.write('[Model Saved!!]~~~~~~~~~~~~~~~~~~\n')
				self.nonbetterCount = 0
			else:
				self.nonbetterCount = self.nonbetterCount + 1
		if(self.nonbetterCount == config.EARLY_STOPPING_EPOCH):
			print('[EARLY STOPPING!!]\n')
			fo.write('[EARLY STOPPING!!]\n')
			return True
		return False # continue flag

	def trainValidDataLoader(self):
		'''
		trainset daloader and X_valid, Y_valid getter.
		Input:
			----
		Output:
			train_loader: dataloader for the trainset.
			X_valid, Y_valid: valid dataset.
		'''
		# Train Loading ... ...
		X_train = []
		Y_train = []
		if('Jamendo' in config.TRAINSET_LIST):
			JamendoDataLoader = JamendoProcessor()
			X_train_Jamendo, Y_train_Jamnedo = JamendoDataLoader.datasetLoader('train', PS = config.PS_ARG_FLAG)
			X_train.append(X_train_Jamendo)
			Y_train.append(Y_train_Jamnedo)
		if('RWC' in config.TRAINSET_LIST or 'RWC_part' in config.TRAINSET_LIST): # or 或者
			RWCDataLoader = RWCProcessor('RWC-MDB-P-2001')
			X_train_RWC, Y_train_RWC = RWCDataLoader.datasetLoader(config.RWC_TRAIN, 'train')
			X_train.append(X_train_RWC)
			Y_train.append(Y_train_RWC)
		if('MIR1K' in config.TRAINSET_LIST or 'MIR1K_part' in config.TRAINSET_LIST): # or 或者
			MIR1KDataLoader = MIR1KProcessor()
			X_train_MIR1K, Y_train_MIR1K = MIR1KDataLoader.datasetLoader(config.MIR1K_TRAIN, 'train')
			X_train.append(X_train_MIR1K)
			Y_train.append(Y_train_MIR1K)
		if('MedleyDB' in config.TRAINSET_LIST or 'MedleyDB_part' in config.TRAINSET_LIST): # or 或者
			MedleyDBLoader = MedleyDBProcessor()
			X_train_MedleyDB, Y_train_MedleyDB = MedleyDBLoader.datasetLoader(config.MedleyDB_TRAIN, 'train')
			X_train.append(X_train_MedleyDB)
			Y_train.append(Y_train_MedleyDB)
		X_train = torch.cat(X_train, axis = 0)
		Y_train = torch.cat(Y_train, axis = 0)
		print('----------Train Loading:{}/{}----------'.format(X_train.shape, Y_train.shape))
		train_torch_dataset = Data.TensorDataset(X_train, Y_train)
		train_loader = Data.DataLoader(dataset = train_torch_dataset, batch_size = config.BATCH_SIZE, shuffle = True)
		# Valid Loading ... ...
		if(config.VALID_SET_NAME == 'Jamendo'):
			JamendoDataLoader = JamendoProcessor()
			X_valid, Y_valid = JamendoDataLoader.datasetLoader('valid')
		elif(config.VALID_SET_NAME == 'RWC_part'):
			RWCDataLoader = RWCProcessor('RWC-MDB-P-2001')
			X_valid, Y_valid = RWCDataLoader.datasetLoader(config.RWC_VALID, 'valid')
		print('----------Valid Loading:{}/{}----------'.format(X_valid.shape, Y_valid.shape))
		# Unlabel Loading ... ...
		if(config.TRAIN_MODE in ['SSL', 'SSSL', 'DA_KD_SSL']):
			UnlabelDataLoader = UnlabelProcessor()
			X_unlabel = UnlabelDataLoader.datasetLoader()
			print('----------Unlabel Loading:{}----------'.format(X_unlabel.shape))
			unlabel_loader = Data.DataLoader(dataset = X_unlabel, batch_size = config.BATCH_SIZE_UNLABEL, shuffle = True)
			return train_loader, X_valid, Y_valid, unlabel_loader
		return train_loader, X_valid, Y_valid, None

	def one_pass_train(self, train_loader, epochIndex, fo, unlabel_loader = None):
		'''
		One pass runnning operation while training for the supervised mode, which will be called for each epoch.
		Input:
			train_loader: dataloder for the trainset
			epcochIndex: epoch index
			fo: info writter file
			unlabel_loader: default as None
		Output:
			----
		'''
		device = deviceGetter()
		Distillers = Distiller() # might not be used.
		epoch_loss = 0.0
		epoch_acc = 0.0
		if config.TRAIN_MODE in ['SSSL', 'DA_KD_SSL', 'SSL']:
			unlabelDataloader_iterator = iter(unlabel_loader)
			minDataloaderLen = min(len(train_loader), len(unlabel_loader))
		self.model.train()
		for step, (batch_x, batch_y) in enumerate(train_loader): # 只要迭代 就说明 train_loader里有东西
			self.model.zero_grad()
			batch_x, batch_y = batch_x.to(device), batch_y.to(device)
			output_train = self.model(batch_x, 1) # must be put
			if(config.TRAIN_MODE == 'NaiveNN'):
				loss = self.criterion_BCE(output_train, batch_y.float())
			elif(config.TRAIN_MODE in ['KD']): # config.TRAIN_MODE in ['KD', 'DA_KD']
				output_train_soft = self.model(batch_x, config.KD_T)
				softLabel_train = Distillers.getDistillerOutput(batch_x, 'KD').to(device)
				loss = self.criterion_KD(output_train, output_train_soft, batch_y.float(), softLabel_train.detach())
			elif(config.TRAIN_MODE in ['SSL']): # config.TRAIN_MODE in ['KD_SSL', 'DA_KD_SSL']
				# load unlabel data
				try:
					unlabeldata = next(unlabelDataloader_iterator)
				except StopIteration:
					unlabelDataloader_iterator = iter(unlabel_loader)
					unlabeldata = next(unlabelDataloader_iterator)
				unlabeldata = unlabeldata.to(device)
				# forward, labeled data has already been feeded.
				output_unlabel_hard = self.model(unlabeldata, 1) # output_unlabel = self.model(unlabeldata)
				pesudo_unlabel = Distillers.getDistillerOutput(unlabeldata, 'SSL').to(device) # pesudo_softlabel
				if(config.HARD_USING == True):
					pesudo_unlabel = soft2Hard(pesudo_unlabel)
				# calcu loss
				loss = self.criterion_SSL(output_train, output_unlabel_hard, batch_y.float(), pesudo_unlabel.detach())	
			elif(config.TRAIN_MODE in ['SSSL']): # config.TRAIN_MODE in ['KD_SSL', 'DA_KD_SSL']
				# for the labeled data
				output_train_soft = self.model(batch_x, config.KD_T)
				softLabel_train = Distillers.getDistillerOutput(batch_x, 'KD').to(device)
				# for the unlabeldata
				try:
					unlabeldata = next(unlabelDataloader_iterator)
				except StopIteration:
					unlabelDataloader_iterator = iter(unlabel_loader)
					unlabeldata = next(unlabelDataloader_iterator)
				unlabeldata = unlabeldata.to(device)
				output_unlabel_soft = self.model(unlabeldata, config.SSL_T)
				softLable_unlabel = Distillers.getDistillerOutput(unlabeldata, 'SSL_KD').to(device)
				# def forward(self, output_1, output_T, hardLabel, softLabel, output_unlabel_T, softLabel_unlabel, output_unlabel_1 = None, pesudoHardLabel_unlabel = None):
				if(config.KD_SSL_MODE == 'soft'):
					loss = self.criterion_KDSSL(
						output_train, output_train_soft, batch_y.float(), softLabel_train.detach(), 
						output_unlabel_soft, softLable_unlabel.detach())
				elif(config.KD_SSL_MODE == 'multi'): # an option that I will never use again in my life
					output_unlabel = self.model(unlabeldata, 1)
					pesudoHardLabel_unlabel = soft2Hard(softLable_unlabel) # no matter which T it is.
					loss = self.criterion_KDSSL(
						output_train, output_train_soft, batch_y.float(), softLabel_train.detach(), 
						output_unlabel_soft, softLable_unlabel.detach(), output_unlabel, pesudoHardLabel_unlabel.float())
				else:
					pass
			else:
				pass

			loss.backward()
			self.optimizer.step()
			acc = accuracy_score(batch_y.cpu(), soft2Hard(output_train.cpu()))
			epoch_loss += loss.item() * batch_x.shape[0]
			epoch_acc += acc * batch_x.shape[0]
			# print perfromance 5 times a epoch
			if(step % (len(train_loader) // 8) == 0):
				self.lossAccWritter(fo, loss.item(), acc, step, epochIndex, 'batch')
				self.writter.add_scalar('Loss/batch', loss.item(), epochIndex*len(train_loader)+step)
				self.writter.add_scalar('Acc/batch', acc, epochIndex*len(train_loader)+step)
				sys.stdout.flush()
		torch.save(self.model.cpu(), os.path.join(self.logPath, 'lastModel.pkl'))
		return epoch_loss / len(train_loader.dataset), epoch_acc / len(train_loader.dataset)

	def train(self):
		'''
		Training
		Input:
			----
		Output:
			----
		'''
		copyfile(os.path.join(config.PROJECT_PATH, 'config.py'), os.path.join(self.logPath, 'config.py'))
		copyfile(os.path.join(config.PROJECT_PATH, 'models', config.MODEL_NAME + '.py'), os.path.join(self.logPath, config.MODEL_NAME + '.py'))
		validEvaluator = Evaluator('Valid')
		device = deviceGetter()
		# self.model = self.model.to(device)\
		train_loader, X_valid, Y_valid, unlabel_loader = self.trainValidDataLoader()
		# unlabel_loader might be None when config.TRAIN_MODE in ['NaiveNN', 'KD', 'DA_KD']
		fo = open(os.path.join(self.logPath, 'trainLog.txt'), 'w+')
		if(config.TRAIN_MODE == 'NaiveNN'):
			print('mode:{}, '.format(config.TRAIN_MODE))
		elif(config.TRAIN_MODE in ['KD']):
			print('mode:{}, KD_T:{}, lambda_KD'.format(config.TRAIN_MODE, config.KD_T, config.LAMBDA_KD))
		elif(config.TRAIN_MODE in ['SSL']):
			print('mode:{}, lambda_SSL:{}'.format(config.TRAIN_MODE, config.LAMBDA_SSL))
		elif(config.TRAIN_MODE in ['SSSL']):
			print('mode:{}, KD_T:{}, SSL_T:{}, lambda_KD:{}, lambda_un:{}'.format(
				config.TRAIN_MODE, config.KD_T, config.SSL_T, config.LAMBDA_KD, config.LAMBDA_UNLABEL))
		else:
			pass
		sys.stdout.flush()
		for epoch in range(config.EPOCH_NUM):
			# print("Train{}:{}\n".format(epoch, next(self.model.parameters()).is_cuda))
			self.model = self.model.to(device)
			loss_train, acc_train = self.one_pass_train(train_loader, epoch, fo, unlabel_loader)
			self.lossAccWritter(fo, loss_train, acc_train, '--', epoch, 'train')
			_, loss_valid, acc_valid, _, _, _, _, _, _, _ = validEvaluator.evaluation(X_valid, Y_valid, self.model, fo)
			self.writter.add_scalars('Loss/epoch', {'train': loss_train, 
													'valid': loss_valid.item()}, epoch)
			self.writter.add_scalars('Acc/epoch', {'train': acc_train, 
													'valid': acc_valid}, epoch)
			ESFlag = self.betterSaver(loss_valid, acc_valid, fo)
			if(ESFlag == True):
				break
			if(config.OPTIM.lower() == 'sgd' and config.SCHEDULER_FLAG == True):
				if(config.MONITOR == 'acc'):
					self.scheduler.step(acc_valid)
				elif(config.MONITOR == 'loss'):
					self.scheduler.step(loss_valid)
			self.writter.flush()
		self.writter.close()
		fo.close()

if __name__ == '__main__':
	args = parse_args()
	trainer = Trainer(args.expName)
	# trainer.trainValidDataLoader()
	trainer.train()
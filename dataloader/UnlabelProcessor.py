
import os
import sys
sys.path.append(sys.path[0][:-10])
import config
import librosa
import torch
import numpy as np
import random
import soundfile as sf
import argparse

'''
Funcs and classes: 
	(1). class UnlabelProcessor(): data procesor and loader for unlabled dataset.
	(2). function parse_args(): arguement parser for what you would a UnlabelProcessor do.
'''

class UnlabelProcessor():
	'''
	 data procesor and loader for unlabled dataset.
	 Funtions:
	 	(1). __init__(self): init func with no operations.
	 	(2). loadSaveMel_Song(self, audioFolderName): make and save song level melSpectro in pkl.
	 	(3). raw2mel(self): make and save song level melSpectro for all raw audio folders.
	 	(4). makeFrameData_Song(self, audioFolderName, audioFileName): Make frame-level data for a certain song.
	 	(5). loadFrameData_Folder(self, audioFolderName): make frame-level data for a folder.
	 	(6). datasetLoader(self): datasetloader.
	 Description:
		(a). (3) calls (2) to process all raw data 2 mel.
		(b). (5) calls (4) to make frame level data for a audio in a certain dataset.
		(c). (6) calls (5) to make the dataset for training.
	Using:
		(a). call 'UnlabelProcessor.raw2mel()' for preparing operations.
		(b). call 'UnlabelProcessor.dataLoader()' for loading data.
		(c). call 'UnlabelProcessor.loadSaveMel_Song(audioFolderName)' for preparing operations for a certain folder.
	'''

	def __init__(self):
		'''
		init func with no operations.
		'''
		pass

	def loadSaveMel_Song(self, audioFolderName):
		'''
		make and save song level melSpectro in pkl
		Input:
			audioFolderName: raw audio folder name 
		Output:
			----
		'''
		melFolderPath = os.path.join(config.UNLABEL_PATH, config.PROCESSER_DATA_FOLDER_NAME, 'mel_' + audioFolderName)
		if not os.path.exists(melFolderPath):
			os.makedirs(melFolderPath)
		for audioFileName in os.listdir(os.path.join(config.UNLABEL_PATH, audioFolderName)):
			audioFilePath = os.path.join(config.UNLABEL_PATH, audioFolderName, audioFileName)
			y, _ = librosa.load(audioFilePath, config.SR)
			melSpectro = librosa.feature.melspectrogram(y, sr=config.SR, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH, 
				window='hann', n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=1.0)
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, os.path.join(melFolderPath, audioFileName.split('.')[0]+'.pkl'))

	def raw2mel(self):
		'''
		make and save song level melSpectro for all raw audio folders
		'''
		for folderName in os.listdir(config.UNLABEL_PATH):
			if(folderName.split('_')[0] == 'audio'):
				self.loadSaveMel_Song(folderName)

		# with open(os.path.join(config.UNLABEL_PATH, 'audioFolderNameList.txt') , 'r') as fo:
		# 	for line in fo.readlines():
		# 		audioFolderName = line.strip('\n')[0]
		# 		self.loadSaveMel_Song(audioFolderName)
	##############################################################

	def makeFrameData_Song(self, audioFolderName, audioFileName):
		'''
		Make frame-level data for a certain song.
		Input:
			audioFileName, audioFileName: song information.
		Output:
			x: frame level data in torch tensor version.
		'''
		melSeries = torch.load(
			os.path.join(config.UNLABEL_PATH, config.PROCESSER_DATA_FOLDER_NAME, 'mel_' + audioFolderName, audioFileName.split('.')[0]+'.pkl'))
		x=[]
		for winIndex in range(0, melSeries.shape[1] - config.WIN_SIZE + 1, config.TRAIN_STEP):
			xWin = melSeries[:, winIndex: winIndex + config.WIN_SIZE]
			x.append(xWin)
		x = np.array(x)
		x = torch.from_numpy(x)
		x = x.float()
		x = x.unsqueeze(1)
		return x

	def loadFrameData_Folder(self, audioFolderName):
		'''
		make frame-level data for a folder
		Input:
			audioFolderName: ----
		Output:
			x_folder: frame-level data for a folder in torch version.
		'''
		x_folder = []
		for audioFileName in os.listdir(os.path.join(config.UNLABEL_PATH, config.PROCESSER_DATA_FOLDER_NAME, 'mel_' + audioFolderName)):
			x_song = self.makeFrameData_Song(audioFolderName, audioFileName)
			x_folder.append(x_song)
		x_folder = torch.cat(x_folder, axis = 0)
		return x_folder

	def datasetLoader(self):
		'''
		dataloader.
		Input:
			----
		Output:
			X: the loaded frame-level data (for raw auedio in several dataset) 
		'''
		X = []
		for audioFolderName in config.UNLABEL_FOLDER_LIST:
			x_folder = self.loadFrameData_Folder(audioFolderName)
			X.append(x_folder)
		X = torch.cat(X, axis = 0)
		print('-----------------Unlabel Dataset Loading: {}-----------'.format(config.UNLABEL_FOLDER_LIST))
		print(X.shape)
		return X

def parse_args():
	'''
	arguement parser for what you would a UnlabelProcessor do.
	--audioFolderProcessor: if you would like to process the raw folder in a certain dataset.
	--audio2mel: if you would like to process all raw data.
	--load: loading data from certain folders.
	'''
	description = 'options'
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument('--audioFolderProcessor', default = None, help = 'audio in which folder you would like to process')
	parser.add_argument('--audio2mel', default = False, help = 'process for all raw data')
	parser.add_argument('--load', default = False, help = 'audio in which folder you would like to process')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	unlabelProcessor = UnlabelProcessor()
	if(args.audioFolderProcessor != None):
		unlabelProcessor.loadSaveMel_Song(args.audioFolderProcessor)
	if(args.audio2mel != False):
		unlabelProcessor.raw2mel()
	if(args.load != False):
		X = unlabelProcessor.datasetLoader()

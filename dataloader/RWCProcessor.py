
import os
import sys
import argparse
sys.path.append(sys.path[0][:-10])
import config
import librosa
import torch
import numpy as np
import random
import soundfile as sf
import datetime

'''
To a certain extent, the audio files are originally grouped by artists.
audio file  num: {1:16, 2:16, 3:16. 4:16, 5:16, 6:10, 7:10}
RWC/
|-- RWC-MDB-P-2001/
|	|-- 1024_315_80/
|		|-- label/
|		|-- mel/
|		|-- mel_PS2/
|	|-- AIST.RWC-MDB-P-2001.VOCA_INST
|	|-- RWC研究用音楽データベース Disc 1
|		|-- 01 永遠のレプリカ.wav
|		|-- 02 Magic in your eyes.wav
|		|-- ... ...
|		|-- 16 Game of Love.wav
|		|-- desktop.ini
|	|-- RWC研究用音楽データベース Disc 2
|	|-- ... ...
|	|-- RWC研究用音楽データベース Disc 7
|-- RWC-MDB-R-2001/
|-- filelists/
'''

class RWCProcessor():
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). __init__(self): Initializer.
			(2). getFileNameList_folderlevel(self, folderName): Getting ordered file name list for a certain audio folder.
			(3). trackIndex2Location(self, trackIndex): From trackIndex to folderIndex and audioFileIndex.
			(4). location2TrackIndex(self, folderName, audioFileName): From location(folderName, audioFileName) to track index.
			(5). geneAudio_PS(self, dataType): Generate pitch shifted wave file. // Data Arguenment.
			(6). raw2mel(self, dataType): Generate melSpectro for all songs.
			(7). txt2label(self): Generate label for all song with corresponding info logger.
			(8). splitDataSet(self): split dataset into 5 parts, preparsion for future cross-validation.
			(9). loadFileList(self, trackIndexListFile_Group): Loading trackIndexList using given files.
			(10). makeFramePair_Song(self, dataType, trackIndex, step): Make frame-wise paired data for a certain song (given the trackIndex).
			(11). datasetLoader(self, trackIndexListFile_Group, phase): 
					DataSet Loading for songs in trackIndexListFile_Group. If phase equals 'test', it is also loaded in songlevel.
		Description:
			(a). (2), (3), (4) are indexing helping functions. They will be called frequently.
			(b). (5), (6), (7) and (8) are data preparing funtions.
			(c). (11) calls (10) to make dataset
		Using:
			(a). If it is the 1st time to load data, the following preparing work should be done.
				 Call 'RWCProcessor().geneAudio_PS(self, dataType)' for preparing arguemented data.
				 Call 'RWCProcessor().raw2mel(self, dataType)' for preparing mel data.
				 Call 'RWCProcessor().txt2label(self)' for preparing labels.
				 Call 'RWCProcessor().splitDataSet(self)' for spliting the dataset.
			(b). Call 'RWCProcessor().datasetLoader(self, trackIndexListFile_Group, phase)' for loading a certain dataset.
	'''

	def __init__(self, audioFolder):
		'''
		Initializer.
		Description:
			trackIndex, folderIndex, audioFileIndex all begin with 0, no matter how the file is named.
			trackIndex/audioFileIndex: whole/folder_level.
			The arguemented datafolder has the same structure.
		Input:
			audioFolder:
		Output:
			----
		'''
		self.AD_Folder = audioFolder	# AD_Folder: audio folder, 'RWC-MDB-P-2001'
		if(self.AD_Folder == 'RWC-MDB-P-2001'):
			self.ADLabel_Folder = 'AIST.RWC-MDB-P-2001.VOCA_INST'
		# audioFolderNameList: e. g.['RWC研究用音楽データベース Disc 1', ... ..., 'RWC研究用音楽データベース Disc 7']
		self.audioFolderNameList = sorted([x for x in os.listdir(os.path.join(config.RWC_PATH, self.AD_Folder)) if x[:3] == 'RWC'])
		self.discardedTrackIndex = [83]	# trackIndex with 83(84, started with 1) should be discarded.

	def getFileNameList_folderlevel(self, folderName):
		'''
		Getting ordered file name list for a certain audio folder
		Input:
			folderName: e.g. 'RWC研究用音楽データベース Disc 3'
		Output: 
			As dicpited above. e. g. ['01 永遠のレプリカ.wav', ... ..., '16 Game of Love.wav']
		'''
		return sorted([x for x in os.listdir(os.path.join(config.RWC_PATH, self.AD_Folder, folderName)) if x.split('.')[-1]!='ini'])

	def trackIndex2Location(self, trackIndex):
		'''
		From trackIndex to folderIndex and audioFileIndex.
		Input:
			trackIndex: begin from 0.
		Output:
			folderIndex, audioFileIndex: begin from 0.
		'''
		if trackIndex in range(80):
			folderIndex = trackIndex // 16
			audioFileIndex = trackIndex % 16
			# 34 -> [2, 2]
		elif trackIndex in range(80, 100):
			trackIndex = trackIndex - 80
			folderIndex = trackIndex // 10 + 5
			audioFileIndex = trackIndex % 10
			# 84 -> 4 -> [5, 4]
		folderName = self.audioFolderNameList[folderIndex]
		audioFileName = self.getFileNameList_folderlevel(folderName)[audioFileIndex]
		# sorted([x for x in os.listdir(os.path.join(config.RWC_PATH, self.AD_Folder, folderName)) if x.split('.')[-1]!='ini'])
		return folderIndex, audioFileIndex

	def location2TrackIndex(self, folderName, audioFileName):
		'''
		From location(folderName, audioFileName) to track index.
		Input:
			folderName, audioFileName: realName, originally begin from 1.
		Output:
			trackIndex: begin from 0. 
		'''
		# trasnform to index(begin with 0)
		folderIndex = int(folderName.split(' ')[-1]) - 1
		audioFileIndex = int(audioFileName.split(' ')[0]) - 1
		if folderIndex in range(5):
			# [3, 3] -> [2, 2] -> 34
			trackIndex = folderIndex * 16 + audioFileIndex
		else:
			# [6, 5] -> [5, 4] -> 84
			trackIndex = (folderIndex - 5) * 10 + audioFileIndex + 80
		return trackIndex

	def geneAudio_PS(self, dataType):
		'''
		Generate pitch shifted wave file. // Data Arguenment.
		Input:
			dataType: dataArguement type.
		Output:
			----
		'''
		orgAudioFolderPath = os.path.join(config.RWC_PATH, self.AD_Folder)	# '/RWC/RWC-MDB-P-2001'
		tarAudioFolderPath = os.path.join(config.RWC_PATH, self.AD_Folder) + '_' + dataType # '/RWC/RWC-MDB-P-2001_PS2'
		for audioFolderName in self.audioFolderNameList:
			if not os.path.exists(os.path.join(tarAudioFolderPath, audioFolderName)):
				os.makedirs(os.path.join(tarAudioFolderPath, audioFolderName))
		for trackIndex in range(100):
			subFolderIndex, audioFileIndex = self.trackIndex2Location(trackIndex)
			subFolderName = self.audioFolderNameList[subFolderIndex]
			audioFileName = self.getFileNameList_folderlevel(subFolderName)[audioFileIndex]
			orgAudioFilePath = os.path.join(orgAudioFolderPath, subFolderName, audioFileName)
			tarAudioFilePath = os.path.join(tarAudioFolderPath, subFolderName, audioFileName)
			y_source, _ = librosa.load(orgAudioFilePath, config.SR)
			y_PS = librosa.effects.pitch_shift(
				y_source, config.SR, n_steps = random.randint(-1 * config.PITCH_SHIFTINF_RANGE, config.PITCH_SHIFTINF_RANGE), bins_per_octave=12)
			sf.write(tarAudioFilePath, y_PS, config.SR, subtype='PCM_24')
			print('{}:{}'.format(subFolderName, audioFileName))

	def raw2mel(self, dataType):
		'''
		Generate melSpectro for all songs.
		Input:
			dataType: (['raw', 'PS2'])
		'''
		audioFolderPath = os.path.join(config.RWC_PATH, self.AD_Folder)	# '/RWC/RWC-MDB-P-2001'
		targetFolderPath = os.path.join(config.RWC_PATH, self.AD_Folder, config.AUDIO_PROCESSSING_METHOD, 'mel') # '/RWC/1024_315_80/RWC-MDB-P-2001/mel'
		if(dataType != 'raw'):
			audioFolderPath = audioFolderPath + '_' + dataType	# '/RWC/RWC-MDB-P-2001_PS2'
			targetFolderPath = targetFolderPath + '_' + dataType # '/RWC/1024_315_80/RWC-MDB-P-2001/mel_PS2'
		if not os.path.exists(targetFolderPath):
			os.makedirs(targetFolderPath)
		for trackIndex in range(100):
			subFolderIndex, audioFileIndex = self.trackIndex2Location(trackIndex)
			subFolderName = self.audioFolderNameList[subFolderIndex]
			audioFileName = self.getFileNameList_folderlevel(subFolderName)[audioFileIndex]
			audioFilePath = os.path.join(audioFolderPath, subFolderName, audioFileName)
			targetFilePath = os.path.join(targetFolderPath, str(trackIndex).zfill(2) + '.pkl') # str(1).zfill(2)
			y, _ = librosa.load(audioFilePath, config.SR)
			melSpectro = librosa.feature.melspectrogram(y, sr=config.SR, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH, 
				window='hann', n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=1.0)
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, targetFilePath)
			print('{}: {}, {}, {}'.format(trackIndex, subFolderName, audioFileName, logMelSpectro.shape))

	def txt2label(self):
		'''
		Generate label for all song with corresponding info logger.
		'''
		labelTXTFolder = os.path.join(config.RWC_PATH, self.AD_Folder, self.ADLabel_Folder)
		# '/RWC/RWC-MDB-P-2001/AIST.RWC-MDB-P-2001.VOCA_INST'
		labelTXTNameList = sorted([x for x in os.listdir(labelTXTFolder) if x[:2] == 'RM'])
		labelPKLFolderPath = os.path.join(config.RWC_PATH, self.AD_Folder, config.AUDIO_PROCESSSING_METHOD, 'label')
		if not os.path.exists(labelPKLFolderPath):
			os.makedirs(labelPKLFolderPath)
		fo_info = open(os.path.join(config.RWC_PATH, self.AD_Folder, config.AUDIO_PROCESSSING_METHOD, 'raw_data_loading_info.txt'), 'w+')
		for trackIndex in range(100):
			logMel = torch.load(os.path.join(config.RWC_PATH, self.AD_Folder, config.AUDIO_PROCESSSING_METHOD, 'mel', str(trackIndex).zfill(2) + '.pkl'))
			label = np.zeros((logMel.shape[-1],))
			with open(os.path.join(labelTXTFolder, labelTXTNameList[trackIndex]), 'r') as fo:
				lines = fo.readlines()
				for lineIndex in range(len(lines)):
					line = lines[lineIndex].strip('\n').split('\t')
					if(line[1][0] in ['f', 'm', 'g']):
						start = librosa.time_to_frames(float(line[0]), sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
						endTimeStamp = lines[lineIndex+1].split('\t')[0]
						end = librosa.time_to_frames(float(endTimeStamp), sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
						label[start: end + 1] = 1
						# print('	T:{}, start:{}, end:{}'.format(line[0], start, end))
					elif(line[1][0] == 'e'):
						finalTimeIndex = librosa.time_to_frames(float(line[0]), sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
			torch.save(label, os.path.join(labelPKLFolderPath, str(trackIndex).zfill(2) + '.pkl'))
			# Logging ... ...
			subFolderIndex, audioFileIndex = self.trackIndex2Location(trackIndex)
			# subFolderName = self.audioFolderNameList[subFolderIndex]
			# audioFileName = self.getFileNameList_folderlevel(subFolderName)[audioFileIndex]
			print('{}: folder_{}, file_{}, dif:{}, {}/{}, {:.3f}'.format(
				trackIndex, subFolderIndex, audioFileIndex, len(label)-finalTimeIndex, len(label), sum(label), sum(label)/len(label)))
			fo_info.write('{}: folder_{}, file_{}, dif:{}, {}/{}, {:.3f}\n'.format(
				trackIndex, subFolderIndex, audioFileIndex, len(label)-finalTimeIndex, len(label), sum(label), sum(label)/len(label)))
		fo_info.close()

	def splitDataSet(self):
		'''
		Split dataset into 5 parts, preparsion for future cross-validation.
		The filelistNames are ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
		'''
		trackIndexList = [x for x in range(100) if not x in self.discardedTrackIndex]
		random.shuffle(trackIndexList)
		filelistsFolder = os.path.join(config.RWC_PATH, self.AD_Folder, 'filelists')
		if not os.path.exists(filelistsFolder):
			os.makedirs(filelistsFolder)
		for i in range(5):
			fo = open(os.path.join(filelistsFolder, '%s.txt' % (str(i))), 'w+')
			for trackIndex in trackIndexList[i*20: (i+1)*20]:
				fo.write('{}\n'.format(trackIndex))
			fo.close()

	def loadFileList(self, trackIndexListFile_Group):
		'''
		Loading trackIndexList using given files.
		Input:
			trackIndexListFile_Group: filelist names organized in a list. e.g. [0, 1, 4]
		Output:
			trackIndexList: ----
		'''
		trackIndexList = []
		for trackIndexList_FileName in trackIndexListFile_Group:
			with open(os.path.join(config.RWC_PATH, self.AD_Folder, 'filelists', str(trackIndexList_FileName) + '.txt'), 'r') as fo:
				for line in fo.readlines():
					trackIndexList.append(int(line.strip('\n')))
		# print(trackIndexList, '\n', len(trackIndexList))
		return trackIndexList

	def makeFramePair_Song(self, dataType, trackIndex, step):
		'''
		Make frame-wise paired data for a certain song (given the trackIndex).
		Input:
			dataType: (['raw', 'PS2']), trackIndex: (begin with 0), step
		Output:
			x, y: frame-level paired data per song in tensor version.
		'''
		melFolderPath = os.path.join(config.RWC_PATH, self.AD_Folder, config.AUDIO_PROCESSSING_METHOD, 'mel')
		labelFolderPath = os.path.join(config.RWC_PATH, self.AD_Folder, config.AUDIO_PROCESSSING_METHOD, 'label')
		if(dataType != 'raw'):
			melFolderPath = melFolderPath + '_' + dataType
		# , str(trackIndex).zfill(2) + '.pkl'
		melSeries = torch.load(os.path.join(melFolderPath, str(trackIndex).zfill(2) + '.pkl'))
		labelSeries = torch.load(os.path.join(labelFolderPath, str(trackIndex).zfill(2) + '.pkl'))
		x = []
		y = []
		# a window corresponding to the central frame in the win; In other word, a window represents the frame. 
		# winIndex + config.WIN_SIZE - 1 <= melSeries.shape[1] -1
		for winIndex in range(0, melSeries.shape[1] - config.WIN_SIZE + 1, step):
			xWin = melSeries[:, winIndex: winIndex + config.WIN_SIZE]
			yWin = labelSeries[winIndex + config.WIN_SIZE//2]
			x.append(xWin)
			y.append(yWin)
		# process data to target format
		x = np.array(x)
		y = np.array(y)
		x = torch.from_numpy(x)
		y = torch.from_numpy(y)
		x = x.float()
		y = y.long()
		x = x.unsqueeze(1)
		# print(x.shape, y.shape)
		return x, y

	def datasetLoader(self, trackIndexListFile_Group, phase):
		'''
		DataSet Loading for songs in trackIndexListFile_Group. If phase equals 'test', it is also loaded in songlevel.
		Input:
			trackIndexListFile_Group: filelist names organized in a list. e.g. [0, 1, 4]
			phase: phase determines the step and what form the dataset is returned. 'train'/'valid'/'test'.
		Output: 
			X, Y/ X, Y, data_label_dict: when phase in ['train', 'valid']/ when phase == 'test'
		'''
		X, Y, dataTypeList = [], [], []
		dataTypeList.append('raw')
		if(phase.lower() == 'train'):
			step = config.TRAIN_STEP
			if(config.PS_ARG_FLAG == True):
				# pass
				dataTypeList.append('PS'+str(config.PITCH_SHIFTINF_RANGE))
			else:
				pass
		elif(phase.lower() == 'valid'):
			step = config.VALID_STEP
		else: 
			# 'test'
			step = config.TEST_STEP
		trackIndexList = self.loadFileList(trackIndexListFile_Group)
		for trackIndex in trackIndexList:
			for dataType in dataTypeList:
				x, y = self.makeFramePair_Song(dataType, trackIndex, step)
				X.append(x)
				Y.append(y)
		X = torch.cat(X, axis=0)
		Y = torch.cat(Y, axis=0)
		print('-----------------RWC Dataset Loading {}:{}--------------------------'.format(phase, dataType))
		print(X.shape, Y.shape)
		if(phase.lower() == 'train' or phase.lower() == 'valid'):
			return X, Y
		# phase == 'test'
		data_label_dict = {}
		for trackIndex in trackIndexList:
			x, y = self.makeFramePair_Song('raw', trackIndex, step)
			subFolderIndex, audioFileIndex = self.trackIndex2Location(trackIndex)
			subFolderName = self.audioFolderNameList[subFolderIndex]
			audioFileName = self.getFileNameList_folderlevel(subFolderName)[audioFileIndex]
			data_label_dict['{}/{}'.format(subFolderName, audioFileName)] = [x, y]
		# e.g. {'RWC研究用音楽データベース Disc 1'/'01 永遠のレプリカ.wav': [x_song, y_song], }
		# print(len(data_label_dict))
		return X, Y, data_label_dict


def parse_args():
	'''
	'''
	description = 'options'
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument('--option',help = '')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	RWCPrcs= RWCProcessor('RWC-MDB-P-2001')
	# The following options are used for data preparasion.
	if(args.option == 'rawDataProcessing'):
		RWCPrcs.splitDataSet()
		RWCPrcs.raw2mel('raw')
		RWCPrcs.txt2label()
	elif(args.option == 'PS_processing'):
		# ongoing ... ...
		RWCPrcs.geneAudio_PS('PS2')
		RWCPrcs.raw2mel('PS2')
	elif(args.option == 'datasetSplit'):
		pass
	# RWCPrcs.datasetLoader([0,1,2,3,4], 'train')
	# trackIndexListFile_Group_train = [0, 1, 2]
	# trackIndexListFile_Group_valid = [3]
	# trackIndexListFile_Group_test = [4]
	# RWCPrcs.datasetLoader(trackIndexListFile_Group_train, 'train')
	# RWCPrcs.datasetLoader(trackIndexListFile_Group_valid, 'valid')
	# RWCPrcs.datasetLoader(trackIndexListFile_Group_test, 'test')
	# for i in range(100):
	# 	folderIndex, audioFileIndex = RWCPrcs.trackIndex2Location(i)
	# 	folderName = RWCPrcs.audioFolderNameList[folderIndex]
	# 	audioFileName = RWCPrcs.getFileNameList_folderlevel(folderName)[audioFileIndex]
	# 	trackIndex = RWCPrcs.location2TrackIndex(folderName, audioFileName)
	# 	print('{}/{}: {},{}'.format(i + 1, trackIndex + 1, folderName, audioFileName))

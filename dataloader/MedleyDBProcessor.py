
import os
import sys
import argparse
sys.path.append(sys.path[0][:-10])
import config
import librosa
import librosa.display
import torch
import numpy as np
import random
import soundfile as sf
import datetime
import medleydb as mdb
import yaml
import matplotlib.pyplot as plt
# import warnings

'''
Notion: Only 64/122 songs contain vocals.
MedleyDB/
|-- Audio/
|	|-- AimeeNorwich_Child/
|		|-- AimeeNorwich_Child_MIX.wav
|		|-- ... ...
|	|-- ... ...
|-- Annotations/
|		|-- Instrument_Activations
|			|-- SOURCEID
|				|-- AimeeNorwich_Child_SOURCEID.lab
|				|-- ... ...
|			|-- ... ...
|		|-- ... ...
|-- filelists/ (generated)
|	|-- 0.txt (all songs split)
|	|-- 1.txt (all songs split)
|	|-- 2.txt (all songs split)
|	|-- 10.txt (non-strumental split)
|	|-- 11.txt (non-strumental split)
|	|-- 12.txt (non-strumental split)
|-- 1024_315_80/ (generated)
|	|-- mel
|	|-- label
'''

class MedleyDBProcessor():
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). __init__(self, songsetTag = 'all'): Initializer: for audioFileNameList, vocalActivateTagList, audioFileNameList_vocal generation.
			(2). geneAudio_PS(self, dataType): ongoing ... ...
			(3). raw2mel(self, dataType): Generate melSpectro for all songs including vocal and non-vocal songs.
			(4). txt2label(self): Generating labels for all songs, including vocal and non-vocal songs.
			(5). splitDataSet(self): Split dataset into 3 parts, preparsion for future cross-validation.
			(6). loadFileList(self, filelistFileIndex_Group): Loading audioFileNameList using given filelistFileIndex_Group.
			(7). makeFramePair_Song(self, dataType, audioFileName, step): Make frame-wise paired data for a certain song.
			(8). datasetLoader(self, filelistFileIndex_Group, phase): 
					DataSet Loading for songs in filelistFileIndex_Group. If phase equals 'test', it is also loaded in songlevel.
			(9). getVocalEnergyThreshold(self): Get non-vocal energy threshold for certain audio processing method.
		Description:
			(a). (2), (3), (4), (5) and (6) are data preparing funtions.
			(b). (8) calls (7) to make dataset.
			(c). (9) calcultaes the non-vocal activity threshold.
		Using:
			(a). If it is the 1st time to load data, the following preparing work should be done.
				 Call 'MedleyDBProcessor().raw2mel(self, dataType)' for preparing mel data.
				 Call 'MedleyDBProcessor().txt2label(self)' for preparing labels.
				 Call 'MedleyDBProcessor().splitDataSet(self)' for spliting the dataset.
			(b). Call 'MedleyDBProcessor().datasetLoader(self, filelistFileIndex_Group, phase)' for loading a certain dataset.
			(c). Call 'MedleyDBProcessor().getVocalEnergyThreshold(self)' for calculting the non-vocal activity threshold.
	'''
	def __init__(self, songsetTag = 'all'):
		'''
		Initializer: for audioFileNameList, vocalActivateTagList, audioFileNameList_vocal generation.
		Input:
			songsetTag: 'all'/'vocal', for which subset of songs you would like to use.
		Output:
			----
		'''
		# 6/122
		# to be selected, if ones with vocals are needed.
		self.audioFileNameList = [x for x in os.listdir(os.path.join(config.MELDEY_DB_PATH, 'Audio')) if x[0] != '.']
		self.vocalActivateTagList = [x for x in list(mdb.get_valid_instrument_labels()) if(x == 'choir' or 'male' in x or 'vocal' in x)] # 'male'/'female'
		self.audioFileNameList_vocal = [x for x in self.audioFileNameList if bool(set(self.vocalActivateTagList) & set(mdb.MultiTrack(x).stem_instruments))]
		if(songsetTag == 'all'): self.songset = self.audioFileNameList
		elif(songsetTag == 'vocal'): self.songset = self.audioFileNameList_vocal

	def geneAudio_PS(self, dataType):
		pass

	def raw2mel(self, dataType):
		'''
		Generate melSpectro for all songs including vocal and non-vocal songs.
		Input:  
			dataType: (['raw', 'PS2'])
		Output: 
			----
		'''
		audioFileNameList = [x for x in os.listdir(os.path.join(config.MELDEY_DB_PATH, 'Audio')) if x[0] != '.']
		audioFolderPath = os.path.join(config.MELDEY_DB_PATH, 'Audio')
		targetFolderPath = os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		if not dataType == 'raw':
			# Detail operations should refer to RWCProcessor.py
			pass
		if not os.path.exists(targetFolderPath):
			os.makedirs(targetFolderPath)
		for audioFileName in audioFileNameList:
			# AimeeNorwich_Flying
			# AimeeNorwich_Flying_MIX.wav
			audioFilePath = os.path.join(audioFolderPath, audioFileName, audioFileName + '_MIX.wav')
			targetFilePath = os.path.join(targetFolderPath, audioFileName + '.pkl')
			y, _ = librosa.load(audioFilePath, config.SR)
			melSpectro = librosa.feature.melspectrogram(y, sr=config.SR, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH, 
				window='hann', n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=1.0)
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, targetFilePath)
			# print(audioFileName)

	def txt2label(self):
		'''
		Generating labels for all songs, including vocal and non-vocal songs; 
		Which's more, vocal/non-vocal ratio in song level and dataset level is also logged.
		'''
		statistics = []
		audioFileNameList = [x for x in os.listdir(os.path.join(config.MELDEY_DB_PATH, 'Audio')) if x[0] != '.']
		labelFolderPath = os.path.join(config.MELDEY_DB_PATH, 'Annotations', 'Instrument_Activations', 'SOURCEID')
		labelPKLFolderPath = os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		fo_info = open(os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'raw_data_loading_info.txt'), 'w+')
		if not os.path.exists(labelPKLFolderPath):
			os.makedirs(labelPKLFolderPath)
		for audioFileName in audioFileNameList:
			logMel = torch.load(os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel', audioFileName + '.pkl'))
			label = np.zeros((logMel.shape[-1],))
			with open(os.path.join(labelFolderPath, audioFileName + '_SOURCEID.lab')) as fo:
				for line in fo.readlines():
					line = line.strip('\n').split(',')
					if(line[-1] in self.vocalActivateTagList):
						start = librosa.time_to_frames(float(line[0]), sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
						end = librosa.time_to_frames(float(line[1]), sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
						label[start:end+1] = 1
			torch.save(label, os.path.join(labelPKLFolderPath, audioFileName + '.pkl'))
			print('{}: {}'.format(audioFileName, sum(label)/logMel.shape[-1]))
			fo_info.write('{}: {}\n'.format(audioFileName, sum(label)/logMel.shape[-1]))
			statistics.append(sum(label)/logMel.shape[-1])
		print('all: {}'.format(sum(statistics)/len(statistics)))
		print('non-instrumental: {}'.format(sum(statistics)/len(self.audioFileNameList_vocal)))
		fo_info.write('all: {}\n'.format(sum(statistics)/len(statistics)))
		fo_info.write('non-instrumental: {}\n'.format(sum(statistics)/len(self.audioFileNameList_vocal)))
		# it is ok... ..., with ------ singing/non-singing
		fo_info.close()

	def splitDataSet(self):
		'''
		Split dataset into 3 parts, preparsion for future cross-validation.
		The filelistNames are ['0.txt', '1.txt', '2.txt'] for dataset level split, while ['10.txt', '11.txt', '12.txt'] for vocal subset level split.
		'''
		# for the whole dataset.
		filelistsFolder = os.path.join(config.MELDEY_DB_PATH, 'filelists')
		if not os.path.exists(filelistsFolder): os.makedirs(filelistsFolder)
		for i in range(3):
			fo = open(os.path.join(filelistsFolder, '%s.txt' % (str(i))), 'w+')
			for audioFileName in self.audioFileNameList[i * (len(self.audioFileNameList)//3+1): (i+1) * (len(self.audioFileNameList)//3+1)]:
				fo.write('{}\n'.format(audioFileName))
			fo.close()
		# for vocal subset.
		for i in range(3):
			fo = open(os.path.join(filelistsFolder, '%s.txt' % (str(i+10))), 'w+')
			for audioFileName in self.audioFileNameList_vocal[i * (len(self.audioFileNameList_vocal)//3+1): (i+1) * (len(self.audioFileNameList_vocal)//3+1)]:
				fo.write('{}\n'.format(audioFileName))
			fo.close()

	def loadFileList(self, filelistFileIndex_Group):
		'''
		Loading audioFileNameList using given filelistFileIndex_Group.
		Input:
			filelistFileIndex_Group: e.g. [0, 1, 2].
		Output:
			audioFileNameList: ----.
		'''
		filelistsFolder = os.path.join(config.MELDEY_DB_PATH, 'filelists')
		audioFileNameList = []
		for filelistFileIndex in filelistFileIndex_Group:
			with open(os.path.join(filelistsFolder, '%s.txt' % (str(filelistFileIndex))), 'r') as fo:
				for line in fo.readlines():
					audioFileNameList.append(line.strip('\n'))
		# print(trackIndexList, '\n', len(trackIndexList))
		return audioFileNameList

	def makeFramePair_Song(self, dataType, audioFileName, step):
		'''
		Make frame-wise paired data for a certain song.
		Input:
			dataType: (['raw', 'PS2']), audioFileName: ----, step: ----.
		Output:
			x, y: frame-level paired data per song in tensor version.
		'''
		melFolderPath = os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		labelFolderPath = os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		if(dataType != 'raw'):
			melFolderPath = melFolderPath + '_' + dataType
		melSeries = torch.load(os.path.join(melFolderPath, audioFileName + '.pkl'))
		labelSeries = torch.load(os.path.join(labelFolderPath, audioFileName + '.pkl'))
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

	def datasetLoader(self, filelistFileIndex_Group, phase):
		'''
		DataSet Loading for songs in filelistFileIndex_Group. If phase equals 'test', it is also loaded in songlevel.
		Input:
			filelistFileIndex_Group: filelist names organized in a list. e.g. [0, 1, 2]
			phase: phase determines the step and what form the dataset is returned. 'train'/'valid'/'test'.
		Output: 
			X, Y/ X, Y, data_label_dict: when phase in ['train', 'valid']/ when phase equalls 'test'
		'''
		X, Y, dataTypeList = [], [], []
		dataTypeList.append('raw')
		if(phase.lower() == 'train'):
			step = config.TRAIN_STEP
			if(config.PS_ARG_FLAG == True):
				dataTypeList.append('PS'+str(config.PITCH_SHIFTINF_RANGE))
			else:
				pass
		elif(phase.lower() == 'valid'):
			step = config.VALID_STEP
		else: 
			# 'test'
			step = config.TEST_STEP
		audioFileNameList = self.loadFileList(filelistFileIndex_Group)
		for audioFileName in audioFileNameList:
			for dataType in dataTypeList:
				x, y = self.makeFramePair_Song(dataType, audioFileName, step)
				X.append(x)
				Y.append(y)
		X = torch.cat(X, axis=0)
		Y = torch.cat(Y, axis=0)
		print('-----------------MedleyDB Dataset Loading {}:{}--------------------------'.format(phase, dataType))
		print(X.shape, Y.shape)
		if(phase.lower() == 'train' or phase.lower() == 'valid'):
			return X, Y
		# phase == 'test'
		data_label_dict = {}
		for audioFileName in audioFileNameList:
			x, y = self.makeFramePair_Song('raw', audioFileName, step)
			data_label_dict[audioFileName] = [x, y]
		print('song num: {}'.format(len(data_label_dict)))
		return X, Y, data_label_dict

	def getVocalEnergyThreshold(self):
		'''
		Get non-vocal energy threshold for certain audio processing method.
		Input:
			----
		Output:
			----
		'''
		audioFileNameList = self.loadFileList([10, 11, 12])
		audioFolderPath = os.path.join(config.MELDEY_DB_PATH, 'Audio')
		energyList_song = []
		# fo_info = open(os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'VocalEnergyThreshold.txt'), 'w+')
		for audioFileName in audioFileNameList:
			label = torch.load(os.path.join(config.MELDEY_DB_PATH, config.AUDIO_PROCESSSING_METHOD, 'label', audioFileName + '.pkl'))
			with open(os.path.join(config.MELDEY_DB_PATH, 'Audio', audioFileName, audioFileName + '_METADATA.yaml'), 'r') as stream:
				# for audioName in test_song.keys():
				info_dict = yaml.safe_load(stream)
				for stemIndex in info_dict['stems'].keys():
					if(info_dict['stems'][stemIndex]['instrument'] in self.vocalActivateTagList):
						y, _ = librosa.load(
							os.path.join(config.MELDEY_DB_PATH, 'Audio', audioFileName, audioFileName + '_STEMS', info_dict['stems'][stemIndex]['filename']), 
							config.SR)
						energyList = librosa.feature.rms(y = y, frame_length=config.FRAME_LEN, hop_length=config.HOP_LENGTH)
						break
			#
			# S, phase = librosa.magphase(librosa.stft(y, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH))
			# rms = librosa.feature.rms(S=S, frame_length=config.FRAME_LEN, hop_length=config.HOP_LENGTH)
			# fig, ax = plt.subplots(nrows=2, sharex=True)
			# times = librosa.times_like(rms)
			# ax[0].semilogy(times, rms[0], label='RMS Energy')
			# ax[0].set(xticks=[])
			# ax[0].legend()
			# ax[0].label_outer()
			# librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
			# ax[1].set(title='log Power spectrogram')
			# plt.show()
			#
			# print(len(label), len(energyList[0]))
			# return 1
			salienceEnergyLevel = sum((1 - label) * energyList[0]) / sum((1 - label))
			print('{}: {}, {}'.format(audioFileName, info_dict['stems'][stemIndex]['filename'], salienceEnergyLevel))
			fo_info.write('{}: {}, {}\n'.format(audioFileName, info_dict['stems'][stemIndex]['filename'], salienceEnergyLevel))
			energyList_song.append(salienceEnergyLevel)
			# break
		threshold = sum(energyList_song) / len(energyList_song)
		print('Threshold_statistic: {}'.format(threshold))
		fo_info.write('Threshold_statistic: {}\n'.format(threshold))
		return threshold



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
	MedleyDBPrcs = MedleyDBProcessor()
	if(args.option == 'rawDataProcessing'):
		MedleyDBPrcs.splitDataSet()
		MedleyDBPrcs.raw2mel('raw')
		MedleyDBPrcs.txt2label()
	elif(args.option == 'PS_processing'):
		# ongoing ... ...
		pass
	elif(args.option == 'non-vocal_threshold'):
		threshold = MedleyDBPrcs.getVocalEnergyThreshold()
	threshold = MedleyDBPrcs.getVocalEnergyThreshold()
	# warnings.filterwarnings("ignore")
	
	# X, Y = MedleyDBPrcs.datasetLoader([0, 1], 'train')
	# print(X.shape, Y.shape, len(data_label_dict))
	# MedleyDBPrcs.txt2label()
	# MedleyDBPrcs.splitDataSet()
	# audioFileNameList = MedleyDBPrcs.loadFileList([0, 1])
	# print(len(audioFileNameList))
	# MedleyDBPrcs.raw2mel('raw')

	# mtrack = mdb.MultiTrack('LizNelson_Rainfall')
	# warnings.filterwarnings("ignore")
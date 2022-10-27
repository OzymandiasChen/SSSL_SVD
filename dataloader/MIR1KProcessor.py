
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
# os.environ['OMP_THREAD_LIMIT'] = '1'


'''
Only 20ms resolution tags are provided, it needs to  be converted.
MIR-1K/
|-- vocal-nonvocalLabel/
|	|-- abjones_1_01.vocal
|	|-- abjones_1_02.vocal
|-- Wavfile/
|	|-- abjones_1_01.wav
|	|-- abjones_1_02.wav
|-- UndividedWavfile/
|-- filelists/ (generated)
|-- 1024_315_80/ (generated)
'''

class MIR1KProcessor():
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). __init__(self): Initializer.
			(2). geneAudio_PS(self, dataType): ongoing.
			(3). raw2mel(self, dataType): Generate melSpectro for all songs.
			(4). txt2label(self): Generating labels into the required mode from the originally 20ms resolution tags. 
			(5). splitDataSet(self): Split dataset into 5 parts, preparsion for future cross-validation, which might not be used.
			(6). loadFileList(self, filelistFileIndex_Group): Loading clipNameList using given filelistFileIndex_Group.
			(7). makeFramePair_Song(self, dataType, audioFileName, step): Make frame-wise paired data for a certain song.
			(8). datasetLoader(self, filelistFileIndex_Group, phase): 
					DataSet Loading for songs in filelistFileIndex_Group. If phase equals 'test', it is also loaded in songlevel.
		Description:
			(a). (2), (3), (4), (5) and (6) are data preparing funtions.
			(b). (8) calls (7) to make dataset
		Using:
			(a). If it is the 1st time to load data, the following preparing work should be done.
				 Call 'MIR1KProcessor().raw2mel(self, dataType)' for preparing mel data.
				 Call 'MIR1KProcessor().txt2label(self)' for preparing labels.
				 Call 'MIR1KProcessor().splitDataSet(self)' for spliting the dataset.
			(b). Call 'MIR1KProcessor().datasetLoader(self, filelistFileIndex_Group, phase)' for loading a certain dataset.
	'''

	def __init__(self):
		'''
		Initializer.
		self.audioPath_mix = '', self.audioPath_PS2 = '', self.labelPath_gene = ''
		'''
		pass

	def geneAudio_PS(self, dataType):
		pass

	def raw2mel(self, dataType):
		'''
		Generate melSpectro for all songs.
		Input:  dataType: (['raw', 'PS2'])
		Output: ----
		'''
		audioFolderPath = os.path.join(config.MIR1K_PATH, 'Wavfile') # if make arguement
		targetFolderPath = os.path.join(config.MIR1K_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		if not dataType == 'raw':
			# Detail operations should refer to RWCProcessor.py
			pass
		if not os.path.exists(targetFolderPath):
			os.makedirs(targetFolderPath)
		for audioFileName in os.listdir(audioFolderPath):
			audioFilePath = os.path.join(audioFolderPath, audioFileName)
			targetFilePath = os.path.join(targetFolderPath, audioFileName.split('.')[0] + '.pkl')
			y, _ = librosa.load(audioFilePath, config.SR)
			melSpectro = librosa.feature.melspectrogram(y, sr=config.SR, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH, 
				window='hann', n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=1.0)
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, targetFilePath)
			print(audioFileName)
			pass

	def txt2label(self):
		'''
		Generating labels into the required mode from the originally 20ms resolution tags, thus the transformed tag are not very clear. 
		# 20ms, 0.02ss --> in seconds --> in frames.
		'''
		statistics = []
		labelFolderPath = os.path.join(config.MIR1K_PATH, 'vocal-nonvocalLabel')
		labelPKLFolderPath = os.path.join(config.MIR1K_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		fo_info = open(os.path.join(config.MIR1K_PATH, config.AUDIO_PROCESSSING_METHOD, 'raw_data_loading_info.txt'), 'w+')
		if not os.path.exists(labelPKLFolderPath):
			os.makedirs(labelPKLFolderPath)
		for labelTXTName in os.listdir(labelFolderPath):
			logMel = torch.load(os.path.join(config.MIR1K_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel', labelTXTName.split('.')[0] + '.pkl'))
			label = np.zeros((logMel.shape[-1],))
			frameTagList_20ms = []
			with open(os.path.join(labelFolderPath, labelTXTName)) as fo:
				for line in fo.readlines():
					frameTagList_20ms.append(int(line.strip('\n')))
			while frameTagList_20ms[-1] == 0: frameTagList_20ms.pop()	# remove the final zeros
			endFrame_20ms, stopFlag = 0, False
			while stopFlag == False:
				try:
					startFrame_20ms = frameTagList_20ms.index(1, endFrame_20ms)
					endFrame_20ms = frameTagList_20ms.index(0, startFrame_20ms)
				except:
					endFrame_20ms = len(frameTagList_20ms)
					stopFlag = True
				start = librosa.time_to_frames((startFrame_20ms+1)*20/1000, sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
				end = librosa.time_to_frames((endFrame_20ms+0)*20/1000, sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
				label[start: end+1] = 1
			torch.save(label, os.path.join(labelPKLFolderPath, labelTXTName.split('.')[0] + '.pkl'))
			print('{}: {}/{}, {}'.format(labelTXTName, logMel.shape[-1], end, sum(label)/logMel.shape[-1]))
			fo_info.write('{}: {}/{}, {}\n'.format(labelTXTName, logMel.shape[-1], end, sum(label)/logMel.shape[-1]))
			statistics.append(sum(label)/logMel.shape[-1])
		# print(sum(statistics)/len(statistics))
		fo_info.write('{}\n'.format(sum(statistics)/len(statistics)))
		# it is ok... ..., with 0.76 singing/non-singing
		fo_info.close()

	def splitDataSet(self):
		'''
		Split dataset into 5 parts, preparsion for future cross-validation.
		The filelistNames are ['0.txt', '1.txt', '2.txt', '3.txt', '4.txt']
		'''
		# song_level split
		songNameList = [x.split('.')[0] for x in os.listdir(os.path.join(config.MIR1K_PATH, 'UndividedWavfile'))]
		random.shuffle(songNameList)
		# print(len(songNameList))
		filelistsFolder = os.path.join(config.MIR1K_PATH, 'filelists')
		if not os.path.exists(filelistsFolder):	os.makedirs(filelistsFolder)
		for i in range(5):
			fo = open(os.path.join(filelistsFolder, '%s.txt' % (str(i))), 'w+')
			# python do not deal with index overflow.
			for songName in songNameList[i * (len(songNameList)//5+1): (i+1) * (len(songNameList)//5+1)]:
				# There exists the problem of 2 strings with the same prefix, intresting.
				clipNameList = [x for x in os.listdir(os.path.join(config.MIR1K_PATH, 'Wavfile')) \
					if (songName in x and songName.split('_')[1] == x.split('_')[1])]
				for clipName in clipNameList:
					fo.write('{}\n'.format(clipName))
			fo.close()

	def loadFileList(self, filelistFileIndex_Group):
		'''
		Loading clipNameList using given filelistFileIndex_Group.
		Input:
			filelistFileIndex_Group: filelist names organized in a list. e.g. [0, 1, 4]
		Output:
			clipNameList: ----
		'''
		clipNameList = []
		for filelistFileIndex in filelistFileIndex_Group:
			with open(os.path.join(config.MIR1K_PATH, 'filelists', str(filelistFileIndex) + '.txt'), 'r') as fo:
				for line in fo.readlines():
					clipNameList.append(line.strip('\n'))
		return clipNameList

	def makeFramePair_Song(self, dataType, audioFileName, step):
		'''
		Make frame-wise paired data for a certain song.
		Input:
			dataType: (['raw', 'PS2']), audioFileName: ----, step: ----.
		Output:
			x, y: frame-level paired data per song in tensor version.
		'''
		melFolderPath = os.path.join(config.MIR1K_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		labelFolderPath = os.path.join(config.MIR1K_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		if(dataType != 'raw'):
			melFolderPath = melFolderPath + '_' + dataType
		melSeries = torch.load(os.path.join(melFolderPath, audioFileName.split('.')[0] + '.pkl'))
		labelSeries = torch.load(os.path.join(labelFolderPath, audioFileName.split('.')[0] + '.pkl'))
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
			filelistFileIndex_Group: filelist names organized in a list. e.g. [0, 1, 4]
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
		clipNameList = self.loadFileList(filelistFileIndex_Group)
		for clipName in clipNameList:
			for dataType in dataTypeList:
				x, y = self.makeFramePair_Song(dataType, clipName, step)
				X.append(x)
				Y.append(y)
		X = torch.cat(X, axis=0)
		Y = torch.cat(Y, axis=0)
		print('-----------------MIR1K Dataset Loading {}:{}--------------------------'.format(phase, dataType))
		print(X.shape, Y.shape)
		if(phase.lower() == 'train' or phase.lower() == 'valid'):
			return X, Y
		# phase == 'test'
		data_label_dict = {}
		for clipName in clipNameList:
			x, y = self.makeFramePair_Song('raw', clipName, step)
			data_label_dict[clipName] = [x, y]
		print('song num: {}'.format(len(data_label_dict)))
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
	MIR1KPrcs = MIR1KProcessor()
	if(args.option == 'rawDataProcessing'):
		MIR1KPrcs.splitDataSet()
		MIR1KPrcs.raw2mel('raw')
		MIR1KPrcs.txt2label()
	elif(args.option == 'PS_processing'):
		# ongoing ... ...
		pass
	# _, _, _, = MIR1KPrcs.datasetLoader([0, 1, 2, 3, 4], 'test')
	# clipNameList = MIR1KPrcs.loadFileList([0, 1, 2])
	# print(len(clipNameList))
	# MIR1KPrcs.splitDataSet()
	# MIR1KPrcs.txt2label()
	# MIR1KPrcs.raw2mel('raw')
	# a = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]
	# while a[-1] == 0: a.pop()
	# print(len(a))
	# # '''
	# # 3, 7;
	# # 11, 13;
	# # 16, 16;
	# # '''
	# end = 0 # the first.
	# while 1:
	# 	try:
	# 		start = a.index(1, end)
	# 		end = a.index(0, start)
	# 		print('{}, {}'.format(start, end))
	# 	except:
	# 		end = len(a)
	# 		print('{}, {}'.format(start, end))
	# 		break
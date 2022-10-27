
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
jamendo/
|-- audio/
|-- filelists/
|	|-- train
|	|-- test
|	|-- valid
|	|-- recreate.sh
|-- labels/
|
|-ADDED-|
|-- 1024_315_80/
|	|-- label/
|	|-- mel/
|	|-- mel_PS2
|--audio_PS_2
'''

class printHelper():
	def __init__(self):
		self.count = 0
	def printDataMelLabelLog(self, audioFileName, audioFrameNum, labelFrameNum, fo):
		self.count += 1
		print('{}\n'.format(self.count))
		fo.write('{}:\n'.format(audioFileName))
		print('{}:\n'.format(audioFileName))
		fo.write(('SongFrame: {},	LabelFrame: {},	dif: {}\n'.format(
			audioFrameNum, labelFrameNum, audioFrameNum - labelFrameNum)))
		print(('SongFrame: {},	LabelFrame: {},	dif: {}\n'.format(
			audioFrameNum, labelFrameNum, audioFrameNum - labelFrameNum)))

class JamendoProcessor():
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). loadSaveMel_Song(self, audioSubFolder, audioFileName, mel_category): Make song level MelSpectro and save it in pkl mode.
			(2). loadSaveLabel_Song(self, audioFileName, labelNum_Song): Make song level label in corresponding format and save it in pkl.
			(3). raw2MelLabel_all(self): For each song in '/jamendo/label', make song level melspectro and label.
			(4). geneAudio_PS(self): Generate pitch shifted wave file for Data Arguenment.
			(5). raw2mel_PS(self): Generate melspectro for pitch shifted wave.
			(6). loadDataSetFileName(self, train_test_valid): Getting dataset file name lsit for Train/Test/Valid.
			(7). makeFramePair_Song(self, audioFileName, step, mel_category): Make frame-level paired data by using beforehand saved song-level data.
			(8). makeFramePair_Dataset(self, audioNameList, step, mel_category): Make frame level paired data for a certain dataset.
				 (a real dataset 'train'/'test'/'valid' or a piece of song).
			(9). datasetLoader(self, datasetName, PS = True): Load dataset for 'train'/'test'/'valid'/'test_song'
				 using beforehand save song-level mel and label serires.
		Description:
			(a). (3) calls (1) and (2) to call and save mel and labels for song.
			(b). (4) and (5) make pitch shifted wave and mel.
			(c). (9) calls (8) and (7) to load a certain dataset.
		Using:
			(a) 'Call JamendoProcessor.raw2MelLabel_all()' for song-level mel and label saving .
				'Call JamendoProcessor.geneAudio_PS()' for pitch shifted wave file generation.
				'Call JamendoProcessor.raw2mel_PS()' for pitch shifted  melspectro generation.
			(b) 'Call JamendoProcessor.datasetLoader(datasetName)' for dataset loading
	'''

	def __init__(self):
		pass
	
	def loadSaveMel_Song(self, audioSubFolder, audioFileName, mel_category):
		'''
		Make song level MelSpectro and save it in pkl mode.
		Input:
			audioSubFolder: 'audio', 'audio_PS'+str(config.PITCH_SHIFTINF_RANGE)
			audoFileName: such as '01 - 01 Les Jardins Japonais.ogg'
			mel_category: 'mel', 'mel_PS'+str(config.PITCH_SHIFTINF_RANGE)
		Output:
			how many frames in a song
		'''
		if(audioFileName.split('.')[1]=='sh'):
			return 0
		audioFilePath = os.path.join(config.JAMENDO_PATH, audioSubFolder, audioFileName)
		y, _ = librosa.load(audioFilePath, config.SR)
		melSpectro = librosa.feature.melspectrogram(y, sr=config.SR, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH, 
			window='hann', n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=1.0)
		# window='hann', power=2.0?， center=True,
		# what does power mean, why power is set to be 1?  
		logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
		#print(logMelSpectro.shape)
		#print(logMelSpectro.dtype) # float 32
		# DOES MEL path exists??
		melFolderPath = os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, mel_category)
		torch.save(logMelSpectro, os.path.join(melFolderPath, audioFileName.split('.')[0]+'.pkl'))
		# retrun how many frames in a song
		return logMelSpectro.shape[1]

	def loadSaveLabel_Song(self, audioFileName, labelNum_Song):
		'''
		Make song level label in corresponding format and save it in pkl.
		Input:
			audioFileName: audio file name, such as '01 - 01 Les Jardins Japonais.ogg'
		Output:
			return last time stap
		'''
		if(audioFileName.split('.')[1]=='sh'):
			return 0
		# audioFilePath = os.path.join(config.JAMENDO_PATH, 'audio', audioFileName)
		labelFilePath = os.path.join(config.JAMENDO_PATH, 'labels', audioFileName.split('.')[0]+'.lab')
		# JAMENDO_Path/labels/01 - 01 Les Jardins Japonais.lab
		label = np.zeros((labelNum_Song,))
		with open(labelFilePath, 'r') as fo:
			for line in fo.readlines():
				line = line.strip('\n').split(' ')
				start = librosa.time_to_frames(float(line[0]), sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
				end = librosa.time_to_frames(float(line[1]), sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
				# Optional: 
				#	length of the FFT window. If given, time conversion will include an offset of - n_fft // 2 to counteract windowing effects in STFT.
				is_vocal = 1 if line[2] == 'sing' or line[2] == '1' else 0
				# sing: 1/ nosing: 0
				label[start:end+1] = int(is_vocal) 
				# python doesn't raise error when it goes out of the bound
		# It would be better if it could be wr
		labelFolderPath = os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		torch.save(label, os.path.join(labelFolderPath, audioFileName.split('.')[0]+'.pkl'))
		#print(label.shape)
		return end # return last time stap

	def raw2MelLabel_all(self):
		'''
		For each song in '/jamendo/label', make song level melspectro and label.
		Corresponding info (how many frames mannually labeled) will be saved in 'raw_data_loading_info.txt'.
		'''
		melFolderPath = os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		if not os.path.exists(melFolderPath):
			os.makedirs(melFolderPath)
		# make song level label series folder path for loadSaveLabel_Song(self,random.randint(12, 20)  audioFileName, labelNum_Song)
		labelFolderPath = os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		if not os.path.exists(labelFolderPath):
			os.makedirs(labelFolderPath)
		fo = open(os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, 'raw_data_loading_info.txt'), 'w+')
		ph = printHelper()
		for audioFileName in os.listdir(os.path.join(config.JAMENDO_PATH, 'audio')):
			songFrameNum = self.loadSaveMel_Song('audio', audioFileName, 'mel')
			lastTimeStap = self.loadSaveLabel_Song(audioFileName, songFrameNum)
			ph.printDataMelLabelLog(audioFileName, songFrameNum, lastTimeStap+1, fo)
		fo.close()

	def geneAudio_PS(self):
		'''
		Generate pitch shifted wave file for. // Data Arguenment.
		'''
		if not os.path.exists(os.path.join(config.JAMENDO_PATH, 'audio_PS_'+str(config.PITCH_SHIFTINF_RANGE))):
			os.makedirs(os.path.join(config.JAMENDO_PATH, 'audio_PS_'+str(config.PITCH_SHIFTINF_RANGE)))
		for audioFileName in self.loadDataSetFileName('train'):
			if(audioFileName.split('.')[1]=='sh'):
				continue
			y_source, _ = librosa.load(os.path.join(config.JAMENDO_PATH, 'audio', audioFileName), config.SR)
			y_PS = librosa.effects.pitch_shift(
				y_source, config.SR, n_steps = random.randint(-1 * config.PITCH_SHIFTINF_RANGE, config.PITCH_SHIFTINF_RANGE), bins_per_octave=12)
			sf.write(
				os.path.join(config.JAMENDO_PATH, 'audio_PS_'+str(config.PITCH_SHIFTINF_RANGE), audioFileName.split('.')[0]+'.wav'), 
				y_PS, config.SR, subtype='PCM_24')

	def raw2mel_PS(self):
		'''
		Generate melspectro for pitch shifted wave.
		'''
		if not os.path.exists(os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel_PS'+str(config.PITCH_SHIFTINF_RANGE))):
			os.makedirs(os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel_PS'+str(config.PITCH_SHIFTINF_RANGE)))
		for audioFileName in self.loadDataSetFileName('train'):
			if(audioFileName.split('.')[1]=='sh'):
				continue
			self.loadSaveMel_Song(
				'audio_PS_'+str(config.PITCH_SHIFTINF_RANGE), audioFileName.split('.')[0]+'.wav', 'mel_PS'+str(config.PITCH_SHIFTINF_RANGE))
		
	def loadDataSetFileName(self, train_test_valid):
		'''
		Getting dataset file name lsit for Train/Test/Valid.
		Input:
			train_test_valid: 'Train', 'Test' or 'Valid'
		Return:
			nameList: dataset name list
		'''
		nameList = []
		nameListFilePath  = os.path.join(config.JAMENDO_PATH, 'filelists', train_test_valid.lower())
		fo = open(nameListFilePath, "r")
		for line in fo.readlines():
			line  = line.strip('\n')
			nameList.append(line)
		fo.close()
		return nameList

	def makeFramePair_Song(self, audioFileName, step, mel_category):
		'''
		Make frame-level paired data by using beforehand saved song-level data.
		Input:
			audioFileName: audio name for song
			step: how many frames per tag
			mel_category: 'mel', 'mel_PS'+str(config.PITCH_SHIFTINF_RANGE),
		Output:
			x, y: frame-level paired data per song in tensor version.
		'''
		# melSeries and labelSeries are numpy arraies.	
		melSeries = torch.load(os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, mel_category, audioFileName.split('.')[0]+'.pkl'))
		labelSeries = torch.load(os.path.join(config.JAMENDO_PATH, config.AUDIO_PROCESSSING_METHOD, 'label', audioFileName.split('.')[0]+'.pkl'))
		x=[]
		y=[]
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
		return x, y

	def makeFramePair_Dataset(self, audioNameList, step, mel_category):
		'''
		Make frame level paired data for a certain dataset (a real dataset 'train'/'test'/'valid' or a piece of song).
		Input:
			audioNameList: a song name list.
			step: how many frames per tag
		Output:
			X, Y: frame-level paired data for each dataset in tensor version.
		'''
		X = []
		Y = []
		for audioName in audioNameList:
			x, y = self.makeFramePair_Song(audioName, step, mel_category)
			X.append(x)
			Y.append(y)
		X = torch.cat(X, axis=0)
		Y = torch.cat(Y, axis=0)
		return X, Y

	def datasetLoader(self, datasetName, PS = True):
		'''
		Load dataset for train/test/valid using beforehand save song-level mel and label serires.
		Input:
			datasetName: 'train'/'test'/'valid'/'test_song'
			useful only when datasetName == 'train':
				PS: data arguement for pitch shifting
		Output:
			X, Y: torch tensor for 'train'/'test'/'valid'
			name_data_label_dict: dict for 'test_song' in the format of {audioName: [X, Y]}
		'''
		print('-----------------Jamendo Dataset Loading: {}--------------------------'.format(datasetName))
		if(datasetName.lower() == 'train'):
			step = config.TRAIN_STEP
		elif(datasetName.lower() == 'valid'):
			step = config.VALID_STEP
		else:
			step = config.TEST_STEP
		if(datasetName.lower() == 'test_song'):
			name_data_label_dict = {}
			testFileNameList = self.loadDataSetFileName('test')
			for i in range(len(testFileNameList)):
				X_test_song, Y_test_song = self.makeFramePair_Song(testFileNameList[i], step, 'mel')
				name_data_label_dict[testFileNameList[i]] = [X_test_song, Y_test_song]
			print(len(name_data_label_dict))
			return name_data_label_dict
		audioNameList = self.loadDataSetFileName(datasetName.lower())
		X, Y = self.makeFramePair_Dataset(audioNameList, step, 'mel')
		if(datasetName.lower() == 'train'):
			if(PS == True):
				X_PS, Y_PS = self.makeFramePair_Dataset(audioNameList, step, 'mel_PS'+str(config.PITCH_SHIFTINF_RANGE))
				X = torch.cat((X, X_PS), 0)
				Y = torch.cat((Y, Y_PS), 0)
		print(X.shape, Y.shape)
		return X, Y

def parse_args():
	'''
	'''
	description = 'options'
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument('--option',help = '')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# starttime = datetime.datetime.now()
	args = parse_args()
	jamendoLoader = JamendoProcessor()
	if(args.option == 'rawDataProcessing'):
		jamendoLoader.raw2MelLabel_all()
	elif(args.option == 'PS_processing'):
		jamendoLoader.geneAudio_PS()
		jamendoLoader.raw2mel_PS()
	# name_data_label_dict = jamendoLoader.datasetLoader('test_song', PS = False)
	# for key in name_data_label_dict.keys():
	# 	print('{}: {}, {}'.format(key, name_data_label_dict[key][0].shape, name_data_label_dict[key][1].shape))
	# 
	# X_train, Y_train = jamendoLoader.datasetLoader('train', PS = False)
	# print(X_train.shape, Y_train.shape)
	# jamendoLoader.raw2MelLabel_all()
	# jamendoLoader.geneAudio_PS()
	# jamendoLoader.raw2mel_PS()
	# X_train, Y_train = jamendoLoader.datasetLoader('train', PS = True)
	# X_train, Y_train = jamendoLoader.datasetLoader('train', PS = False)
	# X_test, Y_test = jamendoLoader.datasetLoader('test')
	# X_valid, Y_valid = jamendoLoader.datasetLoader('valid')
	# dic = jamendoLoader.datasetLoader('test_song')
	# endtime = datetime.datetime.now()
	# print (endtime - starttime)

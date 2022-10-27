
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
from ffmpy import FFmpeg
import subprocess
import shutil
import musdb
from pydub import AudioSegment
from pydub.playback import play

# os.environ['OMP_THREAD_LIMIT'] = '1'

'''
# 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6'
musdb18/
|-- RWC-MDB-P-2001/
|	|-- 1024_315_80/
|		|-- label/
|		|-- mel/
|		|-- mel_plus6/
|		|-- mel_minus6/
|		|-- mel_vocal_plus6/
|		|-- mel_vocal_minus6/
|	|-- train
|	|-- test
|-- README.md
|-- filelists/
'''

class Musdb18Processor():
	'''
		The class is structed with following paramaters and functions.
		Note that, in this section, Musdb18 is only used for the pressure test.
		Functions:
			(1). __init__(self): Initializer. Setting for trainSet/testSet/trainSet_womdb/testSet_womdb/mus_train/
			(2). stereo2mono(self, track, target): A helper function for generating wav using tracks obtrained by musdb library.
			(3). geneAudio(self, dataType, phase = 'test'): Generating audio for 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6' in the wave format.
			(4). geneMel(self, dataType, phase = 'test'): Generating melSpectro for corresponding audios.
			(5). geneLabel(self, phase = 'test'): Generating label using the threshold definded in config.py.
			(6). makeFramePair_Song(self, dataType, trackName, step): Make frame-wise paired data for a certain song (given the trackName).
			(7). datasetLoader(self, phase, dataType = None): Dataset Loading.
		Description:
			(a). (3), (4), (5) generates required audio, melspectrogram and labels.
			(b). (7) calls (6) for loading a certain dataset.
		Using:
			dataType: 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6'
			(a). If it is the 1st time to load data, the following preparing work should be done.
				 Call 'Musdb18Processor().geneAudio(self, dataType, phase = 'test')' for required audio.
				 Call 'Musdb18Processor().ggeneMel(self, dataType, phase = 'test')' for required melspectrogram.
				 Call 'Musdb18Processor().geneLabel(self, phase = 'test')' for required labels.
			(b). Call 'Musdb18Processor().datasetLoader(self, phase, dataType = None)' for loading a certain dataset.		 
	'''
	# For each file, the mixture correspond to the sum of all the signals.	 line from the official website.

	def __init__(self):
		'''
		Initializer. Tracks from medleydb: 46/150 for train, 0/50 for test
		Setting for trainSet/testSet/trainSet_womdb/testSet_womdb/mus_train/
		Input:
			----
		Output:
			----
		'''
		self.trainSet = os.listdir(os.path.join(config.MUSDB18_PATH, 'train'))
		# [x.split('.')[0: -4] for x in os.listdir(os.path.join(config.MUSDB18_PATH, 'train'))]
		self.testSet = os.listdir(os.path.join(config.MUSDB18_PATH, 'test'))
		# [x.split('.')[0: -4] for x in os.listdir(os.path.join(config.MUSDB18_PATH, 'test'))]
		MedleyDB_shared = []
		with open(os.path.join(config.MUSDB18_PATH, 'README.md'), 'r') as fo:
			for i, line in enumerate(fo): # for line in fo.readlines():
				if(i < 25):
					continue
				else:
					if(line.split(',')[2] == 'MedleyDB'):
						MedleyDB_shared.append(line.split(',')[0][2:])
		# print(len(MedleyDB_shared))
		self.trainSet_womdb = [x for x in self.trainSet if not (x.split('.')[0] in MedleyDB_shared)]
		self.testSet_womdb = [x for x in self.testSet if not (x.split('.')[0] in MedleyDB_shared)]
		# print('train: {}/{}, test: {}/{}'.format(len(self.trainSet_womdb), len(self.trainSet), len(self.testSet_womdb), len(self.testSet)))

		# The above might be useful later
		# self.mus = musdb.DB(root = config.MUSDB18_PATH)	# for 150 songs
		self.mus_train = musdb.DB(root = config.MUSDB18_PATH, subsets="train") # for 100 songs in train set
		self.mus_test = musdb.DB(root = config.MUSDB18_PATH, subsets="test") # for 100 songs in test
		# self.mus_train_womdb = pass

	def stereo2mono(self, track, target):
		'''
		A helper function for generating wav using tracks obtrained by musdb library.
		Input:
			track: ---- obtrained by musdb library.
			target: 'linear_mixture'/'vocals'/'accompaniment'
		Output:
			y: (nb.samples, ) # mono, 22050, ... ...
		'''
		y = track.targets[target].audio.T # shape: (nb_samples, 2)
		sr = track.rate
		y = librosa.to_mono(y)
		y = librosa.resample(y, orig_sr = sr, target_sr = config.SR, fix = True, scale = False) # downsample
		return y

	def geneAudio(self, dataType, phase = 'test'):
		'''
		Generating audio for 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6' in the wave format.
		Input:
			dataType: 'raw'/'mix_plus_minus'/'vocalSNR'
			phase: 'train'/'test'
		Output:
			----
		'''
		if(phase == 'train'):
			mus = self.mus_train
			rawFolderPath = os.path.join(config.MUSDB18_PATH, 'train')
		else:
			mus = self.mus_test
			rawFolderPath = os.path.join(config.MUSDB18_PATH, 'test')
		if(dataType == 'raw'):
			mixtureFolderPath, vocalFolderPath, accompanyFolderPath = rawFolderPath + '_mix', rawFolderPath + '_vocal', rawFolderPath + '_accom'
			if not os.path.exists(mixtureFolderPath): os.makedirs(mixtureFolderPath)
			if not os.path.exists(vocalFolderPath): os.makedirs(vocalFolderPath)
			if not os.path.exists(accompanyFolderPath): os.makedirs(accompanyFolderPath)
			for track in mus:
				print(track.name)
				y_mix = self.stereo2mono(track, 'linear_mixture')
				sf.write(os.path.join(mixtureFolderPath, track.name + '.wav'), y_mix, config.SR, subtype='PCM_24')
				y_vocal = self.stereo2mono(track, 'vocals')
				sf.write(os.path.join(vocalFolderPath, track.name + '.wav'), y_vocal, config.SR, subtype='PCM_24')
				y_accom = self.stereo2mono(track, 'accompaniment')
				sf.write(os.path.join(accompanyFolderPath, track.name + '.wav'), y_accom, config.SR, subtype='PCM_24')
		elif(dataType == 'mix_plus_minus'): # 'mix_plus_6db'/'mix_minus_6db'
			mixtureFolderPath, mixtureFolderPath_plus6, mixtureFolderPath_minus6 = (
				rawFolderPath + '_mix', rawFolderPath + '_plus6', rawFolderPath + '_minus6')
			print('{}, {}, {}'.format(mixtureFolderPath, mixtureFolderPath_plus6, mixtureFolderPath_minus6))
			if not os.path.exists(mixtureFolderPath_plus6): os.makedirs(mixtureFolderPath_plus6)
			if not os.path.exists(mixtureFolderPath_minus6): os.makedirs(mixtureFolderPath_minus6)
			for track in mus:
				print(track.name)
				mix_org = AudioSegment.from_wav(os.path.join(mixtureFolderPath, track.name + '.wav'))
				mix_plus6 = mix_org + 6
				mix_minus6 = mix_org - 6
				mix_plus6.export(os.path.join(mixtureFolderPath_plus6, track.name + '.wav'), format='wav')
				mix_minus6.export(os.path.join(mixtureFolderPath_minus6, track.name + '.wav'), format='wav')
		elif(dataType == 'vocalSNR'): # 'vocal + 6db'/'vocal - 6db'
			vocalFolderPath, accompanyFolderPath, vocalPlus6FolderPath, vocalMinus6FolderPath = (
				rawFolderPath + '_vocal', rawFolderPath + '_accom', rawFolderPath + '_vocal_plus6', rawFolderPath + '_vocal_minus6')
			if not os.path.exists(vocalPlus6FolderPath): os.makedirs(vocalPlus6FolderPath)
			if not os.path.exists(vocalMinus6FolderPath): os.makedirs(vocalMinus6FolderPath)			
			for track in mus:
				print(track.name)
				vocal = AudioSegment.from_wav(os.path.join(vocalFolderPath, track.name + '.wav'))
				accompaniment = AudioSegment.from_wav(os.path.join(accompanyFolderPath, track.name + '.wav'))
				vocal_plus6 = accompaniment.overlay(vocal + 6)
				vocal_minus6 = accompaniment.overlay(vocal - 6)
				vocal_plus6.export(os.path.join(vocalPlus6FolderPath, track.name + '.wav'), format = 'wav')
				vocal_minus6.export(os.path.join(vocalMinus6FolderPath, track.name + '.wav'), format = 'wav')
		else:
			pass

	def geneMel(self, dataType, phase = 'test'):
		'''
		Generating melSpectro for corresponding audios.
		Input:
			dataType: 'raw'/'plus_6'/'minus_6'/'vocal_plus6'/'vocal_minus6'
			phase: 'train'/'test'
		Output:
			----
		'''
		targetFolderPath = os.path.join(config.MUSDB18_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		if(phase == 'train'):
			mus = self.mus_train
			audioFolderPath_fake = os.path.join(config.MUSDB18_PATH, 'train')
		else:
			mus = self.mus_test
			audioFolderPath_fake = os.path.join(config.MUSDB18_PATH, 'test')
		if(dataType == 'raw'): dataType = 'mix'
		audioFolderPath = audioFolderPath_fake + '_' + dataType
		if(dataType != 'mix'): targetFolderPath = targetFolderPath + '_' + dataType
		if not os.path.exists(targetFolderPath): os.makedirs(targetFolderPath)
		for track in mus:
			print(track.name)
			y, _ = librosa.load(os.path.join(audioFolderPath, track.name + '.wav'), config.SR)
			melSpectro = librosa.feature.melspectrogram(y, sr=config.SR, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH, 
				window='hann', n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=1.0)
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, os.path.join(targetFolderPath, track.name + '.pkl'))
			# def add_noise2(x, snr):
			#     # snr：生成的语音信噪比
			#     P_signal = np.sum(abs(x) ** 2) / len(x)  # 信号功率
			#     P_noise = P_signal / 10 ** (snr / 10.0)  # 噪声功率
			#     return x + np.random.randn(len(x)) * np.sqrt(P_noise)

	def geneLabel(self, phase = 'test'):
		'''
		Generating label using the threshold definded in config.py; 
		Currently setted as  0.0001, since a majority of calculated salience threshhold is in (1e-4, 1e-6)
		Input:
			phase: 'train'/'test'
		Output:
			----
		'''
		fo = open(os.path.join(config.MUSDB18_PATH, config.AUDIO_PROCESSSING_METHOD, 'label_info'), 'w+')
		if(phase == 'train'): 
			mus = self.mus_train
		else: 
			mus = self.mus_test
		targetFolderPath = os.path.join(config.MUSDB18_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		if not os.path.exists(targetFolderPath):
			os.makedirs(targetFolderPath)
		for track in mus:
			print(track.name)
			y = track.targets['vocals'].audio.T # shape: (nb_samples, 2)
			sr = track.rate
			y = librosa.to_mono(y)
			y = librosa.resample(y, orig_sr = sr, target_sr = config.SR, fix = True, scale = False) # downsample
			energyList = librosa.feature.rms(y = y, frame_length=config.FRAME_LEN, hop_length=config.HOP_LENGTH)[0]
			label = [int(x > config.ENERGY_THRESHOLD) for x in energyList]
			torch.save(label, os.path.join(targetFolderPath, track.name + '.pkl'))
			print('{}: {}/{} {}'.format(track.name, sum(label), len(label), sum(label)/len(label)))
			print('{}/{}'.format(len(label), 
				torch.load(os.path.join(config.MUSDB18_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel', track.name + '.pkl')).shape[1]))
			fo.write('{}: {}/{} {}\n'.format(track.name, sum(label), len(label), sum(label)/len(label)))
		fo.close()

	def makeFramePair_Song(self, dataType, trackName, step):
		'''
		Make frame-wise paired data for a certain song (given the trackName).
		Input:
			dataType: 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6'
			trackName: ----
			step: ----
		Output:
			x, y: frame-level paired data per song in tensor version.
		'''
		melFolderPath = os.path.join(config.MUSDB18_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		labelFolderPath = os.path.join(config.MUSDB18_PATH, config.AUDIO_PROCESSSING_METHOD, 'label')
		if(dataType != 'raw'): melFolderPath = melFolderPath + '_' + dataType
		melSeries = torch.load(os.path.join(melFolderPath, trackName + '.pkl'))
		labelSeries = torch.load(os.path.join(labelFolderPath, trackName + '.pkl'))
		x = []
		y = []
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

	def datasetLoader(self, phase, dataType = None):
		'''
		Dataset Loading.
		Input:
			phase: 'train'/'test'/'pressure_test'
		Output:
			X, Y: dataset.
		'''
		X, Y = [], []
		print('-----------------Musdb18 Dataset Loading {}/{}--------------------------'.format(phase, dataType))
		if(phase == 'pressure_test'):
			mus = self.mus_test
			step = config.TEST_STEP
			step = 14
			for track in mus:
				x, y = self.makeFramePair_Song(dataType, track.name, step)
				X.append(x)
				Y.append(y)
			X = torch.cat(X, axis=0)
			Y = torch.cat(Y, axis=0)
			print(Y.shape)		
			# data_label_dict = {}
			# for track in mus:
			# 	x, y = self.makeFramePair_Song(dataType, track.name, step)
			# 	data_label_dict['{}'.format(track.name)] = [x, y]
			# return X, Y, data_label_dict
			return X, Y

def parse_args():
	'''
	'''
	description = 'options'
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument('--option',help = 'audio_gene/')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# dataType: 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6'
	args = parse_args()
	Musdb18Prcs = Musdb18Processor()
	if(args.option == 'audio_gene'):
		# 'raw'/'mix_plus_minus'/'vocalSNR'
		Musdb18Prcs.geneAudio(dataType = 'raw', phase = 'test')
		Musdb18Prcs.geneAudio(dataType = 'mix_plus_minus', phase = 'test')
		Musdb18Prcs.geneAudio(dataType = 'mix_plus_minus', phase = 'test')
	elif(args.option == 'mel_gene'):
		Musdb18Prcs.geneMel('raw')
		Musdb18Prcs.geneMel('plus6')
		Musdb18Prcs.geneMel('minus6')
		Musdb18Prcs.geneMel('vocal_plus6')
		Musdb18Prcs.geneMel('vocal_minus6')
	elif(args.option == 'label_gene'):
		Musdb18Prcs.geneLabel()
	else:
		pass
	# Musdb18Prcs.geneMel('plus6')
	# print('--------------------------------')
	# Musdb18Prcs.geneMel('minus6')
	# print('--------------------------------')
	# Musdb18Prcs.geneMel('vocal_plus6')
	# print('--------------------------------')
	# Musdb18Prcs.geneMel('vocal_minus6')
	# Musdb18Prcs.geneAudio(dataType = 'vocalSNR', phase = 'test')
	# X, Y, data_label_dict = Musdb18Prcs.datasetLoader('test')
	# print(X.shape, Y.shape)
	# Musdb18Prcs.geneMel('raw')
	# Musdb18Prcs.geneLabel()
	# Musdb18Prcs.raw2mel('raw')
	# Musdb18Prcs.geneAudio_Raw()
	# if(args.option == 'rawDataProcessing'):
	# 	pass
	# elif(args.option == 'PS_processing'):
	# 	pass

import os
import sys
import argparse
sys.path.append(sys.path[0][:-10])
import config
import librosa
import torch
import numpy as np
import random
# import warnings

'''
sawtooth_200/
|-- songs/
|	|-- .wav
'''

class VibratoProcessor():
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). __init__(self): Initializer.
			(2). mel_gene(self): Generating melspectro for all songs.
			(3). makeFramePair_Song(self, audioFileName, step = 3): Make paired data for a certain song.
			(4). datasetLoader(self): Dataset loader in a dict version, for fast dataset search.
		Description:
			(a). (2) helps to generate melspectro.
			(b). (4) calls (3) to load the dataset in dict version.
		Using:
			(a). If it is the 1st time to load data, the following preparing work should be done.
				 Call 'MIR1KProcessor().raw2mel(self, dataType)' for preparing mel data.
				 
	'''
	def __init__(self):
		pass

	def mel_gene(self):
		'''
		Generating melspectro for all songs.
		'''
		targetFolderPath = os.path.join(config.VIBRATO_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		if not os.path.exists(targetFolderPath): os.makedirs(targetFolderPath)
		for audioFileName in os.listdir(os.path.join(config.VIBRATO_PATH, 'songs')):
			print(audioFileName)
			audioFilePath = os.path.join(os.path.join(config.VIBRATO_PATH, 'songs', audioFileName))
			targetFilePath = os.path.join(targetFolderPath, audioFileName.split('.')[0] + '.pkl')
			y, _ = librosa.load(audioFilePath, config.SR)
			melSpectro = librosa.feature.melspectrogram(y, sr=config.SR, n_fft=config.FRAME_LEN, hop_length=config.HOP_LENGTH, 
				window='hann', n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX, power=1.0)
			logMelSpectro = librosa.amplitude_to_db(melSpectro, amin=1e-07)
			torch.save(logMelSpectro, targetFilePath)

	def makeFramePair_Song(self, audioFileName, step = 3):
		'''
		Make paired data for a certain song.
		Input:
			audioFileName: ----
			step: ----
		Output:
			x, y: ----
		'''
		melFolderPath = os.path.join(config.VIBRATO_PATH, config.AUDIO_PROCESSSING_METHOD, 'mel')
		melSeries = torch.load(os.path.join(melFolderPath, audioFileName.split('.')[0] + '.pkl'))
		x, y = [], []
		for winIndex in range(0, melSeries.shape[1] - config.WIN_SIZE + 1, step):
			xWin = melSeries[:, winIndex: winIndex + config.WIN_SIZE]
			x.append(xWin)
			y.append(0)		
		x = np.array(x)
		y = np.array(y)
		x = torch.from_numpy(x)
		y = torch.from_numpy(y)
		x = x.float()
		y = y.long()
		x = x.unsqueeze(1)
		return x, y	

	def datasetLoader(self):
		'''
		Dataset loader in a dict version, for fast dataset search.
		Output:
			speech_dict: ----
		'''
		speech_names = ['n', 'a', 'e', 'i', 'o', 'u']
		semitones = [0.01, 0.1, 0.3, 0.6, 1, 2, 4, 8]  # frequency deviation ranges in semitone.
		rates = [0.5, 1, 2, 4, 6, 8, 10]  # how fast is the vibrato [num_vibrato per second]
		print('-----------------Vibrato Dataset Loading--------------------------')
		speech_dict = {}
		for speech in speech_names:
			smt_dict = {}
			for smt in semitones:
				rt_dict = {}
				for rt in rates:
					x, y = self.makeFramePair_Song('modulated_%s_%d_%d_%d.wav' % (speech, 220, int(smt*100), int(rt*100)))
					rt_dict[rt] = [x, y]
				# rt_dict: DONE --> smt_dict
				smt_dict[smt] = rt_dict
			# smt_dict: DONE --> speech_dict
			speech_dict[speech] = smt_dict
		# for keys in speech_dict['a'].keys():
		# 	print(keys, end = ' ')
		# 	print(speech_dict['a'][keys].keys())
		return speech_dict


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
	VibratoPrcs = VibratoProcessor()
	# VibratoPrcs.datasetLoader()
	# VibratoPrcs.mel_gene()
	if(args.option == 'rawDataProcessing'):
		VibratoPrcs.mel_gene()
	else:
		pass
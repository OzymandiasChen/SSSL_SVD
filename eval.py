
# coding: utf-8
import os
import config
import argparse
import numpy as np
import datetime
import librosa
from dataloader.JamendoProcessor import JamendoProcessor
from dataloader.RWCProcessor import RWCProcessor
from dataloader.MIR1KProcessor import MIR1KProcessor
from dataloader.MedleyDBProcessor import MedleyDBProcessor
from dataloader.Musdb18Processor import Musdb18Processor
from dataloader.VibratoProcessor import VibratoProcessor
from models.SchCNN import SchCNN
from models.Res_ZeroMean_StdPool import Res_ZeroMean_StdPool
from utils import soft2Hard, deviceGetter
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from distiller import Distiller

'''
Funcs and classes: 
	(1). parse_args(): arguement parser for the python file running.
	(2). getNetPerformance(trueLabel, predictLabel): Model performance getter, acc, precison, r4ecall, tn, fp, fn, tp.
	(3). class evaluator(): evaluation for 'valid'/'test'/'test_song'.
	(4). class Tester(): testing.
Using:
	(a). For distiller evaluation, python eval.py --expName 'distillers_eval'
	(b). 
'''

def getNetPerformance(trueLabel, predictLabel):
	'''
	Model performance getter, acc, precison, r4ecall, tn, fp, fn, tp.
	Input:
		trueLabel: must be on 'cpu'.
		predictLabel: must be in hard version.
	Output:
		acc, f1, precision, recall, tn, fp, fn, tp: corresponding metrics.
	'''
	with torch.no_grad():
		acc = accuracy_score(trueLabel, predictLabel)
		f1 = f1_score(trueLabel, predictLabel)
		precision = precision_score(trueLabel, predictLabel)
		recall = recall_score(trueLabel, predictLabel)
		try:
			tn, fp, fn, tp = confusion_matrix(trueLabel, predictLabel).ravel()/(trueLabel.shape[0])
		except:
			tn, fp, fn, tp = -1, -1, -1, -1
		return acc, f1, precision, recall, tn, fp, fn, tp

class Evaluator():
	'''
	The class is structed with following paramaters and functions.
	Functions:
		(1). __init_(self): criterion and phase setter.
		(2). infoWritter(self, fo, loss, acc, f1, precision, recall, tn, fp, fn, tp): Evaluation info writter, both on screeen and in file
		(3). evaluation(self, X, Y, model, fo): evaluation for a certain datset, 'valid'/'test'/'test_song'
	Description:
		(a). (3) calls (2) to use.
	Using:
		(a). evaluator().evaluation(self, X, Y, model, fo) for a single model
		(b). evaluator().evaluation(self, X, Y, '', '') for disitillers
	'''
	def __init__(self, phase):
		'''
		criterion and phase setter.
		Input:
			phase: evaluator for whom 'test'/'valid'/'distillers_eval'/audioName
		Output:
			----
		'''
		self.phase = phase
		self.criterion = nn.BCELoss()

	def infoWritter(self, fo, loss, acc, f1, precision, recall, tn, fp, fn, tp):
		'''
		Evaluation info writter, both on screeen and in file
		Input:
			fo: log file opener.
			loss, acc, f1, precision, recall, tn, fp, fn, tp: corresponding metric
		Output:
			----
		'''
		print('[{}]\n --/--: loss: {:.3f}, acc: {:.3f}, F1: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(
			self.phase, loss, acc, f1, precision, recall))
		print('tn:{:.3f}, fp:{:.3f}, fn:{:.3f}, tp:{:.3f}'.format(tn, fp, fn, tp))
		if(self.phase != 'demo'):
			fo.write('[{}]\n --/--: loss: {:.3f}, acc: {:.3f}, F1: {:.3f}, precision: {:.3f}, recall: {:.3f}\n'.format(
				self.phase, loss, acc, f1, precision, recall))
			fo.write('tn:{:.3f}, fp:{:.3f}, fn:{:.3f}, tp:{:.3f}\n'.format(tn, fp, fn, tp))

	def evaluation(self, X, Y, model, fo):
		'''
		Evaluation for a certain datset, 'valid'/'test'/'test_song'
		Input:
			X, Y: dataset in torch tensor version.
			model: the model to be evaluated, if self.pahse == 'distillers', it could be NULL
			fo: log file opener, which could be also null if self.phase == 'distillers_eval'
		Output:
			output: model output on 'cpu'.
			loss, acc, f1, precision, recall, tn, fp, fn, tp: corresponding metric.
		'''
		torch_dataset = Data.TensorDataset(X, Y)
		dataLoader = Data.DataLoader(dataset = torch_dataset, batch_size = config.VALID_BATCH_SIZE, shuffle = False)
		# print("valid:{}".format(next(model.parameters()).is_cuda))
		device = deviceGetter()
		if(self.phase == 'distillers_eval'):
			distillers = Distiller()
			fo = open(os.path.join(config.PROJECT_PATH, 'distillers', 'evalDistiller.txt'), 'w+')
		else:
			model =model.to(device)
			model.eval()
		with torch.no_grad():
			output = []
			for _, (batch_x, batch_y) in enumerate(dataLoader):
				batch_x = batch_x.to(device)
				if(self.phase == 'distillers_eval'):
					output_batch = distillers.getDistillerOutput(batch_x, 'distillers_eval').to(device)
				else:
					output_batch = model(batch_x, 1)
				output.append(output_batch)
			output = torch.cat(output, axis = 0)
			output = output.cpu()
			loss = self.criterion(output, Y.float())
			acc, f1, precision, recall, tn, fp, fn, tp = getNetPerformance(Y, soft2Hard(output))
			self.infoWritter(fo, loss, acc, f1, precision, recall, tn, fp, fn, tp)
		if(self.phase == 'distillers_eval'):
			fo.close()
		return output.cpu(), loss, acc, f1, precision, recall, tn, fp, fn, tp

class Tester():
	'''
	The class is structed with following paramaters and functions.
	Functions:
		(1). __init__(self, expName): Set log/model path and model loadder.
		(2). songInfoWritter(self, output, Y, fo_test_song): True/predict info writter.
		(3). fram2Time(self, sampleIndex): Converting from sample index 2 time.
		(4). testsetLoader(self): Testset dataloader, for the whole dataset and song_level dataset.
		(5). test(self, modelTag): testing.
	Description:
		(a). (5) is the main function.
	Using:
		(a). python tester(expName).test(self)
	'''
	def __init__(self, expName, testsetName = None):
		'''
		Set log/model path and model loadder.
		Input:
			expName: experiment log folder name.
		Output:
			----
		'''
		self.logPath = os.path.join(config.PROJECT_PATH, 'logs', expName) # the path must exist
		self.lastModel = torch.load(os.path.join(self.logPath, 'lastModel.pkl'))
		self.bestModel = torch.load(os.path.join(self.logPath, 'bestModel_acc.pkl'))
		# 做一个屏蔽，但是我脑子已经糊了
		if(testsetName != None):
			self.testsetName = testsetName
		else: 
			# testsetName == None
			self.testsetName = config.TEST_SET_NAME
		# self.model = torch.load(os.path.join(self.logPath, config.TESTED_MODEL))

	def songInfoWritter(self, output, Y, fo_test_song):
		'''
		True/predict info writter.
		Input:
			output, Y: model output and the true label
			fo_test_song: song-level log file opener.
		Output:
			----
		'''
		hardLabel = soft2Hard(output)
		TF_flag = (hardLabel == Y)
		for i in range(len(Y)):
			timeStamp = self.fram2Time(i)
			fo_test_song.write('Time: {},  TF: {:7s},  predictLabel: {},  predictProb: {:.3f},  TrueLabel: {}\n'.format(
				str(datetime.timedelta(seconds=timeStamp))[2:-3], 'True' if hardLabel[i] == Y[i] else 'False', hardLabel[i], output[i], Y[i]))

	def fram2Time(self, sampleIndex):
		'''
		Converting from sample index 2 time.
		Input:
			sampleIndex: meaning windowIndex as well, in other word, the sample index in dataset.
		Output:
			time: real number time in seconds for each sample.
		'''
		winIndex = sampleIndex
		frameIndex = winIndex * config.TEST_STEP + config.WIN_SIZE//2
		time = librosa.frames_to_time(frameIndex, sr=config.SR, hop_length=config.HOP_LENGTH, n_fft=config.FRAME_LEN)
		return time

	def testsetLoader(self):
		'''
		Testset dataloader, for the whole dataset and song_level dataset.
		Input:
			----
		Output:
			X_test, Y_test: the testset
			nameList_test_song: audio name list for the testset
			test_song: dictionary version, in the format of {audioName: [X, Y]}, which can be iterated.
		'''
		if(self.testsetName == 'Jamendo'):
			JamendoDataLoader = JamendoProcessor()
			X_test, Y_test = JamendoDataLoader.datasetLoader('test')
			# newdict.keys()
			# nameList_test_song = JamendoDataLoader.loadDataSetFileName('test')
			test_song = JamendoDataLoader.datasetLoader('test_song')
		elif(self.testsetName == 'RWC_part' or self.testsetName == 'RWC'):
			RWCDataLoader = RWCProcessor('RWC-MDB-P-2001')
			X_test, Y_test, test_song = RWCDataLoader.datasetLoader(config.RWC_TEST, 'test')
		elif(self.testsetName == 'MIR1K'):
			MIR1KDataLoader = MIR1KProcessor()
			X_test, Y_test, test_song = MIR1KDataLoader.datasetLoader(config.MIR1K_TEST, 'test')
		elif(self.testsetName == 'MedleyDB'):
			MedleyDBDataLoader = MedleyDBProcessor()
			X_test, Y_test, test_song = MedleyDBDataLoader.datasetLoader(config.MedleyDB_TEST, 'test')
		elif(self.testsetName == 'MUSDB18'):
			Musdb18DataLoader = Musdb18Processor()
			X_test, Y_test, test_song = Musdb18DataLoader.datasetLoader('test')
		else:
			pass
		return X_test, Y_test, test_song
		# , nameList_test_song

	def test(self, modelTag):
		'''
		testing.
		Input:
			modelTag: which model you would like to test,
		Output:
			----
		'''
		if(modelTag == 'Test_lastModel'):
			model = torch.load(os.path.join(self.logPath, 'lastModel.pkl'))
		elif(modelTag == 'Test_bestAccModel'):
			model = torch.load(os.path.join(self.logPath, 'bestModel_acc.pkl'))
		if not os.path.exists(os.path.join(self.logPath, modelTag)):
			os.makedirs(os.path.join(self.logPath, modelTag))
		fo_test_dataset = open(os.path.join(self.logPath, modelTag, 'testLog.txt'), 'w+')
		songReportFolder = os.path.join(self.logPath, modelTag, 'song_test_reports')
		if not os.path.exists(songReportFolder):
			os.makedirs(songReportFolder)
		X_test, Y_test, test_song = self.testsetLoader()
		evaluatorHelper = Evaluator('Testset')
		_, _, _, _, _, _, _, _, _, _ = evaluatorHelper.evaluation(X_test, Y_test, model, fo_test_dataset)
		for audioName in test_song.keys():
			evaluatorHelper = Evaluator(audioName)
			output_song, _, _, _, _, _, _, _, _, _ = evaluatorHelper.evaluation(
				test_song[audioName][0], test_song[audioName][1], model, fo_test_dataset)			
			fo_test_song = open(os.path.join(songReportFolder, audioName.split('.')[0] + '.txt'), 'w+')
			self.songInfoWritter(output_song, test_song[audioName][1], fo_test_song)

class preasureTester():
	'''
		The class is structed with following paramaters and functions.
		Functions:
			(1). __init__(self): Initializer.
			(2). accGetter(self, X, Y, model): get accuray score, a helper function.
			(3). songlevlTest(self): ----
			(4). vibratoTest(self): ----
			(5). vocalSNRTest(self): ----
		Description:
			----
		Using:
			(a). 
				 Call 'preasureTester().songlevlTest(self)'
				 Call 'preasureTester().vibratoTest(self)'
				 Call 'preasureTester().vocalSNRTest(self)'
 
	'''
	def __init__(self, expName):
		'''
		Initializer.
		'''
		self.logPath = os.path.join(config.PROJECT_PATH, 'logs', expName) # the path must exist
		self.lastModel = torch.load(os.path.join(self.logPath, 'lastModel.pkl'))
		self.bestModel = torch.load(os.path.join(self.logPath, 'bestModel_acc.pkl'))

	def accGetter(self, X, Y, model):
		'''
		get accuray score.
		Input:
			X, Y: --
			model: ---
		Output:
			accscore: in 3 dicimal
		'''
		torch_dataset = Data.TensorDataset(X, Y)
		dataLoader = Data.DataLoader(dataset = torch_dataset, batch_size = config.VALID_BATCH_SIZE, shuffle = False)
		device = deviceGetter() 
		model =model.to(device)
		model.eval()
		with torch.no_grad():
			output = []
			for _, (batch_x, batch_y) in enumerate(dataLoader):
				output_batch = model(batch_x.to(device), 1)
				output.append(output_batch)
			output = torch.cat(output, axis = 0)
			acc = accuracy_score(Y, soft2Hard(output.cpu()))
			# soft2Hard(output)
		return round(acc, 3) 

	def songlevlTest(self):
		'''
		'''
		fo_last = open(os.path.join(self.logPath, 'Test_lastModel', 'pt_songlevel.txt'), 'w+') 
		fo_best = open(os.path.join(self.logPath, 'Test_bestAccModel', 'pt_songlevel.txt'), 'w+') 
		Musdb18Prcs = Musdb18Processor()
		for dataType in ['raw', 'plus6', 'minus6']:
			X, Y = Musdb18Prcs.datasetLoader(phase = 'pressure_test', dataType = dataType)
			evaluatorHelper = Evaluator(dataType + '_lastModel')
			_, _, _, _, _, _, _, _, _, _ = evaluatorHelper.evaluation(X, Y, self.lastModel, fo_last)
		for dataType in ['raw', 'plus6', 'minus6']:
			X, Y = Musdb18Prcs.datasetLoader(phase = 'pressure_test', dataType = dataType)
			evaluatorHelper = Evaluator(dataType + '_bestModel')
			_, _, _, _, _, _, _, _, _, _ = evaluatorHelper.evaluation(X, Y, self.bestModel, fo_best)		
		# dataType: 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6'
		fo_last.close()
		fo_best.close()

	def vibratoTest(self):
		'''
		'''
		fo_last = open(os.path.join(self.logPath, 'Test_lastModel', 'pt_vibrato.txt'), 'w+') 
		fo_best = open(os.path.join(self.logPath, 'Test_bestAccModel', 'pt_vibroto.txt'), 'w+')
		speech_names = ['n', 'a', 'e', 'i', 'o', 'u']
		semitones = [0.01, 0.1, 0.3, 0.6, 1, 2, 4, 8]  # frequency deviation ranges in semitone.
		rates = [0.5, 1, 2, 4, 6, 8, 10]  # how fast is the vibrato [num_vibrato per second]
		VibratoPrcs = VibratoProcessor()
		speech_dict = VibratoPrcs.datasetLoader()
		speech_ans_last = []
		speech_ans_best = []
		for speech in speech_names:
			smt_ans_last = []
			smt_ans_best = []
			for smt in semitones:
				rt_ans_last = []
				rt_ans_best = []
				for rt in rates:
					data = speech_dict[speech][smt][rt]
					X, Y = data[0], data[1]
					acc_last = self.accGetter(X, Y, self.lastModel)
					acc_best = self.accGetter(X, Y, self.bestModel)
					rt_ans_last.append(acc_last)
					rt_ans_best.append(acc_best)
				smt_ans_last.append(rt_ans_last)
				smt_ans_best.append(rt_ans_best)
			speech_ans_last.append(smt_ans_last)
			speech_ans_best.append(smt_ans_best)
		speech_ans_last = np.array(speech_ans_last)
		speech_ans_best = np.array(speech_ans_best)
		for i in range(6):
			print(speech_names[i])
			print(np.matrix(speech_ans_last[i]))
			fo_last.write('{}\n'.format(speech_names[i]))
			fo_last.write('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in speech_ans_last[i]]))
		for i in range(6):
			print(speech_names[i])
			print(np.matrix(speech_ans_best[i]))
			# print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in speech_ans_best[i]]))
			fo_best.write('{}\n'.format(speech_names[i]))
			fo_best.write('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in speech_ans_best[i]]))
		fo_last.close()
		fo_best.close()

	def vocalSNRTest(self):
		'''
		'''
		fo_last = open(os.path.join(self.logPath, 'Test_lastModel', 'pt_vocalSNR.txt'), 'w+') 
		fo_best = open(os.path.join(self.logPath, 'Test_bestAccModel', 'pt_vocalSNR.txt'), 'w+') 
		Musdb18Prcs = Musdb18Processor()
		for dataType in ['raw', 'vocal_plus6', 'vocal_minus6']:
			X, Y = Musdb18Prcs.datasetLoader(phase = 'pressure_test', dataType = dataType)
			evaluatorHelper = Evaluator(dataType + '_lastModel')
			_, _, _, _, _, _, _, _, _, _ = evaluatorHelper.evaluation(X, Y, self.lastModel, fo_last)
		for dataType in ['raw', 'plus6', 'minus6']:
			X, Y = Musdb18Prcs.datasetLoader(phase = 'pressure_test', dataType = dataType)
			evaluatorHelper = Evaluator(dataType + '_bestModel')
			_, _, _, _, _, _, _, _, _, _ = evaluatorHelper.evaluation(X, Y, self.bestModel, fo_best)		
		# dataType: 'raw'/'plus6'/'minus6'/'vocal_plus6'/'vocal_minus6'
		fo_last.close()
		fo_best.close()

	def __del__(self):
		pass

class DistillerEvaluator(): # 这段写的真的不好,应该合适地融入到以上的步骤中。
	'''
	The class is structed with following paramaters and functions.
	Functions:
		(1). __init__(self): initiallization
		(2). testsetLoader(self):Testset dataloader, for a corresponding whole dataset
		(3). KDEval(self): evalutor for loss, acc, f1, precision, recall, tn, fp, fn, tp, 
	Description:
		(a). (3) is the main function.KDEval()
	Using:
		(a). DistillerEvaluator()
	'''
	def __init__(self):
		pass

	def testsetLoader(self):
		'''
		Testset dataloader, for a corresponding whole dataset
		Input:
			----
		Output:
			X_test, Y_test: the testset
		'''
		# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! to be modified.
		JamendoDataLoader = JamendoProcessor()
		if(config.TEST_SET_NAME == 'Jamendo'):
			X_test, Y_test = JamendoDataLoader.datasetLoader('test')
		return X_test, Y_test

	def KDEval(self):
		'''
		evalutor for loss, acc, f1, precision, recall, tn, fp, fn, tp, 
		Input:
			----
		Output:
			----
		'''
		X_test, Y_test = self.testsetLoader()
		evaluator = Evaluator('distillers_eval')
		evaluator.evaluation(X_test, Y_test, '', '')

def parse_args():
	'''
	arguement parser for the python file running
	argument:
		'--expName': experiment log folder name.
	'''
	description = 'Trainer arguement parser'
	parser = argparse.ArgumentParser(description=description)
	parser.add_argument('--expName',help = '')
	parser.add_argument('--type',help = '')
	parser.add_argument('--testsetName',help = '')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	if(args.expName == 'distillers_eval'):
		distillerEvaluator = DistillerEvaluator()
		distillerEvaluator.KDEval()
	else:
		if(args.type == 'test'):
			tester = Tester(args.expName, args.testsetName)
			tester.test('Test_lastModel')
			tester.test('Test_bestAccModel')
		elif(args.type == 'pt_sl'):
			pressureTester = preasureTester(args.expName)
			pressureTester.songlevlTest()
		elif(args.type == 'pt_vcSNR'):
			pressureTester = preasureTester(args.expName)
			pressureTester.vocalSNRTest()
		elif(args.type == 'pt_vbrt'):
			pressureTester = preasureTester(args.expName)
			pressureTester.vibratoTest()
			# pressureTester.vibratoTest()
			# pressureTester.vocalSNRTest()

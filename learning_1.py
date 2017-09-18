#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:22:24 2017

@author: dirktheeng
"""

import numpy as np
import tflearn
import tensorflow as tf
import os
import sys
import pickle
import glob

class NeuralNetowrk():
    def __init__(self, filePrefix = 'CH4_BFER', ext='.npy', nFeatures=8, nLabels = 7, learnerDir='learner'):
        self._filePrefix = filePrefix
        self._ext = ext
        self._nFeatures = nFeatures
        self._nLabels = nLabels
        self.trainingDataDir = os.path.join('.', self._filePrefix + '_Training_Data/')
        self.clearNetwork()
        
        self.parentdir = os.getcwd()
        self.learnerDir = os.path.join(self.parentdir, learnerDir)
        if not os.path.exists(self.learnerDir):
            os.makedirs(self.learnerDir)
    
    def clearNetwork(self):
        self.net = None
        self.model = None
        self._modelParams = {}
        
    def clearData(self):
        self.trainFeatures = None
        self.trainLabels = None
        self.testFeatures = None
        self.testLabels = None
    
    def setupNewNet(self, layers = [10, 10], activation=['sigmoid']*3,
                    optimizer='adam', **optimizerOptions):
        if 'loss' not in optimizerOptions:
            optimizerOptions['loss'] = 'mean_square'
        self._modelParams = {'layers': layers,
                             'activation': activation,
                             'optimizer': optimizer,
                             'optimizerOptions': optimizerOptions}
        if self.net is not None:
            self.clearNetwork()
        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, self._nFeatures])
        for i,nNodes in enumerate(layers):
            net = tflearn.fully_connected(net, nNodes, activation=activation[i])
        net = tflearn.fully_connected(net, self._nLabels, activation=activation[-1])
        net = tflearn.regression(net, optimizer=optimizer, **optimizerOptions)
        self.net = net
        self.model = tflearn.DNN(net, tensorboard_verbose=0)

    def arrayToTFData(self, array):
        features = array[:,:self._nFeatures]
        labels = array[:, self._nFeatures:]
        return features, labels

    def createTrainingData(self, nFiles=2, testingFrac=0.2, save=True):
        dataArrays = []
        for fNum in range(nFiles):
            fName = '-'.join([self._filePrefix, str(fNum).rjust(10, '0')])+self._ext
            fPath = os.path.join(self.trainingDataDir, fName)
            dataArrays.append(np.load(fPath))
        data = np.vstack(dataArrays)
        totalSamples, flWidth = data.shape
        testingNum = int(np.ceil(totalSamples * testingFrac))
        mask = [False]*totalSamples
        randTestingInts =  list(np.random.randint(0,totalSamples, testingNum*2))
        randTestingInts = list(set(randTestingInts))[:testingNum]
        for i in randTestingInts:
            mask[i]=True
        notMask = [not(i) for i in mask]
        trainData = data[notMask, :]
        testData = data[mask, :]
        if save:
            trainfileName = '-'.join([self._filePrefix, 'training', str(totalSamples), str(testingFrac)])+self._ext
            testfileName  = '-'.join([self._filePrefix, 'testing' , str(totalSamples), str(testingFrac)])+self._ext
            trainPathName = os.path.join(self.trainingDataDir,trainfileName)
            testPathName = os.path.join(self.trainingDataDir,testfileName)
            if os.path.exists(trainPathName):
                print('warning: training file already exists, not overwriting, save manually')
            else:
                np.save(trainPathName, trainData)
                np.save(testPathName, testData)
        self.trainFeatures, self.trainLabels = self.arrayToTFData(trainData)
        self.testFeatures, self.testLabels = self.arrayToTFData(testData)

    def loadTrainingData(self, nSamples=100000, testingFrac=0.2):
        trainfileName = '-'.join([self._filePrefix, 'training', str(nSamples), str(testingFrac)])+self._ext
        testfileName  = '-'.join([self._filePrefix, 'testing' , str(nSamples), str(testingFrac)])+self._ext
        trainPathName = os.path.join(self.trainingDataDir,trainfileName)
        testPathName = os.path.join(self.trainingDataDir,testfileName)
        if not os.path.exists(trainPathName) or not os.path.exists(testPathName):
            print('Cant load, data missing')
            return
        print('Loading Test Data: ' + testPathName)
        dss = list(range(self._nFeatures)) + [self._nFeatures+4]
        testData = np.load(testPathName)[:,dss]
        print('Loading Training Data: ' + trainPathName)
        trainData = np.load(trainPathName)[:,dss]
        print('converting test to TF')
        self.testFeatures, self.testLabels = self.arrayToTFData(testData)
        print('convering training to TF')
        self.trainFeatures, self.trainLabels = self.arrayToTFData(trainData)
        print('Done')
            
    def _fitModel(self, show_metric = True, epochNum = 0):
        epochDir = os.path.join(self.learnerDir,'epoch_'+str(epochNum).rjust(10, '0'))
        if not os.path.exists(epochDir):
            os.makedirs(epochDir)
        os.chdir(epochDir)
        tfile = './training_output.txt'
        if os.path.exists(tfile):
            os.remove(tfile)
        print('-----------------------')
        print('starting training on Epoch ' + str(epochNum))
        original = sys.stdout
        sys.stdout = open(tfile, 'a')
        self.model.fit(self.trainFeatures, self.trainLabels, n_epoch=1,
                       show_metric=show_metric)
        sys.stdout = original
        print('done training')
        print('saving net output')
        self.model.save('nnet')
        print('done saving')
        print('calculating cross validation')
        trainMSE, testMSE = self.crossValidate()
        np.save('./crossval.npy', np.asarray([trainMSE, testMSE]))
        print('training MSE: ' + str(trainMSE) + '  testing MSE: ' + str(testMSE))
        print('-----------------------')
        print()
        os.chdir(self.parentdir)

    def trainNN(self, show_metric = True, epochs=1):
        if self.model is None:
            print('model does not exist, set up NN first')
            return
        if self.trainFeatures is None or self.testFeatures is None:
            print('training/testing data, load or create first')
            return
        #self.removeTrainingFile()
        epochDirs = glob.glob(self.learnerDir+'/epoch_*')
        if epochDirs:
            epochDirs.sort()
            lastDir = epochDirs[-1]
            if 'epoch_' in lastDir:
                startEpoch = int(lastDir.split('epoch_')[-1])+1
            else:
                startEpoch = 0
        else:
            startEpoch = 0
        modelParamsPath = os.path.join(self.learnerDir,'model_params.pickle')
        if os.path.exists(modelParamsPath):
            os.remove(modelParamsPath)
        with open(modelParamsPath, 'wb') as f:
            pickle.dump(self._modelParams, f)
        for i in range(epochs):
            self._fitModel(show_metric=show_metric, epochNum=i+startEpoch)
            self.crossValidate()
        
    def crossValidate(self):
        if self.model is None:
            print('model does not exist, set up NN first')
            return
        if self.trainFeatures is None or self.testFeatures is None:
            print('training/testing data, load or create first')
            return
        trainingPred = self.model.predict(self.trainFeatures)
        testingPred = self.model.predict(self.testFeatures)
        trainMSE = self.calcMeanSquaredError(self.trainLabels, trainingPred)
        testMSE = self.calcMeanSquaredError(self.testLabels, testingPred)
        return trainMSE, testMSE
        
    def reloadNN(self, epoch=-1, **optimizerOptions):
        modelParPath = os.path.join(self.learnerDir, 'model_params.pickle')
        if not os.path.exists(modelParPath):
            print('could not find: ' + modelParPath)
            return
        with open(modelParPath, 'rb') as f:
            modelParams = pickle.load(f)
        
        self.setupNewNet(layers=modelParams['layers'], activation=modelParams['activation'],
                         optimizer=modelParams['optimizer'], **optimizerOptions)
        epochDirs = glob.glob(self.learnerDir+'/epoch_*')
        epochDirs.sort()
        if epoch <0:
            epochDir = epochDirs[-1]
        else:
            found = False
            for epochDir in epochDirs:
                if 'epoch_'+str(epoch).rjust(10, '0') in epochDir:
                    found = True
                    break
            if not found:
                print('did not find directory')
                return
        self.model.load(os.path.join(epochDir,'nnet'))
        

    def calcMeanSquaredError(self, y, yhat):
        n = y.shape[0]
        mse = np.sum((y-yhat)**2)/n
        return mse

if __name__ == '__main__':
    NN = NeuralNetowrk(learnerDir='standard200n2_var4_adam_sigmoid_fitnet_linear', nLabels=1)
    nlayers = 2
#    NN.reloadNN(learning_rate=0.0001)
    
    NN.setupNewNet(optimizer='adam',layers=[100]*nlayers, activation=['sigmoid']*(nlayers)+['linear'],
                   learning_rate=0.01, batch_size = 1024)
    NN.loadTrainingData()
    NN.trainNN(epochs=40)

    
    
    
    
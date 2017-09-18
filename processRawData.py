#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:32:04 2017

@author: dirktheeng
"""
import os, glob
import numpy as np
from scipy.interpolate import interp1d



rawDataDir = 'CH4_BFER_Raw_Data/'

def findMinMax(filePrefix='CH4_BFER', ext='.npy'):
    rawDataDir = os.path.join('.', filePrefix + '_Raw_Data')
    fileWildCard = os.path.join(rawDataDir, filePrefix+'*'+ext)
    fileList = glob.glob(fileWildCard)
    
    testD = np.load(fileList[0])
    tdShape = testD.shape
    nInputs = tdShape[1]
    minVals = np.ones(nInputs)*1e12
    maxVals = np.ones(nInputs)*-1e12
    saveArr = np.zeros((3,2))
    for ii,f in enumerate(fileList):
        if ii % 10 == 0:
            print('starting loop: ', ii)
        data = np.load(f)
        minv = np.min(data, 0)
        minVals = np.minimum(minVals, minv)
        maxv = np.max(data,0)
        maxVals = np.maximum(maxVals, maxv)
    saveArr[:,0] = minVals
    saveArr[:,1] = maxVals
    saveRangeFile = os.path.join(rawDataDir, filePrefix+'_Ranges'+ext)
    np.save(saveRangeFile, saveArr)

def convertDataToNN(data, dataRanges, nnMin=0.0, nnMax=1.0):
    timeData = data[:,0]
    tData = data[:,1]
    pData = data[:,2]
    xData = data[:,3:]
    timeRangeData = dataRanges[0,:]
    timeRange = timeRangeData[1] - timeRangeData[0]
    tRangeData = dataRanges[1,:]
    tRange = tRangeData[1] - tRangeData[0]
    pRangeData = dataRanges[2,:]
    pRange = pRangeData[1] - pRangeData[0]
    xRangeData = np.zeros(2)
    xRangeData[0] = np.min(dataRanges[3:,0])
    xRangeData[1] = np.max(dataRanges[3:,1])
    xRange = xRangeData[1] - xRangeData[0]
    nnRange = nnMax - nnMin
    data[:, 0] = (timeData - timeRangeData[0])/timeRange*nnRange+nnMin
    data[:, 1] = (tData - tRangeData[0])/tRange*nnRange+nnMin
    data[:, 2] = (pData - pRangeData[0])/pRange*nnRange+nnMin
    data[:, 3:] = (xData - xRangeData[0])/xRange*nnRange+nnMin
    return data

def generateRandomFeatureLabel(rawData, inputRanges, outputRanges, nnMin=0.0, nnMax=1.0):
    # look at data to create the appropriate sized feature/label array
    dataShape = rawData.shape
    nFeatures = min(dataShape)
    nLabels = nFeatures-1
    featureLabel = np.zeros(nFeatures+nLabels)
    nnRange = nnMax - nnMin
    
    #first find the dt that is associated with the feture
    nnDt = np.random.random()

    #load data coded in range 0-1
    initialCond = np.copy(rawData[0,:])
    inletTRanges = inputRanges[0,:]
    inletPRanges = inputRanges[1,:]
    inletXRanges = inputRanges[2,:]
    rawData = convertDataToNN(rawData, outputRanges)
    
    #set feature
    featureLabel[:nFeatures] = rawData[0,:]
    featureLabel[0] = nnDt
    featureLabel[1] = (initialCond[1] - inletTRanges[0])/(inletTRanges[1] - inletTRanges[0])*nnRange+nnMin
    featureLabel[2] = (initialCond[2] - inletPRanges[0])/(inletPRanges[1] - inletPRanges[0])*nnRange+nnMin
    featureLabel[3:nFeatures] = (initialCond[3:nFeatures] - inletXRanges[0])/(inletXRanges[1] - inletXRanges[0])*nnRange+nnMin
    
    # set up for interpolation
    times = rawData[:,0]
    interpList = [interp1d(times, rawData[:,i+1])  for i in range(nFeatures-1)]
    vals = [float(interp(nnDt)) for interp in interpList]
    featureLabel[nFeatures:] = np.array(vals)
    return featureLabel, nFeatures, nLabels

def generateTrainingFiles(nfiles=1, samplesPerFile = 100,
                          filePrefix='CH4_BFER', ext='.npy', fileStartNum=0,
                          inputTRanges=[300.0, 2500.0],
                          inputPRanges=[80000.0, 400000.0],
                          inputXRanges= [0.0, 1.0]):
    trainingDataDir = os.path.join('.', filePrefix + '_Training_Data')
    if not os.path.exists(trainingDataDir):
        os.makedirs(trainingDataDir)
    
    #read raw data and ranges
    rawDataDir = os.path.join('.', filePrefix + '_Raw_Data')
    outputRanges = np.load(os.path.join(rawDataDir, filePrefix+'_Output_Ranges'+ext))
    inputRanges = np.asarray([inputTRanges, inputPRanges, inputXRanges])
    
    # cound the number of raw data files
    fileWildCard = os.path.join(rawDataDir, filePrefix+'-*'+ext)
    fileList = glob.glob(fileWildCard)
    nRawDataFiles = len(fileList)
    fName = '-'.join([filePrefix, str(0).rjust(10, '0')])+ext
    firstFileData = np.load(os.path.join(rawDataDir, fName))
    
    # get number of features and labels
    _, nFeatures, nLabels = generateRandomFeatureLabel(firstFileData, inputRanges, outputRanges)
    
    fileData = np.zeros((samplesPerFile, nFeatures+ nLabels))
    
    for fileNum in range(nfiles):
        for sampleNum in range(samplesPerFile):
            if sampleNum % 500 == 0.0:
                print('file num: ', fileNum, 'sample num: ', sampleNum)
            rawFileNum = np.random.randint(0,nRawDataFiles)
            fName = '-'.join([filePrefix, str(rawFileNum).rjust(10, '0')])+ext
            fPath = os.path.join(rawDataDir, fName)
            data = np.load(fPath)
            fileData[sampleNum, :], _, _ = generateRandomFeatureLabel(data, inputRanges, outputRanges)
        trainingFileName = '-'.join([filePrefix, str(fileNum+fileStartNum).rjust(10, '0')])+ext
        trainingFilePath = os.path.join(trainingDataDir, trainingFileName)
        np.save(trainingFilePath, fileData)
            
    

if __name__ == '__main__':
#    rawData = np.load('./CH4_BFER_Raw_Data/CH4_BFER-0000000001.npy')
#    ranges = np.load('./CH4_BFER_Raw_Data/CH4_BFER_Ranges.npy')
#    fL = generateRandomFeatureLabel(rawData, ranges)
#    print('feature: ', list(fL[:8]))
#    print('label: ', list(fL[8:]))
    generateTrainingFiles(nfiles=100, samplesPerFile=100000, fileStartNum=0)
#    data = np.load('./CH4_BFER_Training_Data/CH4_BFER-0000000000.npy')
#    print(data)
    
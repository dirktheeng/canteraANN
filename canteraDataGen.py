#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:32:04 2017

@author: dirktheeng
"""

import cantera as ct
import numpy as np
import os

p = 101325.0
T_low = 300
T_high = 2500
T_range = T_high - T_low
P_low  = 80000.0
P_high = 400000.0
P_range = P_high - P_low
dt = 1e-6

gas = ct.Solution('2S_CH4_BFER2.cti', 'CH4_BFER')
m=gas.n_species

def generateRandomGas(Tlow, Trange, Plow, Prange):
    mix = np.random.random(m)
    fAvail = 1.0 - mix[0]
    for i in range(m-2):
        num = fAvail*mix[i+1]
        fAvail -= num
        mix[i+1] = num
    mix[-1] = 1.0 - np.sum(mix[:-1])
    np.random.shuffle(mix)
    temp = np.random.rand()
    temp = temp* Trange + Tlow
    p = np.random.rand()
    p = p* Prange + Plow
    gas.TPX = temp, p, mix
    
def runReactor(num_steps = 10000, dt = 1e-6):
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    time = 0.0
    iN2 = gas.species_index('N2')
    gx = list(gas.X)
    gx.pop(iN2)
    state = [0.0, gas.T, gas.P] + gx
    states = [state]*(num_steps+1)
    for n in range(num_steps):
        time += dt
        sim.advance(time)
        gx = list(gas.X)
        gx.pop(iN2)
        state = [time, gas.T, gas.P] + gx
        states[n+1]=state
    return np.asarray(states)

def runReactorForRates(num_steps = 10000, dt = 1e-6):
    initialGasState = list(gas.TPX)
    x = initialGasState.pop(-1)
    species = x.tolist()
    iN2 = gas.species_index('N2')
    species.pop(iN2)
    initialGasState += species
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    time = 0.0
    rateData = np.zeros((num_steps, r.kinetics.n_reactions))
    for n in range(num_steps):
        rateData[n,:] = r.kinetics.net_rates_of_progress
        time += dt
        sim.advance(time)
    integrated = np.zeros(rateData.shape)
    integrated[1:,] = np.cumsum((rateData[1:,:]+rateData[:-1,:])/2.0 * dt, axis=0)
    return initialGasState, integrated

def estimateMinMax(Tlow, Trange, Plow, Prange, nLoops = 1000, dt = 1e-6):
    minVals = np.ones(m+3)*1e12
    maxVals = np.ones(m+3)*-1e12
    for ii in range(nLoops):
        if ii % 10 == 0:
            print('starting loop: ', ii)
        try:
            generateRandomGas(Tlow, Trange, Plow, Prange)
            states = runReactor(dt = dt)
            minv = np.min(states, 0)
            minVals = np.minimum(minVals, minv)
            maxv = np.max(states,0)
            maxVals = np.maximum(maxVals, maxv)
        except Exception as e:
            print(e)
    return minVals, maxVals

def generateData(Tlow, Trange, Plow, Prange,
                 nLoops = 1000, dt = 1e-6,
                 baseFile = 'CH4_BFER', ext='.npy', startNum=0,
                 num_steps = 10000):
    count = startNum
    dirName = os.path.join('.', baseFile+'_Raw_Data')
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    minOutputVals = np.ones(m+2)*1e12
    maxOutputVals = np.ones(m+2)*-1e12
    saveOutputArr = np.zeros((m+2,2))
    minInputVals = np.ones(m+2)*1e12
    maxInputVals = np.ones(m+2)*-1e12
    saveInputArr = np.zeros((m+2,2))
    for ii in range(nLoops):
        if ii % 10 == 0:
            print('starting loop: ', ii)
        try:
            generateRandomGas(Tlow, Trange, Plow, Prange)
            states = runReactor(num_steps = num_steps, dt = dt)
            fName = '-'.join([baseFile, str(count).rjust(10, '0')])
            fName += ext
            fName = os.path.join('.', baseFile+'_Raw_Data', fName)
            np.save(fName, states)
            count+=1
            minoutputv = np.min(states, 0)
            minOutputVals = np.minimum(minOutputVals, minoutputv)
            maxoutputv = np.max(states,0)
            maxOutputVals = np.maximum(maxOutputVals, maxoutputv)
            minInputVals = np.minimum(minInputVals, states[0,:])
            maxInputVals = np.maximum(maxInputVals, states[0,:])
        except Exception as e:
            print(e)
    saveOutputArr[:,0] = minOutputVals
    saveOutputArr[:,1] = maxOutputVals
    saveInputArr[:,0] = minInputVals
    saveInputArr[:,1] = maxInputVals
    
    saveOutputRangeFile = os.path.join(dirName, baseFile + '_Output_Ranges'+ext)
    saveInputRangeFile = os.path.join(dirName, baseFile + '_Input_Ranges'+ext)
    np.save(saveOutputRangeFile, saveOutputArr)
    np.save(saveInputRangeFile, saveInputArr)
    
def generateDataForRates(Tlow, Trange, Plow, Prange,
                 nLoops = 1000, dt = 1e-9,
                 baseFile = 'CH4_BFER_Rates', ext='.npy', fileNum=0,
                 numSteps = 1200000, nRandSamples = 100):

    dirName = os.path.join('.', baseFile+'_Raw_Data')
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        
    rTest = ct.IdealGasReactor(gas)
    maxTime = numSteps*dt
    nFeatures = gas.n_species+2
    nLabels = rTest.kinetics.n_reactions
    
#    minOutputVals = np.ones(nRxns)*1e12
#    maxOutputVals = np.ones(nRxns)*-1e12
#    saveOutputArr = np.zeros((nRxns,2))
    minInputVals = np.zeros(nFeatures)
    maxInputVals = np.ones(nFeatures)
    minInputVals[0] = 0.0
    maxInputVals[0] = maxTime   
    minInputVals[1:3] = [Tlow, Plow]
    maxInputVals[1:3] = [Tlow+Trange, Plow+Prange]
    saveInputArr = np.zeros((nFeatures,2))
    saveInputArr[:,0] = minInputVals
    saveInputArr[:,1] = maxInputVals
    for ii in range(nLoops):
        print('starting loop: ', ii)
        randSamples = np.zeros((nRandSamples, nFeatures+nLabels))
        continueCalc = False
        try:
            # generate high res rate integration
            generateRandomGas(Tlow, Trange, Plow, Prange)
            initialState, rates = runReactorForRates(num_steps = numSteps, dt = dt)
            continueCalc = True
        except Exception as e:
            print(e)
            
        if continueCalc:
            maxTime = numSteps * dt
            sampleTimes = np.random.random(nRandSamples)
            floatIntTime = sampleTimes*numSteps
            sampleTimes *= maxTime
            indexToData = np.floor(floatIntTime)
            fractionalStep = floatIntTime - indexToData
            
            for i in range(nRandSamples):
                ind = int(indexToData[i])
                frc = fractionalStep[i]
                rtsL = rates[ind,:]
                rtsH = rates[ind+1,:]
                interpRts = (rtsH-rtsL) * frc + rtsL
                randSamples[i,:] = [sampleTimes[i]] + initialState + interpRts.tolist()
        if ii == 0:
            saveArr = randSamples
        else:
            saveArr = np.vstack((saveArr,randSamples))
                
    np.random.shuffle(saveArr)
    saveInputRangeFile = os.path.join(dirName, baseFile + '_Input_Ranges'+ext)
    np.save(saveInputRangeFile, saveInputArr)
    saveDataFile = os.path.join(dirName, baseFile + '_Raw_data_'+str(fileNum).rjust(10, '0')+ext)
    np.save(saveDataFile, saveArr)
    saveFL = os.path.join(dirName, baseFile + '_num_FL_'+ext)
    np.save(saveFL, [nFeatures, nLabels])
    
def genRateData(Tlow, Trange, Plow, Prange, nFiles = 10,
                nLoops = 10000, dt = 1e-9,
                baseFile = 'CH4_BFER_Rates', ext='.npy', fileStartNum=0,
                numSteps = 1200000, nRandSamples = 100):
    fNums = [x+fileStartNum for x in range(nFiles)]
    for f in fNums:
        generateDataForRates(Tlow, Trange, Plow, Prange,
                 nLoops = nLoops, dt = dt,
                 baseFile = baseFile, ext=ext, fileNum=f,
                 numSteps = numSteps, nRandSamples = nRandSamples)
    
if __name__ == '__main__':
    genRateData(300,1500, 100000, 500000, nFiles = 1, fileStartNum=10, nLoops=100, nRandSamples = 100)

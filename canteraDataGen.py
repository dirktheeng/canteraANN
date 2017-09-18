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
    initialGasState += x.tolist() + [dt]
    initialGasState = np.asarray(initialGasState)
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
                 baseFile = 'CH4_BFER_Rates', ext='.npy', startNum=0,
                 num_steps = 1200000):
    count = startNum
    dirName = os.path.join('.', baseFile+'_Raw_Data')
    if not os.path.exists(dirName):
        os.makedirs(dirName)
        
    rTest = ct.IdealGasReactor(gas)
    nRxns = rTest.kinetics.n_reactions
    
    minOutputVals = np.ones(nRxns)*1e12
    maxOutputVals = np.ones(nRxns)*-1e12
    saveOutputArr = np.zeros((nRxns,2))
    minInputVals = np.zeros(m+2)
    maxInputVals = np.ones(m+2)
    minInputVals[:2] = [Tlow, Plow]
    maxInputVals[:2] = [Tlow+Trange, Plow+Prange]
    saveInputArr = np.zeros((m+2,2))
    saveInputArr[:,0] = minInputVals
    saveInputArr[:,1] = maxInputVals
    for ii in range(nLoops):
        print('starting loop: ', ii)
        try:
            generateRandomGas(Tlow, Trange, Plow, Prange)
            initialState, rates = runReactorForRates(num_steps = num_steps, dt = dt)
            fName = '-'.join([baseFile, str(count).rjust(10, '0')])
            fName += ext
            fName = os.path.join('.', baseFile+'_Raw_Data', fName)
            np.save(fName, rates)
            fName = '-'.join([baseFile+'_initState', str(count).rjust(10, '0')])
            fName += ext
            fName = os.path.join('.', baseFile+'_Raw_Data', fName)
            np.save(fName, initialState)
            count+=1
            minoutputv = np.min(rates, 0)
            minOutputVals = np.minimum(minOutputVals, minoutputv)
            maxoutputv = np.max(rates,0)
            maxOutputVals = np.maximum(maxOutputVals, maxoutputv)
        except Exception as e:
            print(e)
    saveOutputArr[:,0] = minOutputVals
    saveOutputArr[:,1] = maxOutputVals

#    saveInputArr[:,0] = minInputVals
#    saveInputArr[:,1] = maxInputVals
#    
    saveOutputRangeFile = os.path.join(dirName, baseFile + '_Output_Ranges'+ext)
    saveInputRangeFile = os.path.join(dirName, baseFile + '_Input_Ranges'+ext)
    np.save(saveOutputRangeFile, saveOutputArr)
    np.save(saveInputRangeFile, saveInputArr)
    
if __name__ == '__main__':
    generateDataForRates(300,1500, 100000, 500000, nLoops=5, baseFile='test')

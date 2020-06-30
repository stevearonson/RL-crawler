# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:20:28 2020

@author: steve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

df = pd.read_csv('crawlerLog.csv', converters={'State' : eval, 'Next State' : eval})
test_mode = df[df['LearningMode'] == 'Test']

#gdf = test_mode.groupby(['Epsilon', 'LearningRate', 'Discount', 'Episode'])['Reward'].mean().reset_index()
#sns.catplot(kind='point', x='Episode', y='Reward', col='LearningRate', row='Discount', hue='Epsilon', data=gdf, height=3)
#plt.show(block=False)

# compare forward and reverse learners
gdf = test_mode.groupby(['LearningRate', 'Direction', 'Discount', 'Episode'])['Reward'].mean().reset_index()
sns.catplot(kind='point', x='Episode', y='Reward', col='LearningRate', row='Discount', hue='Direction', data=gdf, height=3)
plt.show(block=False)

#plot the entire log file
fig, ax = plt.subplots(figsize=(12,8))
sns.scatterplot(x='Step', y='Reward', hue='LearningMode', style='Direction', data=df, ax=ax)
plt.show(block=False)

'''
    create boxplots of rewards by state,action pair

'''
fig, ax = plt.subplots(nrows=2, figsize=(12,10))
forSteps = df[df['Direction'] == 'forward']
if not forSteps.empty:
    forSteps.boxplot(column='Reward', by=['State', 'Action'], ax=ax[0])
    ax[0].tick_params(axis='x', labelrotation=90)
    plt.suptitle('')
    ax[0].set_title('Forward Reward Distributions by State/Action')

revSteps = df[df['Direction'] == 'reverse']
if not revSteps.empty:
    revSteps.boxplot(column='Reward', by=['State', 'Action'], ax=ax[1])
    ax[1].tick_params(axis='x', labelrotation=90)
    plt.suptitle('')
    ax[1].set_title('Reverse Reward Distributions by State/Action')

plt.tight_layout()


'''
    plot the resulting Value function for both forward and reverse learners
'''
# read in the trained Q value files
with open('qValues-forward.npy', 'rb') as f:
    qvaluesForward = np.load(f, allow_pickle=True).item()

with open('qValues-reverse.npy', 'rb') as f:
    qvaluesReverse = np.load(f, allow_pickle=True).item()

# seperate the forward and reverse steps from the log file
# forSteps = pd.DataFrame(df[df['Direction'] == 'forward']['State'].tolist())
forSteps = pd.DataFrame(test_mode[test_mode['Direction'] == 'forward']['State'].tolist())
if not forSteps.empty:
    forSteps.columns = ['arm', 'hand']
    
# revSteps = pd.DataFrame(df[df['Direction'] == 'reverse']['State'].tolist())
revSteps = pd.DataFrame(test_mode[test_mode['Direction'] == 'reverse']['State'].tolist())
if not revSteps.empty:
    revSteps.columns = ['arm', 'hand']
    

'''
    Plot the value function and optimal policy
    Playback the robots steps from the log file
    By default, this is following the optimal policy
'''

def plotQvalues(qValues, fig, ax):
    # extract and save the number of states
    if 'numStates' in qValues:
        numArmStates, numHandStates = qValues.pop('numStates')
    else:
        numArmStates, numHandStates = 9, 9
        
    df = pd.DataFrame.from_dict(qValues, orient='index')
    df = df.reset_index()
    
    df[['state', 'action']] = pd.DataFrame(df['index'].tolist())
    df[['arm', 'hand']] = pd.DataFrame(df['state'].tolist())
    
    df = df[['arm', 'hand', 'action', 0]]
    df = df.rename(columns={0 : 'Qvalue'})
    df = df.sort_values(by=['arm', 'hand', 'action'])
    
    # extract the value function from the Q values        
    V = df[df.groupby(['arm', 'hand'])['Qvalue'].transform(max) == df['Qvalue']]
    
    # 
    arrowDir = pd.DataFrame({'action' : ['arm-up', 'arm-down', 'hand-up', 'hand-down'],
                             'arrowX' : [0, 0, 0.3, -0.3],
                             'arrowY' : [0.3, -0.3, 0, 0]})
    V = pd.merge(V, arrowDir, on='action')
    
    Vimage = np.zeros((numArmStates, numHandStates))
    for _,row in V.iterrows():
        Vimage[int(row['arm']), int(row['hand'])] = row['Qvalue']
        
    ax.imshow(Vimage, cmap='Oranges', origin = 'lower')
    ax.quiver(V['hand'], V['arm'], V['arrowX'], V['arrowY'])


def updateState(row, sdf, stateCircle):
    '''
        animation support function
        updates the cirlce marker location
    '''
    stateCircle.set_xdata(sdf['hand'].iloc[row])
    stateCircle.set_ydata(sdf['arm'].iloc[row])
    return stateCircle,
    
def animateStates(qValues, steps, direction):
    '''
        make a grid of the value function
        overlay with the estimated optimal policy
        play back the steps as recorded in the log file
    '''
    fig, ax = plt.subplots(figsize=(12,6))
    plotQvalues(qValues, fig, ax)
    ax.set_title(direction + ' Value function and Policy')
    
    hc, = ax.plot(steps['hand'], steps['arm'], 
                  'o', mfc='none', mec='black', markersize=20)
    FuncAnimation(fig, updateState, frames=len(steps), fargs=(steps, hc), 
                  repeat=False, blit=True)
    plt.show(block=False)
    
if not forSteps.empty:
    animateStates(qvaluesForward, forSteps, 'Forward')
if not revSteps.empty:
    animateStates(qvaluesReverse, revSteps, 'Reverse')
plt.show()

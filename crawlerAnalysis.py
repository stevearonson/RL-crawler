# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:20:28 2020

@author: steve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('crawlerLog.csv')

gdf = df.groupby(['Epsilon', 'LearningRate', 'Discount', 'Episode'])['Reward'].mean().reset_index()
sns.catplot(kind='point', x='Episode', y='Reward', col='LearningRate', row='Discount', hue='Epsilon', data=gdf, height=3)
plt.show()

# compare forward and reverse learners
gdf = df.groupby(['Epsilon', 'Direction', 'Discount', 'Episode'])['Reward'].mean().reset_index()
sns.catplot(kind='point', x='Episode', y='Reward', col='Epsilon', row='Discount', hue='Direction', data=gdf, height=3)
plt.show()




'''
    plot the resulting Value function for both forward and reverse learners
'''

def plotQvalues(qValues, stateX, stateY, ax):
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
    
    Vimage = np.zeros((9,13))
    for _,row in V.iterrows():
        Vimage[int(row['arm']), int(row['hand'])] = row['Qvalue']
        
    ax.imshow(Vimage, cmap='Oranges', origin = 'lower')
    ax.quiver(V['hand'], V['arm'], V['arrowX'], V['arrowY'])
    

with open('qValues-forward.npy', 'rb') as f:
    qvaluesForward = np.load(f, allow_pickle=True).item()

with open('qValues-reverse.npy', 'rb') as f:
    qvaluesReverse = np.load(f, allow_pickle=True).item()

fig, ax = plt.subplots(1, 2, figsize=(12,6))
plotQvalues(qvaluesForward, 9, 13, ax[0])
ax[0].set_title('Forward Value function and Policy')

plotQvalues(qvaluesReverse, 9, 13, ax[1])
ax[1].set_title('Reverse Value function and Policy')
plt.show()

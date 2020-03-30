# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:47:12 2020

@author: steve
"""

import qlearningAgents
import myCrawler
import optparse

from sklearn.model_selection import ParameterGrid
import pandas as pd


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.8,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.8,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--trainIterations',action='store',
                         type='int',dest='trainIters',default=1000,
                         metavar="K", help='Interval of training steps (default %default)')
    optParser.add_option('-t', '--testIterations',action='store',
                         type='int',dest='testIters',default=100,
                         metavar="K", help='Interval of test steps (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=20,
                         metavar="K", help='Number of epsiodes to run (default %default)')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')

    opts, args = optParser.parse_args()
    return opts

def runEpisode(iters, robotEnvironment, learner, logEnable, episode, startStep, params):


    dl_list = []
    for i in range(iters):

        state = robotEnvironment.getCurrentState()
        actions = robotEnvironment.getPossibleActions(state)
        if len(actions) == 0.0:
            robotEnvironment.reset()
            state = robotEnvironment.getCurrentState()
            actions = robotEnvironment.getPossibleActions(state)
            print('Reset!')

        action = learner.getAction(state)
        if action == None:
            raise Exception('None action returned: Code Not Complete')
        nextState, reward = robotEnvironment.doAction(action)
        learner.observeTransition(state, action, nextState, reward)
            
        if logEnable:
            dl_list.append([eps,
                      startStep + i, 
                      params['Eps'], 
                      params['LR'], 
                      params['Disc'],
                      state,
                      action,
                      nextState,
                      reward])
    
    return dl_list



if __name__ == '__main__':

    opts = parseOptions()

    robot = myCrawler.CrawlingRobot(None)
    robotEnvironment = myCrawler.CrawlingRobotEnvironment(robot)

    actionFn = lambda state: robotEnvironment.getPossibleActions(state)

    param_grid = {
            'Eps' : [opts.epsilon],
            'LR': [0.2, 0.4, 0.6, 0.8],
            'Disc' : [0.2, 0.4, 0.6, 0.8]
            }

    grid = ParameterGrid(param_grid)
    data_log_list = []        

    for params in grid:
        # reset the robot and the learner
        robotEnvironment.reset()
        learner = qlearningAgents.QLearningAgent(actionFn=actionFn)
        
        stepCount = 0
    
        for eps in range(1, opts.episodes+1):
            # run a learning episode    
            learner.setEpsilon(params['Eps'])
            learner.setLearningRate(params['LR'])
            learner.setDiscount(params['Disc'])
        
            learner.startEpisode()
            runEpisode(opts.trainIters, robotEnvironment, learner, False, eps, stepCount, params)
            stepCount += opts.trainIters
            learner.stopEpisode()
            
            learner.setEpsilon(0.0)
            learner.setLearningRate(0.0)
        
            temp = runEpisode(opts.testIters, robotEnvironment, learner, True, eps, stepCount, params)
            stepCount += opts.testIters
            data_log_list.extend(temp)

    # convert data log into Pandas DataFrame and display
    cols = ['Episode', 'Step', 'Epsilon', 'LearningRate', 'Discount', 'State', 'Action', 'Next State', 'Reward']
    df = pd.DataFrame(data_log_list, columns=cols)
    print(df)

# -*- coding: utf-8 -*-
"""
Robot Crawler Application

Execute learning on a robot crawler by combining the robot, its environment,
and a learning agent. Each learning cycle is a series of episodes, where each
each episode contains:
    Learning steps
    Testing steps


This script accepts setting hyper parameters via the command line including:
    Learning length controls:
        Number of episodes
        Number of learning steps per episode
        Number of testing steps per episode
    Learning hyper parameters:
        Learning Rate
        Discount
        Probability of random action (epsilon)
        
Hyperparameters can be arrays, in which case a learning cycle will be executed
for all combinations of parameters. The results will be plotted in a grid 
for comparisons
        
Key Python dependencies:
    sklearn
    pandas
    matplotlib
    seaborn
    
Created on Mon Mar 30 11:47:12 2020

@author: steve
"""

import qlearningAgents
import myCrawler
import argparse

from sklearn.model_selection import ParameterGrid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parseOptions():
    """ creates a parser object for storing command line arguments 
    
    Parameters
    ----------
    None
        
    Returns
    -------
    args : Namespace object
        dict like list of command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--discount', action='store', nargs='+',
                         type=float, dest='discount', default=[0.1, 0.5, 0.9],
                         help='Discount on future (default  %(default)s)')
    parser.add_argument('-n', '--noise', action='store',
                         type=float, dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default  %(default)s' )
    parser.add_argument('-e', '--epsilon', action='store', nargs='+',
                         type=float, dest='epsilon', default=[0.2, 0.4, 0.6, 0.8],
                         metavar="E", help='Chance of taking a random action in q-learning (default  %(default)s')
    parser.add_argument('-l', '--learningRate', action='store', nargs='+',
                         type=float, dest='learningRate', default=[0.2, 0.4, 0.6, 0.8],
                         metavar="P", help='TD learning rate (default  %(default)s' )
    parser.add_argument('-p', '--planningSteps', action='store', nargs='+',
                         type=int, dest='planningSteps', default=[0],
                         metavar="P", help='Dyna-Q planning steps (default  %(default)s' )
    parser.add_argument('-i', '--trainIterations', action='store',
                         type=int, dest='trainIters', default=1000,
                         metavar="K", help='Interval of training steps (default  %(default)s')
    parser.add_argument('-t', '--testIterations', action='store',
                         type=int, dest='testIters', default=100,
                         metavar="K", help='Interval of test steps (default  %(default)s')
    parser.add_argument('-k', '--episodes', action='store',
                         type=int, dest='episodes', default=20,
                         metavar="K", help='Number of epsiodes to run (default  %(default)s')
    parser.add_argument('-q', '--quiet', action='store_true',
                         dest='quiet', default=False,
                         help='Skip display of any learning episodes')

    args = parser.parse_args()
    return args

def runEpisode(iters, robotEnvironment, learner, logEnable, episode, startStep, params):
    """ executes learning iterations on robot by applying actions and 
    tracking next state and reward 
    
    Parameters
    ----------
    iters : int
        number of actions iterations to perform
    robotEnvironment : class CrawlingRobotEnvironment
        initialized robot environment for actions
    learner : class ReinforcementAgent
        initialized class for tracking value function for all states
    logEnable : boolean
        enables logging of each action step
    episode : int
        count of current learning cycle
    startStep : int
        current total step count
    params : class ParameterGrid
        cointains current hyper parameters for this learning cycle
        
    Returns
    -------
    data_log_list : list
        array of lists containing hyperparameters and state action values for 
        each step. Empty list returned when logEnable=False
    """

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
                      params['PSteps'],
                      state,
                      action,
                      nextState,
                      reward])
    
    return dl_list



if __name__ == '__main__':

    opts = parseOptions()

    robot = myCrawler.CrawlingRobot()
    robotEnvironment = myCrawler.CrawlingRobotEnvironment(robot)

    actionFn = lambda state: robotEnvironment.getPossibleActions(state)

    param_grid = {
            'Eps' : opts.epsilon,
            'LR': opts.learningRate,
            'Disc' : opts.discount,
            'PSteps' : opts.planningSteps
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
            learner.setPlanningSteps(params['PSteps'])
        
            learner.startEpisode()
            runEpisode(opts.trainIters, robotEnvironment, learner, False, eps, stepCount, params)
            stepCount += opts.trainIters
            learner.stopEpisode()
 
            # halt learning and measure best velocity using current value function           
            learner.setEpsilon(0.0)
            learner.setLearningRate(0.0)
            learner.setPlanningSteps(0)
        
            temp = runEpisode(opts.testIters, robotEnvironment, learner, True, eps, stepCount, params)
            stepCount += opts.testIters
            data_log_list.extend(temp)

    # convert data log into Pandas DataFrame and display
    cols = ['Episode', 'Step', 'Epsilon', 'LearningRate', 'Discount',
            'Planning Steps', 'State', 'Action', 'Next State', 'Reward']
    df = pd.DataFrame(data_log_list, columns=cols)
    
    gdf = df.groupby(['Epsilon', 'LearningRate', 'Discount', 'Episode'])['Reward'].mean().reset_index()
    sns.catplot(kind='point', x='Episode', y='Reward', col='LearningRate', row='Discount', hue='Epsilon', data=gdf, height=3)
    plt.show()

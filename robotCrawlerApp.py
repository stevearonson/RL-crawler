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
import random


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
                         type=float, dest='discount', default=[0.9],
                         help='Discount on future (default  %(default)s)')
    parser.add_argument('-n', '--noise', action='store', nargs='+',
                         type=float, dest='noise', default=[0.0],
                         metavar="P", help='Rewards additive noise standard deviation ' +
                         '(default  %(default)s' )
    parser.add_argument('-e', '--epsilon', action='store', nargs='+',
                         type=float, dest='epsilon', default=[0.8],
                         metavar="E", help='Chance of taking a random action in q-learning (default  %(default)s')
    parser.add_argument('-l', '--learningRate', action='store', nargs='+',
                         type=float, dest='learningRate', default=[0.8],
                         metavar="P", help='TD learning rate (default  %(default)s' )
    parser.add_argument('-p', '--planningSteps', action='store', nargs='+',
                         type=int, dest='planningSteps', default=[0],
                         metavar="P", help='Dyna-Q planning steps (default  %(default)s' )
    parser.add_argument('-i', '--trainIterations', action='store',
                         type=int, dest='trainIters', default=200,
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
    parser.add_argument('-r', '--robot', action='store_true',
                         dest='useRobot', default=False,
                         help='Use real robot hardware vs. sim')
    parser.add_argument('-s', '--states', action='store', nargs='+',
                         type=int, dest='states', default=[5, 5],
                         metavar="K", help='Number of states for each servo (default  %(default)s')
    parser.add_argument('-sl', '--saveLog', action='store',
                         dest='saveLog', default='crawlerLog.csv',
                         help='Destination for saving log file')
    parser.add_argument('-sq', '--saveQvalues', action='store',
                         dest='saveQvalues', default='qValues',
                         help='Base file name for saving Q value matrices')
    parser.add_argument('-lq', '--loadQvalues', action='store',
                         dest='loadQvalues', default='',
                         help='Base file name for loading Q value matrices before learning')
    parser.add_argument('-ra', '--rewardAvg', action='store_true',
                         dest='rewardAvg', default=False,
                         help='Use average rewards vs last value')
    args = parser.parse_args()
    return args

class CrawlerRobot:
    
    def __init__(self, useRobot, numStates):
        
        if useRobot:
            self.robot = myCrawler.CrawlingRobotGene()
        else:
            self.robot = myCrawler.CrawlingRobot()
            
        self.robotEnvironment = myCrawler.CrawlingRobotEnvironment(self.robot, numStates)

        self.actionFn = lambda state: self.robotEnvironment.getPossibleActions(state)
        
        self.learner = {}
        
        self.direction = 'forward'
        
        
    def runEpisode(self, iters, logEnable, episode, startStep, params):
        
        """ executes learning iterations on robot by applying actions and 
        tracking next state and reward 
        
        Parameters
        ----------
        iters : int
            number of actions iterations to perform
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
    
            state = self.robotEnvironment.getCurrentState()
            actions = self.robotEnvironment.getPossibleActions(state)
            if len(actions) == 0.0:
                self.robotEnvironment.reset()
                state = self.robotEnvironment.getCurrentState()
                actions = self.robotEnvironment.getPossibleActions(state)
                print('Reset!')
    
    
            action = self.learner[self.direction].getAction(state)
            if action == None:
                raise Exception('None action returned: Code Not Complete')
    
            nextState, reward, railFlags = self.robotEnvironment.doAction(action)
            
            # invert the reward for the reverse direction
            if self.direction == 'reverse':
                reward = -reward
                
            reward = reward + random.normalvariate(0, params['Noise'])
            
            self.learner[self.direction].observeTransition(state, action, nextState, reward)
            
            if ((startStep + i) % 100) == 0:
                print(episode,
                       startStep + i, 
                       logEnable,
                       params['Eps'], 
                       params['LR'], 
                       params['Disc'],
                       params['PSteps'],
                       params['Noise'],
                       self.direction,
                       state,
                       action,
                       nextState,
                       reward)
        
                
            if logEnable:
                dl_list.append([episode,
                          startStep + i, 
                          logEnable,
                          params['Eps'], 
                          params['LR'], 
                          params['Disc'],
                          params['PSteps'],
                          params['Noise'],
                          self.direction,
                          state,
                          action,
                          nextState,
                          reward])
        
            """
                check if we need to switch direction
            """
            if (self.direction == 'forward') and railFlags['Max']:
                self.direction = 'reverse'
            elif (self.direction == 'reverse') and railFlags['Min']:
                self.direction = 'forward'
        
        return dl_list
    
        
    def learningCycle(self, opts, params):
        '''
            a learning cycle is a series of episodes using the same training
            parameters. A learning cycle always starts with a new or loaded 
            learner. The cycle loops through the episodes, toggling between
            training mode and test mode
            
            Parameters
            ----------
            opts : argspace.Namespace
                Command line arguments
            params : class ParameterGrid
                cointains current hyper parameters for this learning cycle
                
            Returns
            -------
            data_log_list : list
                array of lists containing hyperparameters and state action values for 
                each step. Empty list returned when logEnable=False
        '''

        self.robotEnvironment.reset()
        self.direction = 'forward'
        
        self.learner = {
            'forward' : qlearningAgents.QLearningAgent(actionFn=self.actionFn),
            'reverse' : qlearningAgents.QLearningAgent(actionFn=self.actionFn)
        }
        self.learner['forward'].setNumStates(opts.states)
        self.learner['reverse'].setNumStates(opts.states)
        
        '''
            initialize the q values from saved files if requested
            make sure that current number of states matches with
            value in saved file
        '''
        if opts.loadQvalues:
            self.loadQValues(opts.loadQvalues)
            if self.learner['forward'].getNumStates() != opts.states:
                print('Specified number of states %d,%d does not match loaded qValues %d,%d' % 
                      (tuple(opts.states) +
                       tuple(self.learner['forward'].getNumStates())))
                raise SystemExit
        
        stepCount = 0
        data_log_list = []        
    
        for eps in range(1, opts.episodes+1):
            # run a learning episode    
            self.learner['forward'].setEpsilon(params['Eps'])
            self.learner['forward'].setLearningRate(params['LR'])
            self.learner['forward'].setDiscount(params['Disc'])
            self.learner['forward'].setPlanningSteps(params['PSteps'])
            self.learner['forward'].setUseRewardAvg(opts.rewardAvg)
        
            self.learner['reverse'].setEpsilon(params['Eps'])
            self.learner['reverse'].setLearningRate(params['LR'])
            self.learner['reverse'].setDiscount(params['Disc'])
            self.learner['reverse'].setPlanningSteps(params['PSteps'])
            self.learner['reverse'].setUseRewardAvg(opts.rewardAvg)
        
            self.learner['forward'].startEpisode()
            self.learner['reverse'].startEpisode()

            temp = self.runEpisode(opts.trainIters, 'Train', eps, stepCount, params)
            data_log_list.extend(temp)
            stepCount += opts.trainIters

            self.learner['forward'].stopEpisode()
            self.learner['reverse'].stopEpisode()
 
            # halt learning and measure best velocity using current value function           
            self.learner['forward'].setEpsilon(0.0)
            self.learner['forward'].setLearningRate(0.0)
            self.learner['forward'].setPlanningSteps(0)
        
            self.learner['reverse'].setEpsilon(0.0)
            self.learner['reverse'].setLearningRate(0.0)
            self.learner['reverse'].setPlanningSteps(0)
        
            temp = self.runEpisode(opts.testIters, 'Test', eps, stepCount, params)
            stepCount += opts.testIters
            data_log_list.extend(temp)
            
        return data_log_list
    
    
    def saveQValues(self, QVbaseName):
        self.learner['forward'].saveQvalues(QVbaseName + '-forward.npy')
        self.learner['reverse'].saveQvalues(QVbaseName + '-reverse.npy')
        

    def loadQValues(self, QVbaseName):
        self.learner['forward'].loadQvalues(QVbaseName + '-forward.npy')
        self.learner['reverse'].loadQvalues(QVbaseName + '-reverse.npy')
        


if __name__ == '__main__':

    opts = parseOptions()

    param_grid = {
            'Eps' : opts.epsilon,
            'LR': opts.learningRate,
            'Disc' : opts.discount,
            'PSteps' : opts.planningSteps,
            'Noise' : opts.noise
            }

    grid = ParameterGrid(param_grid)
    data_log_list = []        
    
    crawlerRobot = CrawlerRobot(opts.useRobot, opts.states)
    
    for params in grid:
        
        temp = crawlerRobot.learningCycle(opts, params)
        data_log_list.extend(temp)

    # convert data log into Pandas DataFrame and display
    cols = ['Episode', 'Step', 'LearningMode', 'Epsilon', 'LearningRate', 
            'Discount', 'Planning Steps', 'Noise', 'Direction', 'State', 
            'Action', 'Next State', 'Reward']
    df = pd.DataFrame(data_log_list, columns=cols)
    
    if opts.saveLog:
        df.to_csv(opts.saveLog, index=False)
        
    if opts.saveQvalues:
        crawlerRobot.saveQValues(opts.saveQvalues)
    

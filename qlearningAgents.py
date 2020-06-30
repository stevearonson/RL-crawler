# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# from game import *
from learningAgents import ReinforcementAgent
# from featureExtractors import *

import random,util
import numpy as np

class StateModel():
    '''
        state space reprensetation of the robot
        for each state, action pair it stores the next state, sum(reward),
        and count of times this state, action pair was visited
        To reduce variance in the movement, the model returns the running
        average of the reward, rather than the last value
    '''
    def __init__(self):
        self.model = {}


    def setModel(self, state, action, next_state, reward):
        if (state, action) in self.model:
            _, rewardSum, count = self.model[(state, action)]
            rewardSum += reward
            count += 1
            self.model[(state, action)] = (next_state, rewardSum, count)
            
        else:
            self.model[(state, action)] = (next_state, reward, 1)


    def getModel(self, state, action):
        next_state, rewardSum, count = self.model[(state, action)]
        return (next_state, rewardSum/count)

    
    def getModelKeys(self):
        return list(self.model)


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        """
            store QValues as a dictionary using the state,action pair as the key
        """
        self.qvalues = {}
        
        """
            Store model of all learned state and actions
        """
        self.model = StateModel()
        
        '''
            command line arguments initialized here
            updated using class functions
        '''
        self.planningSteps = 0
        self.useRewardAvg = False

    '''
        The following functions set/get learning parameters
    '''
    def setPlanningSteps(self, steps):
        self.planningSteps = steps

        
    def setUseRewardAvg(self, useRewardAvg):
        self.useRewardAvg = useRewardAvg

        
    def getNumStates(self):
        return self.qvalues['numStates']


    def setNumStates(self, numStates):
        self.qvalues['numStates'] = numStates


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state,action) in self.qvalues:
            return self.qvalues[(state,action)]
        else:
            return 0.0
        
        
    def setQValue(self, state, action, value):
        self.qvalues[(state, action)] = value
        
        

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        
        max_value = None
        for action in actions:
            
            value = self.getQValue(state, action)
            # update best action and max value
            if max_value is None or value > max_value:
                max_value = value

        return max_value


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if not actions:
            return None

        max_value = self.computeValueFromQValues(state)
        best_actions = []
        for action in actions:
            value = self.getQValue(state, action)
            # update best action and max value
            if value >= max_value:
                best_actions.append(action)

        return random.choice(best_actions)


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)

        return action
    
    
    def runPlan(self, planningSteps):
        """
            run planning steps
            select a previously visited random state/action
            update Q values from the model
        """
        disc = self.discount
        alpha = self.alpha

        for step in range(planningSteps):
            (mState, mAction) = random.choice(self.model.getModelKeys())
            (mNextState, mReward) = self.model.getModel(mState, mAction)

            qvalue = self.getQValue(mState, mAction)
            next_value = self.getValue(mNextState)
        
            new_value = (1 - alpha) * qvalue + alpha * (mReward + disc * next_value)
            self.setQValue(mState, mAction, new_value)

        

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        disc = self.discount
        alpha = self.alpha
        qvalue = self.getQValue(state, action)
        next_value = self.getValue(nextState)
        
        """ update the model using this transition """
        self.model.setModel(state, action, nextState, reward)
        if self.useRewardAvg:
            _, reward = self.model.getModel(state, action)

        new_value = (1 - alpha) * qvalue + alpha * (reward + disc * next_value)
        self.setQValue(state, action, new_value)
        
        
        """ planning steps """
        if (self.planningSteps > 0):
            self.runPlan(self.planningSteps)


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)


    def getValue(self, state):
        return self.computeValueFromQValues(state)

    '''
        The next 2 functions save/load the internal Q values for
        1. offline analysis
        2. continuing learning from previous run
        3. demonstrate a pretrained setof values
    '''    
    def saveQvalues(self, qValueFileName):
        
        with open(qValueFileName, 'wb') as f:
            np.save(f, self.qvalues)


    def loadQvalues(self, qValueFileName):

        with open(qValueFileName, 'rb') as f:
            self.qvalues = np.load(f, allow_pickle=True).item()



class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)

        result = 0
        for feature in features:
            result += self.weights[feature] * features[feature]
        return result

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        disc = self.discount
        alpha = self.alpha
        qvalue = self.getQValue(state, action)
        next_value = self.getValue(nextState)

        features = self.featExtractor.getFeatures(state, action)
        
        diff = (reward + disc*next_value) - qvalue
     
        for feature in features:
            self.weights[feature] += alpha*diff*features[feature]


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

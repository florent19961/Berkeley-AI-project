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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

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
        # dans ce dictionnaire, on aura comme clé pour chaque élément "state, action" et en valeurs les Q-values
        self.Q_values = util.Counter()
        
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # on veut retourner la Q_value correspondant à la paire (state, action). 
        return(self.Q_values[(state, action)])
        util.raiseNotDefined()
               


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Ici on veut créer la grille des valeurs, donc le code consistera à appeler l'ensemble des actions possibles depuis l'état
        #
        actions = self.getLegalActions(state)
        Q_values = []
        if len(actions) == 0:
            return 0
        

        for action in actions:
            Q_values.append(self.getQValue(state, action))
        return(max(Q_values))
                            
                            
            
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        Q_values = []
        if actions == []:
            return None
        for action in actions:
            Q_values.append(self.getQValue(state, action))
        index_action = Q_values.index(max(Q_values))
        
        return actions[index_action] 
            
        
        util.raiseNotDefined()

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
        follow_best_action = util.flipCoin(1 - self.epsilon)
        if len(legalActions) == 0:
            return None
        elif follow_best_action:
            return self.computeActionFromQValues(state)
        else:
            # Dans l'exploration ca veut dire qu'on s'autoriserait a prendre 
            # la meilleure action de maniere aleatoire => pour moi on l'excluait 
            
            #list_legalActions = list(legalActions)
            #list_legalActions.remove(self.computeActionFromQValues(state))
            return random.choice(legalActions)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
        # Btw le crawler lache directement un bug mais celui ci semble venir du code du crawler en lui meme 
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        val_to_update = self.Q_values[(state, action)]
        #print(val_to_update, 'aaaaaaa')
        val_to_update = (1 - self.alpha) * val_to_update + self.alpha * (reward + self.discount * self.getValue(nextState))
        #print(val_to_update, 'bbbbb')
        self.Q_values[(state, action)] = val_to_update
                      
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


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
        weights = self.getWeights()
        #print(len(weights), 'longueur des poids')
        return weights * features

        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # difference = reward + gamma*Q(s', a') - Q(s,a)
        difference = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        weights = self.getWeights()
        # si il est vide on l'initialise à zero
        if len(weights) == 0:
            weights[(state,action)] = 0
        # on récupère les features
        features = self.featExtractor.getFeatures(state, action)
        
        # on actualise nos features avec la difference et le learning rate
        for key in features.keys():
            features[key] = features[key] * self.alpha * difference
            
        # il faut maintenant sommer feature et poids pour avoir les nouveaux poids
        # donc on effectue cette somme en se servant de la méthode __radd__
        weights.__radd__(features)
        
        # et on actualise nos poids
        self.weights = weights.copy()
        
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

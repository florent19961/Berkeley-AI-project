# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        
        
            

    def runValueIteration(self):
        # Write value iteration code here
        # on va faire une boucle for 
        # quand on sort de cette boucle le but c'est de renvoyer le tableau des value iteration 'quelle forme ce tableau' ?
        
        for i in range(self.iterations):
            # on doit créer ceci car si on garde juste self.values, au moment des itérations sur les q states on reprendrait les
            # valeurs deja actualisées pour faire les calculs => on veut pas faire ca. On veut actualiser sur t pour ressortir 
            # une grille t+1 whole_values permet d'hold les actualisations de t+1 pour ensuite les renvoyer dans self.values
            whole_values = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    value_state = util.Counter()
                    for action in self.mdp.getPossibleActions(state):
                        value_state[action] = self.computeQValueFromValues(state, action)
                    whole_values[state] = max(value_state.values())
            self.values = whole_values.copy()
            
            
            

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        
        
        next_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        # liste des recompenses R(s,a,s')
        rewards = []
        # liste des probas de transitions P(s'|a,s)
        probs = []
        # liste des Vk(s')
        previous_values = []
        # occurence[0] = les next_state
        # occurence[1] = les proba de transi
        for occurence in next_states_probs:
            rewards.append(self.mdp.getReward(state, action, occurence[0]))
            probs.append(occurence[1])
            previous_values.append(self.getValue(occurence[0]))
        Q_value = 0
        #  boucle qui calcule somme des ( P(s'|a,s) * [R(s,a,s') + gamma * Vk(s')] ) sur les s'
        for i in range(len(probs)):
            Q_value += probs[i] * (rewards[i] + self.discount * previous_values[i])
                                  
        return Q_value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
      
        actions = self.mdp.getPossibleActions(state)
       
        if len(actions) == 0:
            return None
        
        # compteur des q_values
        values = util.Counter()
        
        for action in actions:
            values[action] = self.computeQValueFromValues(state, action)
        
        return values.argMax()
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        
        states = self.mdp.getStates()
        # un update = un state
        # boucle va de 0 a iterations - 1
        for i in range(self.iterations):
            # reste de la division euclidienne est le state
            state = states[i % len(states)]
            # si mdp pas terminal alors on fait l'update
            if not self.mdp.isTerminal(state):
                value_state = util.Counter()
                # pour les actions possibles depuis le state a updater
                for action in self.mdp.getPossibleActions(state):
                    # on calcule le Q_value
                    value_state[action] = self.computeQValueFromValues(state, action)
                # on prend max des Q_value pour faire l'update du state
                self.values[state] = max(value_state.values())
        # ici la copie n'est pas nécessaire car on définit le whole_values avant la boucle des itérations
        # self.values = whole_values.copy()
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = {}
        def max_Q_value(state):
            Q_values = []
            for action in self.mdp.getPossibleActions(state):
                Q_values.append(self.getQValue(state, action))
            return(max(Q_values))
        # quelle structure adopter pour stocker les prédecesseurs de chaque state ?
        # tout dépend de comment on compte trouver les prédecesseurs a proba non nulle.
        # etant donné la contrainte sur la proba, l'emploi de mdp.getTransitionStatesAndProbs(state, action) qui renvoie
        # une liste de (nextState, prob) parait plus que pertinent.
        # donc on pourrait itérer sur chacun des états, leur appliquer cette fonction et a chaque etat on va voir le nextState grace
        # a la fonction donc on sait qu'on pourra rajouter le state a la liste des predecesseurs du nextState (moyennant une contraine
        # sur la proba). 
        # ainsi la structure du code serait la suivante
        # boucle for sur tous les etats dans cette boucle une boucle for sur toutes les actions possibles depuis cet etat
        # dans cette boucle on applique mdp.getTransitionStatesAndProbas(state, action)
        # on récupère la liste des next state et probas
        # si la proba > 0 alors on rajoute au nextState le state courant
        # etant donné la redondance qu'il y aura dans ce code, l'emploi du set est en effet utile
        # donc on va créer d'abord un dico avec comme clé chacun des etats et en valeurs le set() qui sera modifié dans les boucles for
        
        # initialisation du dictionnaire qui contiendra les predecesseurs
        for state in states:
            predecessors[state] = set()
        # pour chaque etat on trouve les predecesseurs
        for state in states:
            # pas oublier de tester l'etat terminal, vu la manière dont notre code est construit la suite renverra des erreurs si terminal
            if not self.mdp.isTerminal(state):
            # pour chaque action on trouve les successeurs
                for action in self.mdp.getPossibleActions(state):
                    # les transitions possibles
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            predecessors[nextState].add(state)
        # là on a un dictionnaire dont les clés sont les etats, et les valeurs, un set contenant l'ensemble des etats prédecesseurs.
        
        # l'étape suivante de l'algorithme est de trouver la valeur absolue entre V(s) (stocké dans self.values) et la plus grande
        # de Q-value sur toutes les actions possibles depuis s et de pousser l'inverse de cette différence dans la queue
        queue = util.PriorityQueue()
        
        for state in states:
            if not self.mdp.isTerminal(state):
                
                diff = abs(self.values[state] - max_Q_value(state))
                
                queue.push(state, - diff)
        
            
        for iteration in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
                    
            self.values[state] = max_Q_value(state)
            
            for predecessor in predecessors[state]:
                if not self.mdp.isTerminal(predecessor):
                    diff = abs(self.values[predecessor] - max_Q_value(predecessor))
                    if diff > self.theta:
                        queue.update(predecessor, -diff) 
                
        
                        
            


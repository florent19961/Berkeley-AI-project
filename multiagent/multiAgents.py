# multiAgents.py
# --------------
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
import pdb; 

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        # Sous la forme ['West', 'Stop', ... ] Toujours dans le même ordre d'apparition les legalmoves
        
        legalMoves = gameState.getLegalActions()
        # pour que ca aille plus vite
        if 'Stop' in legalMoves:
            legalMoves.remove('Stop')
        # Choose one of the best actions
        # le choix de notre action se fait sur la base de self.evaluationFunction D'ou la nécessité de travailler sur cette fonction
        # Liste de longueur au max 4 ? (ou 5 si on  peut se stopper ? A checker // C'est checké, longueur 5 max)
        
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print(scores)
        # envoie le score max
        bestScore = max(scores)
        # donne les indices des actions menant au bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # On choisit un parmi ces meilleurs indices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # Ici est renvoyé une liste de tuple [(x,y)]
        
        currentGhostPos = currentGameState.getGhostPositions()
        # donne la grille de jeu pour toutes les actions légales % sont des murs . la bouffe  G ghost  < > v ^ la directio du pacman
        # ainsi que le score associé au move
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        scores = successorGameState.getScore()
        # print(successorGameState)
        # donne la position du pacman au temps t+1 pour chacun des legalMoves sous la forme d'un tuple (x, y)
        newPos = successorGameState.getPacmanPosition()
        

        #print(newPos)
        # donne la carte (objet Grid, qu'on peut appeler comme ca newFood[x][y] à t+1 pour chacun des legal moves T : food / F : no food
        foodGrid = successorGameState.getFood()
        #print(newFood)
        # renvoie un objet fantome
        newGhostStates = successorGameState.getGhostStates()
        # renvoie la liste des timer des ghost quand ils sont effrayés
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # boucle avec contrainte sur les fantômes (si ils sont trop proches on se taille) à moins d'être en état où on les mange
        for i, ghost in enumerate(currentGhostPos):
            if util.manhattanDistance(newPos, ghost) < 2 and newScaredTimes[i] == 0:
                scores -= 60
            if util.manhattanDistance(newPos, ghost) < 10 and newScaredTimes[i] != 0:
                scores += 20*(15 - util.manhattanDistance(newPos, ghost))
        # maintenant boucle avec la bouffe
        distances = []
        foodList = foodGrid.asList()
        # cette boucle est limitée à cause de la manhattandistance qui prend pas en compte les murs on ne s'attardera pas la dessus
        # meme si quelque chose pourrait etre fait pour améliorer ce défaut
        for x,y in foodList:
            distances.append(util.manhattanDistance((x,y), newPos))
        if distances != []:
            closest = min(distances)
            scores += (40 - closest) / 10 # si on divise pas par 10 ca va jamais manger une food isolée car score(eat) < score(stay_around)
        #print(newScaredTimes)
        
        return scores

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """
    
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        
        def max_value(gameState, depth, num_ghosts):
            # tour pacman
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            max_score = -(float("inf"))
            actions = gameState.getLegalActions(0)
            for action in actions:
                max_score = max(max_score, min_value(gameState.generateSuccessor(0, action), depth, 1, num_ghosts))
            return max_score
        def min_value(gameState, depth, agentIndex, numghosts):
            # tour ghost
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            elif agentIndex != num_ghosts:
                min_score = +(float("inf"))
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:
                    min_score = min(min_score, min_value(gameState.generateSuccessor(agentIndex, action), 
                                                         depth, agentIndex + 1, num_ghosts))
                return min_score
            elif agentIndex == num_ghosts:
                min_score = +(float("inf"))
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:
                    min_score = min(min_score, max_value(gameState.generateSuccessor(agentIndex, action), depth - 1, num_ghosts))
                return min_score
        num_ghosts = gameState.getNumAgents() - 1
        actions = gameState.getLegalActions(0)
        best_action = Directions.STOP
        best_score = -(float("inf"))
        
        for action in actions:
            prev_score = best_score
            best_score = max(prev_score, min_value(gameState.generateSuccessor(0, action), self.depth, 1, num_ghosts))
            if best_score > prev_score:
                best_action = action
        return best_action
    
                             
            
        util.raiseNotDefined()

   
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        
        def max_value(gameState, depth, num_ghosts, alpha, beta):
            # tour pacman
            # si etat terminal on appelle l'évaluation
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            # si état autre on lance la recherche (type dfs, appel récursif)
            max_score = -(float("inf"))
            actions = gameState.getLegalActions(0)
            for action in actions:
                max_score = max(max_score, min_value(gameState.generateSuccessor(0, action), depth, 1, num_ghosts, alpha, beta))
                # pruning, sur un etage max si valeur retournée > beta alors on doit élaguer 
                if max_score > beta:
                    return beta
                # update alpha
                alpha = max(max_score, alpha)
            return max_score
        
        def min_value(gameState, depth, agentIndex, numghosts, alpha, beta):
            # tour ghost
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            elif agentIndex != num_ghosts:
                min_score = +(float("inf"))
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:
                    min_score = min(min_score, min_value(gameState.generateSuccessor(agentIndex, action), 
                                                         depth, agentIndex + 1, num_ghosts, alpha, beta))
                    # pruning, sur un etage min si valeur retournée < alpha alors on doit élaguer 
                    if min_score < alpha:
                        return alpha
                    # update beta
                    beta = min(min_score, beta)
                return min_score
            
            elif agentIndex == num_ghosts:
                min_score = +(float("inf"))
                actions = gameState.getLegalActions(agentIndex)
                for action in actions:
                    min_score = min(min_score, max_value(gameState.generateSuccessor(agentIndex, action), 
                                                         depth - 1, num_ghosts, alpha, beta))
                    if min_score < alpha:
                        return alpha
                    beta = min(min_score, beta)
                return min_score
            
        num_ghosts = gameState.getNumAgents() - 1
        actions = gameState.getLegalActions(0)
        best_action = Directions.STOP
        best_score = -(float("inf"))
        alpha = -(float("inf"))
        beta = +(float("inf"))
        for action in actions:
            prev_score = best_score
            best_score = max(prev_score, min_value(gameState.generateSuccessor(0, action), self.depth, 1, num_ghosts, alpha, beta))
            if best_score > prev_score:
                best_action = action
            # comme dans les boucles dans les fonction min et max value, on se doit de faire le pruning si on est en haut de l'arbre
            if best_score > beta:
                return best_action
                # mise a jour de alpha
            alpha = max(best_score, alpha)
            
        return best_action
    
                             
            
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    # pour celui ci il est nécessaire de modifier plusieurs choses. On doit toujours suivre un algorithme un peu type minmax. 
    # Cependant, il faut maintenantprendre en compte le fait que les fantomes bougent de manière aléatoire. Pour cela on va devoir
    # modifier les values 
    def getAction(self, gameState):
        
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth, num_ghosts):
            # tour pacman
            # si etat terminal on appelle l'évaluation
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            # si état autre on lance la recherche (type dfs, appel récursif)
            max_score = -(float("inf"))
            actions = gameState.getLegalActions(0)
            
            for action in actions:
                max_score = max(max_score, chance_value(gameState.generateSuccessor(0, action), depth, 1, num_ghosts))
            return max_score
        
        def chance_value(gameState, depth, agentIndex, numghosts):
            # tour ghost
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            elif agentIndex != num_ghosts:
                chance_score = 0
                actions = gameState.getLegalActions(agentIndex)
                nb_actions = len(actions)
                for action in actions:
                    chance_score += chance_value(gameState.generateSuccessor(agentIndex, action),depth, agentIndex + 1, num_ghosts)
                return chance_score / nb_actions
            
            elif agentIndex == num_ghosts:
                chance_score = 0
                actions = gameState.getLegalActions(agentIndex)
                nb_actions = len(actions)
                for action in actions:
                    chance_score += max_value(gameState.generateSuccessor(agentIndex, action), depth - 1, num_ghosts)
                return chance_score / nb_actions
            
        num_ghosts = gameState.getNumAgents() - 1
        actions = gameState.getLegalActions(0)
        best_action = Directions.STOP
        best_score = -(float("inf"))
        
        for action in actions:
            prev_score = best_score
            best_score = max(prev_score, chance_value(gameState.generateSuccessor(0, action), self.depth, 1, num_ghosts))
            if best_score > prev_score:
                best_action = action
        return best_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: On a réutilisé la fonction d'évaluation de l'agent réflex.
    """
    "*** YOUR CODE HERE ***"
    
    
    currentGhostPos = currentGameState.getGhostPositions()
    # donne la grille de jeu pour toutes les actions légales % sont des murs . la bouffe  G ghost  < > v ^ la directio du pacman
    # ainsi que le score associé au move
    scores = currentGameState.getScore()
    # print(successorGameState)
    # donne la position du pacman au temps t+1 pour chacun des legalMoves sous la forme d'un tuple (x, y)
    newPos = currentGameState.getPacmanPosition()


    #print(newPos)
    # donne la carte (objet Grid, qu'on peut appeler comme ca newFood[x][y] à t+1 pour chacun des legal moves T : food / F : no food
    foodGrid = currentGameState.getFood()
    #print(newFood)
    # renvoie un objet fantome
    newGhostStates = currentGameState.getGhostStates()
    # renvoie la liste des timer des ghost quand ils sont effrayés
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # boucle avec contrainte sur les fantômes (si ils sont trop proches on se taille) à moins d'être en état où on les mange
    for i, ghost in enumerate(currentGhostPos):
        if util.manhattanDistance(newPos, ghost) < 3 and newScaredTimes[i] == 0:
            scores -= 60
        if util.manhattanDistance(newPos, ghost) < 10 and newScaredTimes[i] != 0:
            scores += 20*(15 - util.manhattanDistance(newPos, ghost))
    # maintenant boucle avec la bouffe
    distances = []
    foodList = foodGrid.asList()
    # cette boucle est limitée à cause de la manhattandistance qui prend pas en compte les murs on ne s'attardera pas la dessus
    # meme si quelque chose pourrait etre fait pour améliorer ce défaut
    for x,y in foodList:
        distances.append(util.manhattanDistance((x,y), newPos))
    if distances != []:
        closest = min(distances)
        scores += (40 - closest) / 10 # si on divise pas par 10 ca va jamais manger une food isolée car score(eat) < score(stay_around)
    #print(newScaredTimes)
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return - float("inf")
    return scores

# Abbreviation
better = betterEvaluationFunction


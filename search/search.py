# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def GraphSearch(problem, strategie):

    # premiere solution a permis de descendre jusqu'en bas de l'arbre
    # comment en ayant ca en retirer la liste des actions ?
    # necessite de garder en memoire tous les chemins associés aux noeuds de frontiere sous forme de listes
    # donc on va avoir une liste de liste avec la longueur des listes de listes correspondant a la profondeur du chemin
    # pour la DFS en soi pour la BFS c'est  pareil c'est la profondeur qu'on représente par la longueur de la liste de liste
    explored_nodes = []
    # initialisation
    # hyper important de push une liste sinon sur le successor_path.append(successor) ca va faire probleme de ndo
    strategie.push([(problem.getStartState(), "Stop", 0)])
    
        # check que liste non vide auquel cas on stoppe
    while not strategie.isEmpty():

        # on récupère le chemin associé au noeud qu'on va explorer
        path = strategie.pop() # revient à prendre le [-1] chemin le plus profond donc (ou au moins un des plus profonds on se place dfs ici)
        current_node = path[-1][0]
        
        if problem.isGoalState(current_node):
            actions_to_do = []
            for elt in path[1:]:
                actions_to_do.append(elt[1])
            return actions_to_do
        
        # releve ici le current_node
        if current_node not in explored_nodes: 
            # on ajoute ce noeud aux visités
            explored_nodes.append(current_node)
            # On ajoute alors aux successeurs le chemin de leurs parents pour la suite (pour que la boucle tourne)
            new_nodes = problem.getSuccessors(current_node)
            for node in new_nodes:
                if node[0] not in explored_nodes:
                    successor_path = path[:]
                    successor_path.append(node)
                    # Met le successeur dans la strategie
                    strategie.push(successor_path)
    return False
        
def depthFirstSearch(problem):
    stack = util.Stack()
    return GraphSearch(problem, stack)

def breadthFirstSearch(problem):
    queue = util.Queue()
    return GraphSearch(problem, queue)

def uniformCostSearch(problem):
    # Dans ce problème il est nécessaire d'associer un coût à chaque chemin pour prioriser les chemins selon le coût. Les
    # explorer en fonction du coût (principe de UCS finalement). 
    # Nous allons donc créer la fonction de coût de la priorityqueuefonction. On se sert d'une lambda (cette derniere releve le cout
    # de chaque action du chemin que l'on sera en train d'etudier (on démarre à 1 pour pas prendre le stop)
    # (pour rappel laction est en position 2, donc index 1)
    
    cost = lambda path: problem.getCostOfActions([etape[1] for etape in path][1:])
    priorityqueue = util.PriorityQueueWithFunction(cost)
    
    return GraphSearch(problem, priorityqueue)
    
def nullHeuristic(state, problem=None):
    return 0
    
def aStarSearch(problem, heuristic=nullHeuristic):
        # il faut ici réutiliser le code de l'UCS mais rajouter l'heuristique

    cost = lambda path: problem.getCostOfActions([etape[1] for etape in path][1:]) + heuristic(path[-1][0], problem)
    priorityqueue = util.PriorityQueueWithFunction(cost)
    return GraphSearch(problem, priorityqueue)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
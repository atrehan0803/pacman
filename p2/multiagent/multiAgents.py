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

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDistances = []
        ghostDistances = [] 

        # Distances of ghosts to pacman 
        for g in newGhostStates:
            ghostDistances.append(manhattanDistance(newPos, g.getPosition()))
        # Closest ghost to pacman 
        closestGhost = min(ghostDistances)

        # Checks if pacman is too close to ghost 
        if closestGhost <= 1:
            return -float('inf'); 
    
        # Distances of food pellets to pacman 
        for f in newFood.asList():
             foodDistances.append(manhattanDistance(newPos, f))

        # Closest food to pacman 
        if len(foodDistances) > 0:
            closestFood = min(foodDistances, default = float('inf'))
        else: 
            closestFood = 0

        # Calculates the score and weights the closestGhost and closestFood accordingly
        score = successorGameState.getScore() - 2.0/(1.0+closestGhost) + 10.0/(1.0+closestFood)
        
        # Reward for getting all the food 
        if successorGameState.getNumFood() == 0:
            score += 1000

        # Returns the score 
        return score 

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth=2):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth=2):
        self.index = 0  # Pacman is always agent index 0
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
        ghostIndex = [i for i in range(1, gameState.getNumAgents())]
        nextAction = None
        v = -float('inf')

        def max_value(state, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = -float('inf')
            bestAction = None
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value = min_value(successor, depth, ghostIndex[0])
                if value > v:
                    v = value
                    bestAction = action
            return v

        def min_value(state, depth, g):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v = float('inf')
            isLastGhost = (g == ghostIndex[-1])
            for action in state.getLegalActions(g):
                successor = state.generateSuccessor(g, action)
                if isLastGhost:
                    value = max_value(successor, depth + 1)
                else:
                    value = min_value(successor, depth, g + 1)
                if isinstance(value, str):
                    continue
                if value < v:
                    v = value
            return v

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = min_value(successor, 0, ghostIndex[0])
            if isinstance(value, str):
                continue
            if value > v:
                v = value
                nextAction = action
        return nextAction


class AlphaBetaAgent(MultiAgentSearchAgent):

    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):

        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        # Initializations:
        alpha = -float('inf')
        beta = float('inf')
        # will increase each time a state is expanded 
        state_counter = 0
        v = -float('inf')
        nextAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value, state_counter = self.min_value(successor, alpha, beta, 0, 1, state_counter)
            if value > v:
                v = value
                nextAction = action
            alpha = max(alpha, v)
        # chooses best course of action to take 
        return nextAction

    # Function for max_value adapted from the alpha-beta implementation given in assignment 
    def max_value(self, state, alpha, beta, depth, agentIndex, state_counter):
        # If the game is over, then the evaluation function value is returned
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), state_counter
        # Initialize v to negative infinity 
        v = -float('inf')
        # For each successor in state
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            value, state_counter = self.min_value(successor, alpha, beta, depth, agentIndex + 1, state_counter + 1)
            v = max(v, value)
            if v > beta:
                return v, state_counter
            #max's best option on path to root 
            alpha = max(alpha, v)
        return v, state_counter

    # Function for min_value adapted from the alpha-beta implementation given in assignment 
    def min_value(self, state, alpha, beta, depth, agentIndex, state_counter):
        # If the game is over, then the evaluation function value is returned
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state), state_counter
        # Initialize v to infinity 
        v = float('inf')
        # For each successor in state 
        for action in state.getLegalActions(agentIndex):
            successor = state.generateSuccessor(agentIndex, action)
            if agentIndex == state.getNumAgents() - 1:
                value, state_counter = self.max_value(successor, alpha, beta, depth + 1, 0, state_counter + 1)
            else:
                value, state_counter = self.min_value(successor, alpha, beta, depth, agentIndex + 1, state_counter + 1)
            v = min(v, value)
            if v < alpha:
                return v, state_counter
            #min's best option on path to root 
            beta = min(beta, v)
        return v, state_counter

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # For each legal action available to Pacman, calculate the expectimax value
        # associated with that action and choose the action with the highest value.
        actions = gameState.getLegalActions(0)
        best_action = None
        best_value = float('-inf')
        for action in actions:
            # Calculate the expectimax value of this action.
            value = self.expectimaxValue(gameState.generateSuccessor(0, action), self.depth, 1)
            # If this value is better than the best one seen so far, update the best
            # action and value accordingly.
            if value > best_value:
                best_action, best_value = action, value
        return best_action
    
    def expectimaxValue(self, gameState, depth, agentIndex):
        # If we've reached the maximum search depth or if the game is over, return the
        # heuristic evaluation of this state.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # If we're dealing with Pacman, calculate the maximum value of all legal actions
        # from this state.
        if agentIndex == 0:
            actions = gameState.getLegalActions(0)
            max_value = float('-inf')
            for action in actions:
                sucState = gameState.generateSuccessor(0, action)
                value = self.expectimaxValue(sucState, depth, 1)
                max_value = max(max_value, value)
            return max_value
        # If we're dealing with a ghost, calculate the average value of all legal actions
        # from this state.
        else:
            actions = gameState.getLegalActions(agentIndex)
            avg_value = 0.0
            for action in actions:
                sucState = gameState.generateSuccessor(agentIndex, action)
                # If this is the last ghost, decrement the depth counter since Pacman
                # will move next.
                next_depth = depth - 1 if agentIndex == gameState.getNumAgents() - 1 else depth
                value = self.expectimaxValue(sucState, next_depth, (agentIndex + 1) % gameState.getNumAgents())
                avg_value += value
            return avg_value / len(actions)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Changed the evaluation function to match the currentGameState. Used different weights to modify the scores. 
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodDistances = []
    score = currentGameState.getScore() 
    # Distances of ghosts to pacman 
    for g in newGhostStates:
        currDistance = manhattanDistance(newPos, g.getPosition())
        if currDistance > 0:
            if g.scaredTimer > 0:
                score +=100.0/currDistance
            else:
                score+= -10.0 
        else:
            return -float('inf')
    
    # Distances of food pellets to pacman 
    for f in newFood.asList():
        foodDistances.append(manhattanDistance(newPos, f))

    # Closest food to pacman 
    if len(foodDistances) > 0:
        score += 10.0/(1.0+min(foodDistances))
    else: 
        score += 10.0

    # Reward for getting all the food 
    if currentGameState.getNumFood() == 0:
        score += 1000

    # Returns the score 
    return score 

# Abbreviation
better = betterEvaluationFunction

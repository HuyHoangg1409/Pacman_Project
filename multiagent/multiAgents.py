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
        for ghost in newGhostStates:
            dist = manhattanDistance(newPos, ghost.getPosition())
            if ghost.scaredTimer == 0 and dist < 2:
                return -float('inf')

        foodList = newFood.asList()
        if foodList:
            minFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
        else:
            minFoodDist = 0

        return successorGameState.getScore() + (10.0 / (minFoodDist + 1))

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        legalAction = gameState.getLegalActions(0)
        if 'Stop' in legalAction:
            legalAction.remove('Stop')
        ans = legalAction[0]
        M = -float('inf')
        for i in legalAction:
            k = self.minimax(gameState.generateSuccessor(0, i), 1)
            if M < k :
                M = k
                ans = i
        return ans

    
    def minimax(self, gameState: GameState, depth: int):
        if depth // gameState.getNumAgents() >= self.depth or gameState.isLose() or gameState.isWin() :
            return self.evaluationFunction(gameState)
        
        numAgents = gameState.getNumAgents()
        agentIndex = depth % numAgents
        legalAction = gameState.getLegalActions(agentIndex)
        if agentIndex == 0 and 'Stop' in legalAction:
            legalAction.remove('Stop')
        
        if depth % gameState.getNumAgents() == 0 :
            score = -float('inf')
        else :
            score = float('inf')
        for i in legalAction :
            a = gameState.generateSuccessor(agentIndex, i)
            p = self.minimax(a, depth + 1)
            if depth % gameState.getNumAgents() == 0:
                score = max(score, p)
            if depth % gameState.getNumAgents() != 0:
                score = min(score, p)
        return score


        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        adj = gameState.getLegalActions(0)
        alpha = -float('inf')
        beta = float('inf')
        ans = adj[0]
        v = - float('inf')

        for successor in adj :
            next_state = gameState.generateSuccessor(0, successor)
            a = self.minValue(next_state, alpha, beta, (1) % gameState.getNumAgents(), 1)
            if v < a :
                v = a
                ans = successor
            if v > beta :
                return ans
            alpha = max(alpha, v)
        return ans
            

    def maxValue(self, gameState: GameState, alpha, beta, agentIndex, depth: int) :
        n = gameState.getNumAgents()
        if depth // n >= self.depth or gameState.isLose() or gameState.isWin() :
            return self.evaluationFunction(gameState)

        
        v = - float('inf')
        adj = gameState.getLegalActions(agentIndex)
        for successor in adj:
            next_state = gameState.generateSuccessor(agentIndex, successor)
            v = max(v, self.minValue(next_state, alpha, beta, (agentIndex + 1) % n, depth + 1))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState: GameState, alpha, beta, agentIndex, depth: int) :
        n = gameState.getNumAgents()
        if depth // n >= self.depth or gameState.isLose() or gameState.isWin() :
            return self.evaluationFunction(gameState)

        v = float('inf')
        adj = gameState.getLegalActions(agentIndex)
        for successor in adj:
            next_state = gameState.generateSuccessor(agentIndex, successor)
            
            if (agentIndex + 1) % n != 0:
                v = min(v, self.minValue(next_state, alpha, beta, (agentIndex + 1) % n, depth + 1))
            else :
                v = min(v, self.maxValue(next_state, alpha, beta, (agentIndex + 1) % n, depth + 1))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

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
        "*** YOUR CODE HERE ***"
        adj = gameState.getLegalActions(0)
        if 'Stop' in adj:
            adj.remove('Stop')
        ans = adj[0]
        v = - float('inf')

        for successor in adj :
            next_state = gameState.generateSuccessor(0, successor)
            a = self.minValue(next_state, (1) % gameState.getNumAgents(), 1)
            if v < a :
                v = a
                ans = successor
        return ans

    def maxValue(self, gameState: GameState, agentIndex, depth: int) :
        n = gameState.getNumAgents()
        if depth // n >= self.depth or gameState.isLose() or gameState.isWin() :
            return self.evaluationFunction(gameState)

        
        v = - float('inf')
        adj = gameState.getLegalActions(agentIndex)
        if 'Stop' in adj:
            adj.remove('Stop')
        for successor in adj:
            next_state = gameState.generateSuccessor(agentIndex, successor)
            v = max(v, self.minValue(next_state, (agentIndex + 1) % n, depth + 1))
        return v
    
    def minValue(self, gameState: GameState, agentIndex, depth: int) :
        n = gameState.getNumAgents()
        if depth // n >= self.depth or gameState.isLose() or gameState.isWin() :
            return self.evaluationFunction(gameState)

        v = 0
        adj = gameState.getLegalActions(agentIndex)
        if not adj:
            return self.evaluationFunction(gameState)

        if 'Stop' in adj:
            adj.remove('Stop')
        
        for successor in adj:
            next_state = gameState.generateSuccessor(agentIndex, successor)
            if (agentIndex + 1) % n != 0:
                v = v + self.minValue(next_state, (agentIndex + 1) % n, depth + 1)
            else :
                v = v + self.maxValue(next_state, (agentIndex + 1) % n, depth + 1)

        v = v / len(adj)
        return v

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    score = current score + 10/khoảng cách thức ăn gần nhất - 20 * số thức ăn còn lại trên sân 
        + 500 * khoảng cách đến con ma bị dọa - 500 / khoảng cách ma cách nó ít hơn 2 ô - 1 / khoảng cách ma cách nó nhiều hơn 2 ô
        - 100 * số viên dọa ma còn lại trên sân
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    score = currentGameState.getScore()

    foodList = newFood.asList()
    if foodList:
        minFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
        score += 10.0 / (minFoodDist + 1)
    
    score -= 20.0 * len(foodList)

    for ghost in ghostStates:
        dist = manhattanDistance(newPos, ghost.getPosition())
        
        if ghost.scaredTimer > 0:
            if dist < ghost.scaredTimer:
                score += 500.0 / (dist + 0.1)
        else:
            if dist > 0:
                if dist <= 2:
                    score -= 500.0 / dist
                else:
                    score -= 1.0 / dist

    capsules = currentGameState.getCapsules()
    score -= 100.0 * len(capsules)

    return score

# Abbreviation
better = betterEvaluationFunction

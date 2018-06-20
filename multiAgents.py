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

# # # # # # # # # # # # # # # # # # # # #
# Student : Albert Eduard Merino Pulido #
# Course : 2016/17                      #
# # # # # # # # # # # # # # # # # # # # #


import random
import sys

import util
from game import Agent
from util import manhattanDistance


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
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in
                  legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                       scores[index] == bestScore]
        chosenIndex = random.choice(
            bestIndices)  # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pos = successorGameState.getPacmanPosition()
        food = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        totalScore = 0.0
        for x in range(food.width):
            for y in range(food.height):
                if currentGameState.hasFood(x, y):
                    d = manhattanDistance((x, y), pos)
                    totalScore += 50 if (d == 0) else 1.0 / (d * d)

        for capsule in currentGameState.getCapsules():
            d = manhattanDistance(capsule, pos)
            totalScore += 200 if d == 0 else 1.0 / (d * d)

        for ghost in newGhostStates:
            d = manhattanDistance(ghost.getPosition(), pos)
            if d <= 1:
                totalScore += 30000 if ghost.scaredTimer != 0 else -300

        return totalScore


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

    def __init__(self, evalFn='betterEvaluationFunction', depth='1'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def utility(self, gameState):
        return self.evaluationFunction(gameState)

    def terminalTest(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or depth == 0

    def result(self, gameState, agent, action):
        return gameState.generateSuccessor(agent, action)


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
        """
        return self.minimax_decision(gameState)

    def max_value(self, gameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.utility(gameState)
        v = -sys.maxsize
        for action in gameState.getLegalActions(agent):
            v = max(v, self.min_value(self.result(gameState, agent, action), 1
                                      , depth))
        return v

    def min_value(self, gameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.utility(gameState)
        v = sys.maxsize
        for action in gameState.getLegalActions(agent):
            if agent == gameState.getNumAgents() - 1:
                # This is the last ghost
                v = min(v,
                        self.max_value(self.result(gameState, agent, action),
                                       0, depth - 1))
            else:
                v = min(v, self.min_value(self.result(gameState, agent, action)
                                          , agent + 1, depth))
        return v

    def minimax_decision(self, gameState):
        v = -sys.maxsize
        actions = []
        for action in gameState.getLegalActions(0):
            u = self.min_value(self.result(gameState, 0, action), 1,
                               self.depth)
            if u == v:
                actions.append(action)
            elif u > v:
                v = u
                actions = [action]
        return random.choice(actions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def max_value(gameState, alpha, beta, agent, depth):
            if self.terminalTest(gameState, depth):
                return self.utility(gameState)
            v = -sys.maxsize
            for action in gameState.getLegalActions(agent):
                v = max(v, min_value(self.result(gameState, agent, action),
                                     alpha, beta, 1, depth))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(gameState, alpha, beta, agent, depth):
            if self.terminalTest(gameState, depth):
                return self.utility(gameState)
            v = sys.maxsize
            for action in gameState.getLegalActions(agent):
                if agent == gameState.getNumAgents() - 1:
                    v = min(v, max_value(self.result(gameState, agent, action),
                                         alpha, beta, 0, depth - 1))
                else:
                    v = min(v, min_value(self.result(gameState, agent, action),
                                         alpha, beta, agent + 1, depth))
                if v < alpha: return v
                beta = min(beta, v)
            return v

        def alpha_beta_decision(gameState, alpha, beta):
            v = -sys.maxsize
            actions = []
            for action in gameState.getLegalActions(0):
                u = min_value(self.result(gameState, 0, action), alpha, beta,
                              1, self.depth)
                if u == v:
                    actions.append(action)
                elif u > v:
                    v = u
                    actions = [action]
                alpha = max(alpha, u)
            return random.choice(actions)

        return alpha_beta_decision(gameState, -sys.maxsize , sys.maxsize )


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.minimax_decision(gameState)

    def max_value(self, gameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.utility(gameState)
        v = -sys.maxsize
        for action in gameState.getLegalActions(agent):
            v = max(v, self.min_value(self.result(gameState, agent, action), 1, depth))
        return v

    def min_value(self, gameState, agent, depth):
        if self.terminalTest(gameState, depth):
            return self.utility(gameState)
        v = sys.maxsize
        all = 0
        number = 0.0
        for action in gameState.getLegalActions(agent):
            if agent == gameState.getNumAgents() - 1:
                # This is the last ghost
                v = self.max_value(self.result(gameState, agent, action),
                                   0, depth - 1)
            else:
                v = self.min_value(self.result(gameState, agent, action)
                                   , agent + 1, depth)
            all += v
            number += 1
        return all / number

    def minimax_decision(self, gameState):
        v = -sys.maxsize
        actions = []
        for action in gameState.getLegalActions(0):
            u = self.min_value(self.result(gameState, 0, action), 1,
                               self.depth)
            if u == v:
                actions.append(action)
            elif u > v:
                v = u
                actions = [action]
        return random.choice(actions)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    pacmanScore = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in
                      ghostStates]

    totalScore = pacmanScore
    for x in range(food.width):
        for y in range(food.height):
            if currentGameState.hasFood(x, y):
                d = manhattanDistance((x, y), pos)
                totalScore += 50 if (d == 0) else 1.0 / (d * d)

    for capsule in currentGameState.getCapsules():
        d = manhattanDistance(capsule, pos)
        totalScore += 200 if d == 0 else 1.0 / (d * d)

    for ghost in ghostStates:
        d = manhattanDistance(ghost.getPosition(), pos)
        if d <= 1:
            totalScore += 30000 if ghost.scaredTimer != 0 else -300

    return totalScore


# Abbreviation
better = betterEvaluationFunction

# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html

import random
import time
# Replay memory
from collections import deque

import game
# Neural nets
from DQN import *
# Pacman game
from pacman import Directions

params = {
    # Model backups
    'save_interval': 10000
}


class PacmanDQN(game.Agent):
    def __init__(self, args):

        print("Initialise DQN Agent")

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['load_file'] = args['load_file']
        self.params['save_file'] = args['save_file']
        self.params['explore_action'] = args['explore_action']
        self.params['train_start'] = args['train_start']
        self.params['batch_size'] = args['batch_size']
        self.params['mem_size'] = args['mem_size']
        self.params['discount'] = args['discount']
        self.params['lr'] = args['lr']
        self.params['lr_cyclic'] = args['lr_cyclic']
        self.params['rms_decay'] = args['rms_decay']
        self.params['rms_eps'] = args['rms_eps']
        self.params['eps'] = args['eps']
        self.params['eps_final'] = args['eps_final']
        self.params['eps_step'] = args['eps_step']
        # time started
        self.record_time = args['record_time']

        self.get_action = getattr(self, 'get_action_' + self.params['explore_action'])
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1,
                                                     inter_op_parallelism_threads=1))
        self.qnet = DQN(self.params)

        # Q and cost
        self.Q_global = []
        self.cost_disp = 0

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.

        self.replay_mem = deque()
        self.last_scores = deque()

        from multiAgents import ReflexAgent, MinimaxAgent
        self.reflex = ReflexAgent()
        self.minimax = MinimaxAgent()
        self.minimax2 = MinimaxAgent(depth='2')

        self.log_filename = 'logs/' + params['save_file'] + '_' + str(self.record_time) + '-l-' + str(
            self.params['width']) + '-m-' + str(self.params['height']) + '.log'
        log_file = open(self.log_filename, 'a')
        log_file.write("#,steps,steps_t,t,r,e,Q,won,training\n")

    def set_agent(self, agent, level):
        if level == 3:
            explore_action = agent
        else:
            explore_action = getattr(self, 'get_agent_' + agent)()

        self.get_action = getattr(self, 'get_action_' + explore_action)
        return explore_action

    def get_agent_random(self):
        return 'random'

    def get_agent_reflex(self):
        return 'reflex'

    def get_agent_minimax(self):
        return 'minimax'

    def get_agent_minimax2(self):
        return 'minimax2'

    def get_agent_random_reflex(self):
        if np.random.rand() < 0.3:
            return self.get_agent_random()
        else:
            return self.get_agent_reflex()

    def get_agent_random_minimax(self):
        if np.random.rand() < 0.3:
            return self.get_agent_random()
        else:
            return self.get_agent_minimax()

    def get_agent_reflex_minimax(self):
        if np.random.rand() < 0.7:
            return self.get_agent_reflex()
        else:
            return self.get_agent_minimax()

    def get_agent_random_reflex_minimax(self):
        rand = np.random.rand()
        if rand < 0.2:
            return self.get_agent_random()
        elif rand < 0.6:
            return self.get_agent_reflex()
        else:
            return self.get_agent_minimax()

    def getMove(self, state):
        # Exploit / Explore
        if np.random.rand() > self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict={self.qnet.x: np.reshape(self.current_state,
                                                   (1, self.params['width'], self.params['height'], 6)),
                           self.qnet.q_t: np.zeros(1),
                           self.qnet.actions: np.zeros((1, 4)),
                           self.qnet.terminals: np.zeros(1),
                           self.qnet.rewards: np.zeros(1)})[0]

            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else:
            # Explore:
            move = self.get_action(state)

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    def get_action_random(self, gameState):
        return self.get_direction(np.random.randint(0, 4))

    def get_action_reflex(self, gameState):
        return self.reflex.getAction(gameState)

    def get_action_minimax(self, gameState):
        return self.minimax.getAction(gameState)

    def get_action_minimax2(self, gameState):
        return self.minimax2.getAction(gameState)

    def get_action_random_reflex(self, gameState):
        if np.random.rand() < 0.3:
            return self.get_action_random(gameState)
        else:
            return self.get_action_reflex(gameState)

    def get_action_random_minimax(self, gameState):
        if np.random.rand() < 0.3:
            return self.get_action_random(gameState)
        else:
            return self.get_action_minimax(gameState)

    def get_action_reflex_minimax(self, gameState):
        if np.random.rand() < 0.7:
            return self.get_action_reflex(gameState)
        else:
            return self.get_action_minimax(gameState)

    def get_action_random_reflex_minimax(self, gameState):
        rand = np.random.rand()
        if rand < 0.2:
            return self.get_action_random(gameState)
        elif rand < 0.6:
            return self.get_action_reflex(gameState)
        else:
            return self.get_action_minimax(gameState)

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.getStateMatrices(state)

            # Process current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 50.  # Eat ghost   (Yum! Yum!)
            elif reward > 0:
                self.last_reward = 10.  # Eat food    (Yum!)
            elif reward < -10:
                self.last_reward = -500.  # Get eaten   (Ouch!) -500
                self.won = False
            elif reward < 0:
                self.last_reward = -1.  # Punish time (Pff..)

            if (self.terminal and self.won):
                self.last_reward = 100.
            self.ep_rew += self.last_reward

            # Store last experience into memory
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            # self.save_model(params['save_file'])

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        if self.params['training']:
            self.params['eps'] = max(self.params['eps_final'], 1.00 - float(self.cnt) / float(self.params['eps_step']))
        else:
            self.params['eps'] = 0.0

    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)

        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open(self.log_filename, 'a')
        log_file.write("%4d,%5d,%5d,%4f,%12f,%10f,%10f,%r,%r\n" %
                       (self.numeps, self.local_cnt, self.cnt, time.time() - self.s, self.ep_rew, self.params['eps'],
                        max(self.Q_global, default=float('nan')), self.won, self.params['training']))

    def train(self):
        # Train
        if self.local_cnt > self.params['train_start']:
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = []  # States (s)
            batch_r = []  # Rewards (r)
            batch_a = []  # Actions (a)
            batch_n = []  # Next states (s')
            batch_t = []  # Terminal state (t)

            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)

    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):
            actions_onehot[i][int(actions[i])] = 1
        return actions_onehot

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """

        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=int)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1 - i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=int)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=int)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=int)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1 - int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=int)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1 - i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=int)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1 - i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    def registerInitialState(self, state):  # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

    def getAction(self, state):
        move = self.getMove(state)

        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP

        return move

    def save_model(self, where):
        if where:
            self.qnet.save_ckpt(where)
            # self.qnet.save_ckpt(
            #     'saves/model_' + params['save_file'] + '_' + where + "_" + str(self.cnt) + '_' + str(self.numeps))
            #
            print('Model saved')

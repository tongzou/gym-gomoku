"""
Game of Gomoku, This implements the Gomoku OPEN AI environment.
"""
import sys
import numpy as np
import re
import gym
from gym import spaces
from gym import error
from gym.utils import seeding
from six import StringIO


def shift(xs, n):
    if n == 0:
        return xs[:]
    e = np.zeros(xs.shape, dtype=int)
    if n > 0:
        e[:n] = 0
        e[n:] = xs[:-n]
    else:
        e[n:] = 0
        e[:n] = xs[-n:]
    return e


# Adversary policies #
def make_random_policy(np_random):
    def random_policy(curr_state, prev_state, prev_action):
        possible_moves = GomokuEnv.get_possible_actions(curr_state)
        # No moves left
        if len(possible_moves) == 0:
            return None
        a = np_random.randint(len(possible_moves))
        return possible_moves[a]
    return random_policy


'''
Implements the naive policy. This will be the evaluation metric for the Agent.
level:  0 do not search for connection
        1 search for connected 3's
        2 search for connected 2's
        3 search for 1's (this is the highest level for the agent)
'''


def make_naive_policy(board_size, level=3, win_len=5):
    print('making naive policy: {}, {}, {}'.format(board_size, level, win_len))
    def opponent_policy(curr_state, prev_state, prev_action):
        opponent_policy.second_move = False
        # check if a new games is started.
        if np.count_nonzero(curr_state[2, :, :]) == board_size ** 2 - 1:
            opponent_policy.second_move = True

        # coords is the coordinate of the previous action.
        coords = GomokuEnv.action_to_coordinate(
            board_size, prev_action) if prev_action is not None else None

        if prev_state is None:
            '''
                First move should be the center of the board.
            '''
            move = (board_size//2, board_size//2)
        elif opponent_policy.second_move:
            '''
                If the AI must go second, it shouldn't think,
                it should just go diagonal adjacent to the first
                placed tile; diagonal into the larger area of the
                board if one exists
            '''
            if coords[1] <= board_size//2:
                dy = 1
            else:
                dy = -1

            if coords[0] <= board_size//2:
                dx = 1
            else:
                dx = -1
            move = (coords[0] + dx, coords[1] + dy)
            opponent_policy.second_move = False
        else:
            free_x, free_y = np.where(curr_state[2, :, :] == 1)
            possible_moves = [(x, y) for x, y in zip(free_x, free_y)]
            if len(possible_moves) == 0:
                # no more moves
                return None
            '''
                Strategy for the naive agent:
                1. Search if there is a win opportunity.
                2. Search if opponent is winning, if yes, then block
                3. Search if opponent has a open stream that equals 2 less than win_len, if yes, then block
                3. Try to extend the longest existing trend.
            '''
            if curr_state[0, coords[0], coords[1]] != 0:
                color = 1
            else:
                color = 0

            # 1: opponent position, 2: empty, 3: my position
            my_board = np.add(np.subtract(
                curr_state[color, :, :], curr_state[1-color, :, :]), 2)
            # print(my_board)
            # check if we have a winning move
            move = search_winning_move(my_board, '3')
            if move is None:
                # check if opponent has a winning move
                move = search_winning_move(my_board, '1')
            if move is None:
                # check if we have open win_len - 2
                move = search_move(
                    my_board, '2' + ('3' * (win_len - 2)) + '2', win_len)
            if move is None:
                # check if opponent has open win_len - 2
                move = search_move(
                    my_board, '2' + ('1' * (win_len - 2)) + '2', win_len)

            if move is None:
                for i in range(2, level + 2):
                    if win_len - i < 1:
                        break
                    # search for connected win_len - i stones
                    move = search_move(
                        my_board, '23{' + str(win_len - i) + '}', win_len - i + 1)
                    if move is None:
                        move = search_move(
                            my_board, '3{' + str(win_len - i) + '}2', win_len - i + 1, False)
                    if move is not None:
                        break

            if move is None:
                print(np.random.choice(possible_moves))
                move = np.random.choice(possible_moves)

        return GomokuEnv.coordinate_to_action(board_size, move)

    '''
        Search for winning move for the specified color
        c is the color of the player. '1' represents opponent position, '3' represents my position.
    '''
    def search_winning_move(board, c):
        # check if we have win_len - 1 connected and empty space to make a win
        for i in range(win_len):
            pattern = c * i + '2' + c * (win_len - i - 1)
            move = search_move(board, pattern, win_len, True, i)
            if move is not None:
                return move

        return None

    '''
        begin: if True, return begin of the pattern, otherwise return the end.
        offset: the extra offset to adjust.
    '''
    def search_move(board, pattern, size, begin=True, offset=0):
        # print('searching for pattern ' + pattern)
        search = GomokuEnv.search_board(board, pattern, size)
        if search is not None:
            # print('found: ' + str(search))
            coord = search[0]
            direction = search[1]
            if begin:
                delta = (0, 0)
            else:
                delta = ((size - 1) * direction[0], (size - 1) * direction[1])

            return [coord[0] + delta[1] + offset * direction[1], coord[1] + delta[0] + offset * direction[0]]

        return None

    return opponent_policy


class GomokuEnv(gym.Env):
    """
        Gomoku environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, board_size, win_len=5):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
            board_size: size of the Go board
            win_len: how many pieces connected will be considered as a win.
        """
        assert isinstance(
            board_size, int) and board_size >= 3, 'Invalid board size: {}'.format(board_size)
        assert isinstance(
            win_len, int) and win_len >= 3, 'Invalid winning length: {}'.format(win_len)
        self.board_size = board_size
        self.win_len = win_len

        colormap = {
            'black': GomokuEnv.BLACK,
            'white': GomokuEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error(
                "player_color must be 'black' or 'white', not {}".format(player_color))

        self._opponent = opponent

        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'numpy3c':
            raise error.Error(
                'Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign
        self.action_space = spaces.Discrete(self.board_size ** 2 + 1)
        observation = self.reset()
        self.observation_space = spaces.Box(
            np.zeros(observation.shape), np.ones(observation.shape))

        self._seed()
        self.prev_move = -1

    @property
    def opponent(self): 
        return self._opponent 

    @opponent.setter 
    def opponent(self, o): 
        self._opponent = o
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            elif self.opponent.startswith('naive'):
                try:
                    level = int(self.opponent[-1])
                except:
                    level = 0
                self.opponent_policy = make_naive_policy(
                    self.board_size, level, self.win_len)
            else:
                raise error.Error(
                    'Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def reset(self):
        self.state = np.zeros((3, self.board_size, self.board_size), dtype=int)
        self.state[2, :, :] = 1.0
        self.to_play = GomokuEnv.BLACK
        self.done = False

        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state, None, None)
            GomokuEnv.make_move(self.state, a, GomokuEnv.BLACK)
            self.to_play = GomokuEnv.WHITE
            self.prev_move = a
        return self.state

    def step(self, action):
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}

        prev_state = self.state
        if GomokuEnv.resign_move(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        elif not GomokuEnv.valid_move(self.state, action):
            if self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state, -1., True, {'state': self.state}
            else:
                raise error.Error(
                    'Unsupported illegal move action: {}'.format(self.illegal_move_mode))

        GomokuEnv.make_move(self.state, action, self.player_color)
        self.prev_move = action
        remaining_moves = np.count_nonzero(self.state[2, :, :])

        # Opponent play
        a = self.opponent_policy(self.state, prev_state, action)

        # Making move if there are moves left
        if a is not None:
            if GomokuEnv.resign_move(self.board_size, a):
                return self.state, 1, True, {'state': self.state}
            elif not GomokuEnv.valid_move(self.state, a):
                if self.illegal_move_mode == 'lose':
                    # Automatic loss on illegal move
                    self.done = True
                    return self.state, 1., True, {'state': self.state}
                else:
                    raise error.Error(
                        'Unsupported illegal move action: {}'.format(self.illegal_move_mode))
            else:
                GomokuEnv.make_move(self.state, a, 1 - self.player_color)

        reward = GomokuEnv.game_finished(
            self.state, self.player_color, self.win_len)
        self.done = reward != 0 or remaining_moves == 0 or remaining_moves == 1

        # check to see if we need to roll back opponent move if we have won already.
        if reward == 1 and a is not None:
            GomokuEnv.revert_move(self.state, a, 1 - self.player_color)
            pass
        else:
            self.prev_move = a
        return self.state, reward, self.done, {'state': self.state}

    def render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        outfile.write('To play: ')
        outfile.write('black' if self.to_play == GomokuEnv.BLACK else 'white')
        outfile.write('\n')
        d = board.shape[1]
        if d > 9:
            outfile.write(' ' * 24)
            for j in range(10, d + 1):
                outfile.write(' ' + str(int(j/10)))
            outfile.write('\n')
        outfile.write(' ' * 6)
        for j in range(d):
            outfile.write(' ' + str((j + 1) % 10))
        outfile.write('\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')
        for i in range(d):
            outfile.write(' ' * (3 if i < 9 else 2) + str(i + 1) + ' | ')
            for j in range(d):
                a = GomokuEnv.coordinate_to_action(self.board_size, [i, j])
                if board[2, i, j] == 1:
                    outfile.write('. ')
                elif board[0, i, j] == 1:
                    if self.prev_move == a:
                        outfile.write('X)')
                    else:
                        outfile.write('X ')
                else:
                    if self.prev_move == a:
                        outfile.write('O)')
                    else:
                        outfile.write('O ')
            outfile.write('|\n')
        outfile.write(' ' * 5 + '+' + '-' * (d * 2 + 1) + '+\n')

        if mode != 'human':
            return outfile

    @staticmethod
    def resign_move(board_size, action):
        return action == board_size ** 2

    @staticmethod
    def valid_move(board, action):
        coords = GomokuEnv.action_to_coordinate(board.shape[-1], action)
        if board[2, coords[0], coords[1]] == 1:
            return True
        else:
            return False

    @staticmethod
    def make_move(board, action, player):
        coords = GomokuEnv.action_to_coordinate(board.shape[-1], action)
        board[2, coords[0], coords[1]] = 0
        board[player, coords[0], coords[1]] = 1

    @staticmethod
    def revert_move(board, action, player):
        coords = GomokuEnv.action_to_coordinate(board.shape[-1], action)
        board[2, coords[0], coords[1]] = 1
        board[player, coords[0], coords[1]] = 0

    @staticmethod
    def coordinate_to_action(board_size, coords):
        return coords[0] * board_size + coords[1]

    @staticmethod
    def action_to_coordinate(board_size, action):
        return action // board_size, action % board_size

    @staticmethod
    def get_possible_actions(board):
        free_x, free_y = np.where(board[2, :, :] == 1)
        return [GomokuEnv.coordinate_to_action(board.shape[-1], [x, y]) for x, y in zip(free_x, free_y)]

    '''
        pattern is a regular expression to test for and size is the length of the pattern. size is only used for searching
        diagonal patterns.
    '''
    @staticmethod
    def search_board(player_board, pattern, size):
        search = GomokuEnv.search_horizontal(player_board, pattern)
        if search is not None:
            return search, [1, 0]

        search = GomokuEnv.search_horizontal(
            np.transpose(player_board), pattern)
        if search is not None:
            return [search[1], search[0]], [0, 1]

        return GomokuEnv.search_diagonal(player_board, pattern, size)

    @staticmethod
    def search_horizontal(player_board, pattern):
        d = player_board.shape[0]
        state = ''
        for i in range(d):
            state += ''.join(map(str, player_board[i])) + '-'

        index = re.search(pattern, state)
        if index is not None:
            index = index.start()
            index -= index // (d + 1)
            index = GomokuEnv.action_to_coordinate(player_board.shape[-1], index)
        return index

    @staticmethod
    def search_diagonal(player_board, pattern, size):
        d = player_board.shape[0]
        for i in range(d-size+1):
            forward = np.zeros((d, d), dtype=int)
            backward = np.zeros((d, d), dtype=int)
            for j in range(i, i + size):
                forward[j] = shift(player_board[j, :], j - i)
                backward[j] = shift(player_board[j, :], i - j)
            index = GomokuEnv.search_horizontal(np.transpose(forward), pattern)
            if index is not None:
                return [index[1], index[0]], [-1, 1]
            index = GomokuEnv.search_horizontal(
                np.transpose(backward), pattern)
            if index is not None:
                return [index[1], index[0]], [1, 1]

        return None

    @staticmethod
    def game_finished(board, first_color, win_len):
        # Returns 1 if first_color wins, -1 if first_color loses and 0 otherwise
        pattern = '1{' + str(win_len) + '}'
        if GomokuEnv.search_board(board[first_color, :, :], pattern, win_len) is not None:
            return 1

        if GomokuEnv.search_board(board[1 - first_color, :, :], pattern, win_len) is not None:
            return -1

        return 0

from gym.envs.registration import register

register(
    id='TicTacToe-v0',
    entry_point='envs.gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 3,
        'win_len': 3
    }
)

register(
    id='Gomoku9x9_5-v0',
    entry_point='envs.gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 9,
        'win_len': 5
    }
)

register(
    id='Gomoku13x13_5-v0',
    entry_point='envs.gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 13,
        'win_len': 5
    }
)

register(
    id='Gomoku19x19_5-v0',
    entry_point='envs.gomoku:GomokuEnv',
    kwargs={
        'player_color': 'black',
        'opponent': 'random',
        'observation_type': 'numpy3c',
        'illegal_move_mode': 'lose',
        'board_size': 19,
        'win_len': 5
    }
)

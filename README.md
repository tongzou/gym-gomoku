# gym-gomoku

OpenAI Gym Style Gomoku Environment. The following environments are available:

    TicTacToe-v0
    Gomoku9x9_5-v0: 9x9 Gomoku board
    Gomoku13x13_5-v0: 13x13 Gomoku board
    Gomoku19x19_5-v0: 19x19 Gomoku board

You can also register your own board with different size and winning length, like the following:

    gym.envs.registration.register(
        id='Gomoku8x8_4-v0',
        entry_point='gym_gomoku.envs:GomokuEnv',
        kwargs={
            'player_color': 'black',
            'opponent': 'random',
            'observation_type': 'numpy3c',
            'illegal_move_mode': 'lose',
            'board_size': 8,
            'win_len': 4
        }
    )
  


## Requirement

Python >= 3.5

## Install

    git clone https://github.com/tongzou/gym-gomoku.git
    cd gym-gomoku/
    pip install -e .


## Try example

    cd examples/
    python main.py

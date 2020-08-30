import gym
import gym_gomoku
import os


def cls():
    os.system('cls' if os.name == 'nt' else 'clear')


def run():
    cls()
    print("#" * 50 + "\n Welcome to Gomoku!\n By Tong Zou\n" + "#"*50 + "\n")
    while True:
        try:
            game = eval(input('Select a game you would like to play:\n' + ' ' * 5 +
                              '1. Tic Tac Toe\n' + ' ' * 5 +
                              '2. Gomoku 9x9\n' + ' ' * 5 +
                              '3. Gomoku 13x13\n' + ' ' * 5 +
                              '4. Gomoku 19x19\n' + ' ' * 5 +
                              '5. Exit\n'))
            if game < 1 or game > 5:
                raise ValueError()
            break
        except:
            print("Please enter a valid choice.")

    if game == 1:
        id  = 'TicTacToe-v0'
    elif game == 2:
        id = 'Gomoku9x9_5-v0'
    elif game == 3:
        id = 'Gomoku13x13_5-v0'
    elif game == 4:
        id = 'Gomoku19x19_5-v0'
    else:
        exit()

    black = input('Would you like to go first? (Y/n): ')
    black = black != 'n'

    cls()
    while True:
        try:
            opponent = eval(input('Select an opponent:\n' + ' ' * 5 +
                              '1. Random\n' + ' ' * 5 +
                              '2. Naive\n' + ' ' * 5 +
                              '3. Exit\n'))
            if opponent < 1 or opponent > 3:
                raise ValueError()
            break
        except:
            print("Please enter a valid choice.")

    if (opponent == 3):
        exit()

    opponent = 'random' if opponent == 1 else 'naive3'

    env = gym.make(id)
    env.player_color = 0 if black == True else 1
    env.opponent = opponent
    env.seed()
    play(env)
    input("Press Enter to continue...")

def play(env):
    observation = env.reset()

    while True:
        cls()
        env.render()
        try:
            action = eval('(' + input('Please enter your move in the form row, column: ') + ')')
        except SyntaxError:
            continue
        except NameError:
            return

        action = [action[0] - 1, action[1] - 1]
        action = env.coordinate_to_action(observation, action)

        observation, reward, done, _ = env.step(action)
        if done:
            cls()
            env.render()
            if reward == 1:
                print('You Win!')
            elif reward == -1:
                print('You Lost!')
            else:
                print('The game is a tie')
            return

if __name__ == '__main__':
    while True:
        run()

from tictactoe_env import TicTacToeEnv

if __name__ == "__main__" :
    env = TicTacToeEnv(size= 5)
    env.reset()
    terminal = False
    while not terminal:
        action = input("Select action (" + ", ".join(map(str, env.get_actions())) + "): ")
        _, winner, terminal, _, _ = env.step(int(action))
        env.render()
    if winner !=0:
        print("Winner is player: " + str(winner))
    else: 
        print('Draw')
    
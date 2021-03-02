import ArenaHP
from MCTS import MCTS
from othello.OthelloGame import OthelloGame, display
from othello.OthelloPlayers import *
from othello.keras.resNNet import NNetWrapper as NNet

import numpy as np
from utils import *


"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = OthelloGame(8)
hp = HumanOthelloPlayer(g)
t = hp.choose_turn()
hp.t = t#1 is hp first turn
g.t = t
# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play


n2 = NNet(g)
n2.load_checkpoint('./temp/','best.pth.tar')
args2 = dotdict({'numMCTSSims': 1200, 'cpuct':1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = ArenaHP.Arena(n2p, hp, g, display=display)
arena.t = t
print(arena.playGame(verbose=True))

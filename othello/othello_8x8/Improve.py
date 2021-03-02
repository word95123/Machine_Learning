from Arena import Arena

#from lastMCTSB import MCTS as lastmctsb
from MCTS import MCTS as mcts
from lastMCTSF import MCTS as lastmcts
from othello.OthelloGame import OthelloGame, display

from othello.keras.NNet import NNetWrapper as NNet

from othello.keras.newNNet import NNetWrapper as newNNet


import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = OthelloGame(8)

n2 = NNet(g)#new
n3 = NNet(g)#old
'''
n2 = NNet(g)
n3 = n2.__class__(g)
'''
sim = 100

#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')

for i in range(24,101,25):
    lastwins = 0
    prewins = 0
    draw = 0

    n2.load_checkpoint('./temp/Implement/deep3_feature',str(i+1)+'best.pth.tar')#last
    n3.load_checkpoint('./temp/Implement/origin',str(i+1)+'best.pth.tar')#pre
    args2 = dotdict({'numMCTSSims': sim, 'cpuct':1.0})
    args3 = dotdict({'numMCTSSims': sim, 'cpuct':1.0})
    #mcts2 = vmcts(g, n2, args2, visual())
    mcts2 = lastmcts(g, n2, args2)
    mcts3 = mcts(g, n3, args3)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    n3p = lambda x: np.argmax(mcts3.getActionProb(x, temp=0))

    arena = Arena(n3p, n2p, g)
    #arena = Arena(n3p, n2p, g, mcts2, visual())
    pwins, nwins, draws = arena.playGames(100)

    print(i+1)
    print('lastmcts/MCTS WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))#nwins == n2pwin == oneloss
    lastwins += nwins
    prewins += pwins
    draw += draws

    print('lastmcts/MCTS WINS : %d / %d ; DRAWS : %d' % (lastwins, prewins, draw))

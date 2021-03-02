
import random
import time
from GoMCTS import MCTS as MCTS
import numpy as np
from Go.GoGame import GoGame as Game
from Go.keras.NNet import NNetWrapper as nn
from utils import *
from gtp import BLACK, WHITE, PASS
from gtp import gtp_boolean, gtp_list, gtp_color, gtp_vertex, gtp_move, parse_vertex
import re



n = 9
game = Game(n)
nnet = nn(game)

nnet.load_checkpoint('./temp/','best.pth.tar')
args = dotdict({'numMCTSSims': 20, 'cpuct':1.0})
mcts = MCTS(game, nnet, args)

board = game.getInitBoard()


#time.sleep(random.uniform(0.1, 1.0)) # simulates process time
def play(vertex, color, board):
    if int(vertex[0])==0 and int(vertex[1])==0:
        board = board
    else:
        vertex = (int(vertex[0])-1, 8-(int(vertex[1])-1))
        action = vertex[1]*9 + vertex[0]
        board, curPlayer = game.getNextState(board, color, action)
    return board

while True:
    
    try:
        message = input()
    except:
        
        break

    #message = re.sub('[(,)]','',message)
    inpt = message.split()
    '''
    uniform (play color point) ex. play W A1
    '''
    if(str(inpt[0]) == 'play'):
        if str(inpt[1]).lower() == 'w':
            color = WHITE
        elif str(inpt[1]).lower() == 'b':
            color = BLACK
        
        vertex = parse_vertex(inpt[2])
        board = play(vertex, color, board)
        print('=',''.rstrip())
    '''
    uniform showboard is show board state
    '''
    if(str(inpt[0]) == 'showboard'):
        n = 9
        showboard = [None]*(n+2)
        for i in range(n+2):
            showboard[i] = [None]*(n+2)
        for i in range(n+2):
            for j in range(n+2):
                if i == 0 or i == n + 1:
                    if j == 0 or j == n + 1:
                        showboard[i][j] = ' '
                    else:
                        showboard[i][j] = 'ABCDEFGHJKLMNOPQRSTYVWYZ'[j - 1]
                else:
                    if j == 0 or j == n + 1:
                        showboard[i][j] = str((n + 1) - i)
                    else:
                        showboard[i][j] = '.'

        for i in range(n):
            for j in range(n):
                if int(board[i][j]) != 0:
                    if int(board[i][j]) == -1:
                        showboard[i+1][j+1] = 'O'
                    else:
                        showboard[i+1][j+1] = 'X'

        result = "\n"
        for i in range(n+2):
            for j in range(n+2):
                result += str(showboard[i][j])
                result += ' '
            result += '\n'
        print('=',result.rstrip())
    '''
    uniform genmove color ex. genmove W
    BLACK ask WHITE require to return point like (1,1) as A1
    '''
    if(str(inpt[0]) == 'genmove'):
        if str(inpt[1]).lower() == 'w':
            color = WHITE
        elif str(inpt[1]).lower() == 'b':
            color = BLACK
        
        if game.getGameEnded(board, color)==0:
            if color == WHITE:
                canonicalBoard = game.getCanonicalForm(board, color)
                pi = mcts.getActionProb(canonicalBoard, temp = 0)
            else:
                pi = mcts.getActionProb(board, temp = 0)
            
            for i in range(n*n):
                if pi[i] == 1:
                    action = i
                    
            board, curPlayer = game.getNextState(board, color, action)
            act = (action % n + 1, 10 - (int(action / n) + 1))
            act = gtp_vertex(act)
            print('=',act.rstrip())
        else:
            print('=','pass'.rstrip())


    print(''.rstrip())









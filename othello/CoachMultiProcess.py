from multiMCTS import MCTS
from othello.OthelloGame import OthelloGame as Game
from othello.keras.mNNet import NNetWrapper as nn
import numpy as np
from multiprocessing import Process,Manager
from collections import deque

from othello.keras.mOthelloNNet import OthelloNNet as onnet


def executeEpisode(game,mcts):
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board,curPlayer)
        temp = int(episodeStep < 15)
        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        sym = game.getSymmetries(canonicalBoard, pi)
        for b,p in sym:
            trainExamples.append([b, curPlayer, p, None])
        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(board, curPlayer, action)
        r = game.getGameEnded(board, curPlayer)
        if r!=0:
            if(r == -2):
                return [(x[0],x[2],0) for x in trainExamples]
            return [(x[0],x[2],r*((-1)**(x[1]!=curPlayer))) for x in trainExamples]
def learn(q):
    
    game = Game(8)
    nnet = onnet(game)
    for i in range(1,2):
        print('------ITER ' + str(i) + '------')
        iterationTrainExamples = deque([], maxlen=200000)
        for eps in range(1):
            mcts = MCTS(game, nnet)
            iterationTrainExamples += executeEpisode(game,mcts)
            print('1')
    q.put(iterationTrainExamples)



    
if __name__ == '__main__':
    

    q = Manager().Queue()
    
    p1 = Process(target=learn, args=(q,))
    p2 = Process(target=learn, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    res = []
    res.append(res1)
    res.append(res2)
    trainExamples = []
    game = Game(8)
    nnet = onnet(game)
    for e in res:
        trainExamples.extend(e)
    nnet.train(trainExamples)
    #print(res1)
    pass
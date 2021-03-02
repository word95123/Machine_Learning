from collections import deque
from Arena import Arena
#from MCTS import MCTS
from lastMCTS import MCTS as MCTS
from lastMCTST import MCTS as MCTST
import numpy as np
from processBar.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import multiprocessing as mp
from utils import *
from othello.OthelloGame import OthelloGame as Game
from othello.keras.newNNet import NNetWrapper as nn
from multiprocessing import Process,Manager

    
def executeEpisode(mcts, game):
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.

    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.

    Returns:
        trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.
    """
    trainExamples = []
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    
    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board,curPlayer)
        
        #temp = int(episodeStep < self.args.tempThreshold)
        
        countstep = 0
        for i in range(game.n):
            for j in range(game.n):
                if(canonicalBoard[i][j] == 0):
                    countstep += 1
        
        pi = mcts.getActionProb(canonicalBoard, temp=0)

        #trainExamples.append([canonicalBoard, self.curPlayer, pi, None])
        if(countstep > 12):
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
            '''
            cv = cv/32
            if(cv > 1):
                cv = 1
            elif cv < -1:
                cv = -1
            return [(x[0],x[2],0.5*(r*((-1)**(x[1]!=self.curPlayer)))+0.5*cv) for x in trainExamples]
            '''
            


def collect_data(q):
    args = dotdict({
        'numIters': 11,
        'numEps': 50,
        'tempThreshold': 15,
        'updateThreshold': 0.5,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 100,
        'arenaCompare': 2,
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': True,
        'load_folder_file': ('./temp/','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,

    })
    game = Game(8)
    nnet = nn(game)
    iterationTrainExamples = deque([], maxlen=args.maxlenOfQueue)

    eps_time = AverageMeter()
    bar = Bar('Self Play', max=args.numEps)
    end = time.time()
    for eps in range(int(args.numEps/2)):
        print(eps)
        mcts = MCTS(game, nnet, args)   # reset search tree
        iterationTrainExamples += executeEpisode(mcts, game)       
        # bookkeeping + plot progress
        
        eps_time.update(time.time() - end)
        end = time.time()
        bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=args.numEps, et=eps_time.avg,
                                                                                                    total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    q.put(iterationTrainExamples)
    bar.finish()

def learn(args, nnet, game, trainExamplesHistory):
    """
    Performs numIters iterations with numEps episodes of self-play in each
    iteration. After every iteration, it retrains neural network with
    examples in trainExamples (which has a maximium length of maxlenofQueue).
    It then pits the new neural network against the old one and accepts it
    only if it wins >= updateThreshold fraction of games.
    """

    for i in range(1, args.numIters+1):

        # bookkeeping
        print('------ITER ' + str(i) + '------')
        # examples of the iteration

        q = Manager().Queue()
        p1 = Process(target=collect_data, args=(q,))
        p2 = Process(target=collect_data, args=(q,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        res1 = q.get()
        res2 = q.get()
        # save the iteration examples to the history 
        trainExamplesHistory.append(res1)
        trainExamplesHistory.append(res2)
        
        if len(trainExamplesHistory) > args.numItersForTrainExamplesHistory:
            print("len(trainExamplesHistory) =", len(trainExamplesHistory), " => remove the oldest trainExamples")
            trainExamplesHistory.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration, so (i-1)  
        saveTrainExamples(i-1, args, trainExamplesHistory)
        
        # shuffle examlpes before training
        
        trainExamples = []
        for e in trainExamplesHistory:
            trainExamples.extend(e)
        shuffle(trainExamples)

        
        nnet.train(trainExamples)
        nnet.save_checkpoint(folder=args.checkpoint, filename=getCheckpointFile(i))
        #self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar') 
        
        '''
        pnet = nnet.__class__(game)
        # training new network, keeping a copy of the old one
        nnet.save_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
        pnet.load_checkpoint(folder=args.checkpoint, filename='temp.pth.tar')
        pmcts = MCTST(game, pnet, args)
        
        nnet.train(trainExamples)
        nmcts = MCTST(game, nnet, args)

        print('PITTING AGAINST PREVIOUS VERSION')
        arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                        lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
        pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
        arena = Arena(lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                        lambda x: np.argmax(pmcts.getActionProb(x, temp=0)), self.game)
        xnwins, xpwins, xdraws = arena.playGames(self.args.arenaCompare)

        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins+xnwins, pwins+xpwins, draws+xdraws))
        if pwins+nwins+xpwins+xnwins > 0 and float(nwins+xnwins)/(pwins+nwins+xpwins+xnwins) < self.args.updateThreshold:
            print('REJECTING NEW MODEL')
            self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
        else:
            print('ACCEPTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                
        '''

def getCheckpointFile(iteration):
    return str(iteration) + 'best.pth.tar'
    #return 'best.pth.tar'

def saveTrainExamples(iteration, args, trainExamplesHistory):
    folder = args.checkpoint
    if not os.path.exists(folder):
        os.makedirs(folder)
    #filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
    filename = os.path.join(folder, "best.pth.tar.examples")
    with open(filename, "wb+") as f:
        Pickler(f).dump(trainExamplesHistory)
    f.closed

def loadTrainExamples(args):
    modelFile = os.path.join(args.load_folder_file[0], args.load_folder_file[1])
    examplesFile = modelFile+".examples"
    if not os.path.isfile(examplesFile):
        print(examplesFile)
        r = input("File with trainExamples not found. Continue? [y|n]")
        if r != "y":
            sys.exit()
    else:
        print("File with trainExamples found. Read it.")
        with open(examplesFile, "rb") as f:
            trainExamplesHistory = Unpickler(f).load()
        f.closed
        # examples based on the model were already collected (loaded)
        skipFirstSelfPlay = True
    return trainExamplesHistory


if __name__ == '__main__':
    args = dotdict({
        'numIters': 11,
        'numEps': 50,
        'tempThreshold': 15,
        'updateThreshold': 0.5,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 100,
        'arenaCompare': 2,
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': True,
        'load_folder_file': ('./temp/','best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,

    })
    g = Game(8)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    trainExamplesHistory = []
    if args.load_model:
        print("Load trainExamples from file")
        trainExamplesHistory = loadTrainExamples(args)
    learn(args, nnet, g, trainExamplesHistory)

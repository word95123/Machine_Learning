from Coach import Coach
from othello.OthelloGame import OthelloGame as Game
from othello.keras.resNNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 20,
    'numEps': 1,
    'tempThreshold': 15,
    'updateThreshold': 0.5,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,
    'arenaCompare': 4,
    'cpuct': 1.0,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


g = Game(8)
nnet = nn(g)

if args.load_model:
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

c = Coach(g, nnet, args)

if args.load_model:
    print("Load trainExamples from file")
    c.loadTrainExamples()
c.learn()

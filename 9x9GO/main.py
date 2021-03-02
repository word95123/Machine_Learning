from GoCoach import Coach
from Go.GoGame import GoGame as Game
from Go.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 2,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.5,
    'maxlenOfQueue': 2000000,
    'numMCTSSims': 400,
    'arenaCompare': 4,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


g = Game(9)
nnet = nn(g)

if args.load_model:
    nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

c = Coach(g, nnet, args)

if args.load_model:
    print("Load trainExamples from file")
    c.loadTrainExamples()
c.learn()

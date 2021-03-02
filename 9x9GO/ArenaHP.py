import numpy as np

import time

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.t = 1

    def undo_import(self, data, undo = True):
        
        if(undo is not True):
            
            playerColor = data[0]
            data = data[2:]
            splitData = data.split(',')
        else:
            playerColor = data[0]
            splitData = data[1:]
        #print(playerColor)
        #print(splitData)
        if playerColor == 'B ':
            return self.undoProcess(splitData,-1)
        else:
            return self.undoProcess(splitData,1)
        #for unstr in splitData:
            #print(unstr)
            #print((int(unstr[0])-1) * self.game.n + ord(unstr[1]) - 96 - 1)

    

    def undoProcess(self, undoData, whoIsFirst):
        undo_board = self.game.getInitBoard()
        
        for unstr in undoData:
            action = (int(unstr[0])-1) * self.game.n + ord(unstr[1]) - 96 - 1
            undo_board, whoIsFirst = self.game.getNextState(undo_board, whoIsFirst, action)
        return undo_board


    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2.play, None, self.player1]
        curPlayer = self.t
        board = self.game.getInitBoard()

        undo_data = []
        undo_WB = []
        it = 0
        f = open('./Process.txt','w')
        if(self.t == -1):
            f.write('my pieces colour : B')
            undo_data.append('B ')
        else:
            f.write('my pieces colour : W')
            undo_data.append('W ')
        f.write('\n')
        f.close()
        #text = []
        while self.game.getGameEnded(board, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                #self.display(board)
            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
            #print(undo_data)
            if(action == -1):
                if(len(undo_data)>2):
                    undo_data = undo_data[:len(undo_data)-2]
                    undo_WB = undo_WB[:len(undo_WB)-2]
                    board = self.undo_import(undo_data)
                    self.player2.refresh()
                    self.player2.update_game(self.game.getCanonicalForm(board, -1))
                    #print(board)
                    f = open('./Process.txt','w')
                    if(self.t == -1):
                        f.write('my pieces colour : B')
                    else:
                        f.write('my pieces colour : W')
                    f.write('\n')
                    for i in range(len(undo_data)-1):
                        f.write(undo_data[i+1] + undo_WB[i] + ',')
                        
                    f.close()
                continue
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            undo_data.append(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97))
            
            
            
            #undo_board, curPlayer = self.game.getNextState(undo_board, curPlayer, action)
            
            if valids[action]==0:
                print(action)
                assert valids[action] >0
            
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            f = open('./Process.txt','a')
            
            if(curPlayer == self.t):
                if(int(action / self.game.n) == self.game.n):
                    #text.append(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + ',W(Pass)')
                    f.write(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + 'W(Pass),')
                    undo_WB.append('W(Pass)')
                else:
                    #text.append(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + ',W')
                    f.write(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + 'W,')
                    undo_WB.append('W')
            else:
                if(int(action / self.game.n) == self.game.n):
                    #text.append(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + ',B(Pass)')
                    f.write(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + 'B(Pass),')
                    undo_WB.append('B(Pass)')
                else:
                    #text.append(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + ',B')
                    f.write(str(int(action / self.game.n) + 1) + chr(action % self.game.n + 97) + 'B,')
                    undo_WB.append('B')
            f.close()
            self.player2.refresh()
            self.player2.last_move(action)
            #self.player2.update_game(board*-1)
            self.player2.update_game(self.game.getCanonicalForm(board, -1))
            
            
        #print(self.game.getGameEnded(board, curPlayer))
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            
            self.player2.winner_game(self.game.getCanonicalForm(board, -1))
            #self.player2.winner_game(board)
            '''
            if(self.t == -1):
                f.write('my pieces colour : B')
            else:
                f.write('my pieces colour : W')
            for i in text:
                f.write('\n' + i)
            f.close()
            '''
            
        
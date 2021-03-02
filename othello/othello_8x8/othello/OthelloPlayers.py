import numpy as np

import pygame
from pygame.locals import *
from sys import exit

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.gridSize = 50
        self.game = game
        self.Surf, self.screen = self.initial_pygame()
        self.lastMove = -1
        self.t = 1
        
        
        
    def savePic(self):
        pygame.image.save(self.screen, "./EndGameBoard.jpeg")

    def initial_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((self.gridSize * (self.game.n + 4),self.gridSize * (self.game.n + 2)))#windows size
        pygame.display.set_caption('Othello')#windows name
        Surf = pygame.Surface(screen.get_size())#this is background to draw a rect size
        Surf = Surf.convert()
        Surf.fill((255, 255, 255))
        
        return Surf, screen

    def choose_rank(self):
        self.Surf.fill((150, 150, 150), ((self.game.n + 1) * self.gridSize / 2 - 30,(self.game.n + 1) * self.gridSize / 2 - 60, 230, 50))
        self.Surf.fill((150, 150, 150), ((self.game.n + 1) * self.gridSize / 2 - 30,(self.game.n + 1) * self.gridSize / 2 + 60, 230, 50))
        font = pygame.font.Font(None, 50)
        text = font.render('Easy', 1, (10, 10, 10))
        self.Surf.blit(text, ((self.game.n + 1) * self.gridSize / 2 - 20,(self.game.n + 1) * self.gridSize / 2 - 50))
        text = font.render('hard', 1, (10, 10, 10))
        self.Surf.blit(text, ((self.game.n + 1) * self.gridSize / 2 - 20,(self.game.n + 1) * self.gridSize / 2 + 70))
        self.screen.blit(self.Surf, (0, 0))
        pygame.display.update()
        while True:
            mouseX, mouseY = self.click_point()
            if(mouseX >= (self.game.n + 1) * self.gridSize / 2 - 30 and mouseX <= (self.game.n + 1) * self.gridSize / 2 + 230):
                if(mouseY >= (self.game.n + 1) * self.gridSize / 2 - 60 and mouseY <= (self.game.n + 1) * self.gridSize / 2 - 10):
                    return -1
            if(mouseX >= (self.game.n + 1) * self.gridSize / 2 - 30 and mouseX <= (self.game.n + 1) * self.gridSize / 2 + 230):
                if(mouseY >= (self.game.n + 1) * self.gridSize / 2 + 60 and mouseY <= (self.game.n + 1) * self.gridSize / 2 + 110):
                    return 1

    def choose_turn(self):
        self.Surf.fill((150, 150, 150), ((self.game.n + 1) * self.gridSize / 2 - 30,(self.game.n + 1) * self.gridSize / 2 - 60, 230, 50))
        self.Surf.fill((150, 150, 150), ((self.game.n + 1) * self.gridSize / 2 - 30,(self.game.n + 1) * self.gridSize / 2 + 60, 230, 50))
        font = pygame.font.Font(None, 50)
        text = font.render('Black pieces', 1, (10, 10, 10))
        
        button = pygame.image.load('button.png')
        button = pygame.transform.scale(button, (260*int(self.gridSize/50), 80*int(self.gridSize/50)))
        background = pygame.image.load('playbackground.png')
        background = pygame.transform.scale(background, (self.gridSize * (self.game.n + 4),self.gridSize * (self.game.n + 2)))
        self.Surf.blit(background,(0,0))
        self.Surf.blit(button,((self.game.n + 1) * self.gridSize / 2 - 45*int(self.gridSize/50),(self.game.n + 1) * self.gridSize / 2 - 75*int(self.gridSize/50)))
        self.Surf.blit(button,((self.game.n + 1) * self.gridSize / 2 - 45*int(self.gridSize/50),(self.game.n + 1) * self.gridSize / 2 + 45*int(self.gridSize/50)))

        self.Surf.blit(text, ((self.game.n + 1) * self.gridSize / 2 - 20,(self.game.n + 1) * self.gridSize / 2 - 50))
        text = font.render('White pieces', 1, (10, 10, 10))
        self.Surf.blit(text, ((self.game.n + 1) * self.gridSize / 2 - 20,(self.game.n + 1) * self.gridSize / 2 + 70))
        self.screen.blit(self.Surf, (0, 0))
        pygame.display.update()
        while True:
            mouseX, mouseY = self.click_point()
            if(mouseX >= (self.game.n + 1) * self.gridSize / 2 - 30 and mouseX <= (self.game.n + 1) * self.gridSize / 2 + 230):
                if(mouseY >= (self.game.n + 1) * self.gridSize / 2 - 60 and mouseY <= (self.game.n + 1) * self.gridSize / 2 - 10):
                    return -1
            if(mouseX >= (self.game.n + 1) * self.gridSize / 2 - 30 and mouseX <= (self.game.n + 1) * self.gridSize / 2 + 230):
                if(mouseY >= (self.game.n + 1) * self.gridSize / 2 + 60 and mouseY <= (self.game.n + 1) * self.gridSize / 2 + 110):
                    return 1
            

    def refresh(self):
        self.Surf.fill((255, 255, 255))

        playbackground = pygame.image.load('playbackground.png')
        playbackground = pygame.transform.scale(playbackground, (self.gridSize * (self.game.n + 4),self.gridSize * (self.game.n + 2)))
        self.Surf.blit(playbackground, (0, 0))

        font = pygame.font.Font(None, 20)
        for digit in range(1, self.game.n + 1):
            text = font.render(str(digit), 1, (10, 10, 10))
            self.Surf.blit(text, (10, self.gridSize * digit - 3))
            text = font.render(chr(digit+96), 1, (10, 10, 10))
            self.Surf.blit(text, (self.gridSize * digit - 3, 10))

        for line in range(0 ,self.game.n + 1):#draw grids
            pygame.draw.line(self.Surf, (0, 0, 0), (25, self.gridSize * line + 25), (self.gridSize * (self.game.n) + 25, self.gridSize * line + 25), 2)
            pygame.draw.line(self.Surf, (0, 0, 0), (self.gridSize * line + 25, 25), (self.gridSize * line + 25, self.gridSize * (self.game.n) + 25), 2)
        
        

    def update_game(self, board):
        p1 = 0
        p2 = 0
        for i in range(self.game.n):
            for j in range(self.game.n):
                if(board[i][j] == self.t):
                    Xpiece = 25 + j * self.gridSize + 25
                    Ypiece = 25 + i * self.gridSize + 25
                    pygame.draw.circle(self.Surf, (0, 0, 0), (Xpiece, Ypiece),15, 5)
                    p1 += 1
                elif(board[i][j] == -self.t):
                    Xpiece = 25 + j * self.gridSize + 25
                    Ypiece = 25 + i * self.gridSize + 25
                    pygame.draw.circle(self.Surf, (0, 0, 0), (Xpiece, Ypiece),15, 0)
                    p2 += 1
        
        
        font = pygame.font.Font(None, 30)
        text = font.render('W - B : ' + str(p1) + ' - ' + str(p2), 1, (10, 10, 10))
        self.Surf.blit(text, (self.gridSize * (self.game.n + 1), 100))
        self.screen.blit(self.Surf, (0, 0))
        pygame.display.update()
                

    def click_point(self):
        pygame.event.clear()
        event = pygame.event.wait()
        while event.type != pygame.MOUSEBUTTONDOWN:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                exit()

        (mouseX, mouseY) = pygame.mouse.get_pos()
        return mouseX, mouseY

    def last_move(self, action):
        self.lastMove = action
        if(int(self.lastMove / self.game.n) < self.game.n and self.lastMove >= 0):
            
            pygame.draw.rect(self.Surf, (255, 0, 0), [25 + (self.lastMove % self.game.n) * self.gridSize, 25 + 
            (int(self.lastMove / self.game.n)) * self.gridSize, self.gridSize, self.gridSize], 3)
            font = pygame.font.Font(None, 30)
            text = font.render('point:' + str(int(self.lastMove / self.game.n) + 1) 
            + ' - ' + chr(self.lastMove % self.game.n + 97), 1, (10, 10, 10))
            self.Surf.blit(text, (self.gridSize * (self.game.n + 1), 200))
        else:
            font = pygame.font.Font(None, 30)
            #text = font.render('point:' + str(0) + ' - ' + str(0), 1, (10, 10, 10))
            text = font.render('point: pass', 1, (10, 10, 10))
            #text = font.render('point:' + str(int(self.lastMove / self.game.n) + 1) + ' - ' + chr(self.lastMove % self.game.n + 97), 1, (10, 10, 10))
            self.Surf.blit(text, (self.gridSize * (self.game.n + 1), 200))

    def undo_font(self):
        self.Surf.fill((150, 150, 150), (self.gridSize * (self.game.n + 1) + 20, 270, 60, 30))
        font = pygame.font.Font(None, 30)
        text = font.render('undo', 1, (10, 10, 10))
        self.Surf.blit(text,  (self.gridSize * (self.game.n + 1) + 25, 275))

    def winner_game(self, board):
        self.Surf.fill((255, 255, 255))
        self.refresh()
        self.last_move(self.lastMove)
        self.update_game(board)
        font = pygame.font.Font(None, 30)
        if(self.game.getGameEnded(board, 1)==self.t):
            text = font.render('White win', 1, (10, 10, 10))
        elif(self.game.getGameEnded(board, 1)==-self.t):
            text = font.render('Black win', 1, (10, 10, 10))
        else:
            text = font.render('Draw', 1, (10, 10, 10))
        self.Surf.blit(text, (self.gridSize * (self.game.n + 1), 300))
        self.screen.blit(self.Surf, (0, 0))
        pygame.display.update()
        self.savePic() 
        self.click_point()

    def play(self, board):
        
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        self.Surf.fill((255, 255, 255))
        self.refresh()
        self.undo_font()
        for i in range(len(valid)):
            if valid[i]:
                #print(int(i/self.game.n), int(i%self.game.n))
                self.Surf.fill((150, 150, 150), ((int(i%self.game.n) + 1) * self.gridSize - 15,
                 (int(i/self.game.n) + 1) * self.gridSize - 15, 30, 30))

        if valid[-1] == 1:
            font = pygame.font.Font(None, 30)
            text = font.render('pass', 1, (10, 10, 10))
            self.Surf.blit(text, (self.gridSize - 25, (self.game.n + 1) * self.gridSize -10))

        self.last_move(self.lastMove)
        self.update_game(board)
        
        while True:
            mouseX, mouseY = self.click_point()
            if(mouseX >= (self.gridSize * (self.game.n + 1) + 20) and mouseX <= (self.gridSize * (self.game.n + 1) + 80)):
                if(mouseY >= 270 and mouseY <= 300):
                    return -1
            mouseX = int((mouseX-25) / self.gridSize)
            mouseY = int((mouseY-25) / self.gridSize)
            a = self.game.n * mouseY + mouseX if mouseX!= -1 else self.game.n ** 2
            if a > self.game.n**2 or a < 0:
                print('Invalid')
                continue
                
            if valid[a]:
                break
            else:
                print('Invalid')
        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

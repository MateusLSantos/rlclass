#Atividade de RL com Q Learning

import time
import os
import numpy
import random
import pandas


#Classes da barra, bola e blocos
class Block:
    def __init__(self, x, y):
        self.active = True
        self.x = x
        self.y = y

    def destroy(self):
        self.active = False

class Bar:
    def __init__(self):
        self.x = int(field_x/2)

    #Movimentação pode ser -1, 0 ou 1
    def move(self, direction):
        self.x = self.x + direction
        if self.x > field_y:
            self.x = field_y
        if self.x < 0:
            self.x = 0

class Ball:
    def __init__(self):
        self.x = int(field_x/2)
        self.y = 1
        self.directionX = True #Vai iniciar para a direita
        self.directionY = True #Começa pra cima

    def changeX(self):
        self.directionX = not self.directionX

    def changeY(self):
        self.directionY = not self.directionY

    def move(self):
        if self.directionX:
            self.x += 1
        else:
            self.x -= 1
        if self.directionY:
            self.y += 1
        else:
            self.y -= 1

#Funções de Controle do Jogo


#Cria os blocos e os posiciona. Backspace é a quantidade de espaços atrás dos últimos blocos. 
def generateField(numBlocks, backspace):
    for n in range(numBlocks):
        new_block = Block(n%field_x, field_y - backspace - int(n/field_x))
        blocks.append(new_block)
    defineCheckBlockCollisionTime()
    createBlocksDict()

def defineCheckBlockCollisionTime():
    minimum = field_y
    for block in blocks:
        if block.y < minimum:
            minimum = block.y

    global checkCollisionWhen
    checkCollisionWhen = minimum-1;
        
def createBlocksDict():
    for x in range(field_x+1):
        for y in range(field_y+1):
            blocksDict[(x, y)] = None
    for block in blocks:
        blocksDict[(block.x, block.y)] = block

def checkBlockCollision():
    if ball.directionY:
        if blocksDict[(ball.x, ball.y+1)]:
            collide(blocksDict[(ball.x, ball.y+1)])
        else:
            if ball.directionX and blocksDict[(ball.x+1, ball.y+1)]:
                collide(blocksDict[(ball.x+1, ball.y+1)])
                ball.changeX()
            elif not ball.directionX and blocksDict[(ball.x-1, ball.y+1)]:
                collide(blocksDict[(ball.x-1, ball.y+1)])
                ball.changeX()
    else:
        if blocksDict[(ball.x, ball.y-1)]:
            collide(blocksDict[(ball.x, ball.y-1)])
        else:
            if ball.directionX and blocksDict[(ball.x+1, ball.y-1)]:
                collide(blocksDict[(ball.x+1, ball.y-1)])
                ball.changeX()
            elif not ball.directionX and blocksDict[(ball.x-1, ball.y-1)]:
                collide(blocksDict[(ball.x-1, ball.y-1)])
                ball.changeX()
                

def collide(block):
    block.destroy()
    blocks.remove(block)
    blocksDict[(block.x, block.y)] = None
    ball.changeY()
    global score;
    score += 10

def checkBarCollision():
    if not ball.directionY:
        if abs(ball.x - bar.x) > 1:
            pass
        elif ball.x == bar.x:
            ball.changeY()
        elif ball.directionX and bar.x > ball.x:
            ball.changeY()
            ball.changeX()
        elif not ball.directionX and bar.x < ball.x:
            ball.changeY()
            ball.changeX()
        else:
            pass

def checkFault():
    if ball.y == 0:
        print("Game Over!")
        global score
        score -= 100000
        return True
    else:
        return False

def collideWithWall():
    ball.changeX()

def collideWithRoof():
    ball.changeY()

def checkCollisions():
    if ball.y == 1:
        checkBarCollision()
    checkFault()
    if ball.directionX and ball.x == field_x-1:
        collideWithWall()
    if not ball.directionX and ball.x == 0:
        collideWithWall()
    if ball.y == field_y and ball.directionY:
        collideWithRoof()
    checkBlockCollision()
    if ball.y == 1:
        checkBarCollision()
    checkFault()
    if ball.directionX and ball.x == field_x-1:
        collideWithWall()
    if not ball.directionX and ball.x == 0:
        collideWithWall()
    if ball.y == field_y and ball.directionY:
        collideWithRoof()
    

def nextRound():
    global score
    score -= 1
    if not checkFault():
        checkCollisions()
        ball.move()
        return True
    else:
        return False

def printGame(): #FIQUE MUITO LONGE DAQUI, NÃO TENTE ENTENDER
    #Linha superior
    rows = []
    for num in range(field_y+1):
        newRow = '|' + " "*field_x + '|'
        rows.append(newRow)
    for block in blocks:
        rows[block.y] = rows[block.y][:block.x+1] + 'X' + rows[block.y][block.x+2:]
    rows[0] = rows[0][:bar.x+1] + "_" + rows[0][bar.x+2:]
    rows[ball.y] = rows[ball.y][:ball.x+1] + "o" + rows[ball.y][ball.x+2:]

    print("_"*(field_x+2))
    for row in rows[::-1]:
        print(row)
    print("_"*(field_x+2))




'''
#######################################################################
q = {}
def qTable():
    for x1 in range(field_x+1):
        for y1 in range(field_y+1):
            for x2 in range(field_x+1):
                q[(x1, y1, x2, 'up', 'right')] = [0, 0, 0]

qTable()
'''

class QLearn:
    def __init__(self, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = [-1, 0, 1]

    def getQ(self, state, action):
        return self.q[state][self.actions.index(action)]

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        ind = self.actions.index(action)
        oldv = self.q[state][ind]
        self.q[state][ind] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]
        if return_q: 
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def buildStates(self, field_x, field_y):
        for x1 in range(field_x+1):
            for y1 in range(field_y+1):
                for x2 in range(field_x+1):
                    self.q[(x1, y1, x2)] = [0, 0, 0]


for n in range(200):
    #Variáveis do Campo
    field_x = 20;
    field_y = 10;
    score = 0;

    ball = Ball()
    bar = Bar()
    blocks = []
    blocksDict = {}
    checkCollisionWhen = 0;    #Altura a partir da qual o programa checará se haverá colisão com algum bloco

    
    generateField(80,2)
    qL = QLearn(0.1, 0.3, 0.3)
    qL.buildStates(field_x, field_y)



    for n in range(500):
        if len(blocks) == 0:
            print("Parabéns, você venceu!")
            break
            score += 1000
        time.sleep(0.2)
        print('\n'*100)
        printGame()

        state = (ball.x, ball.y, bar.x)
        action = qL.chooseAction(state)
        #print(action)
        bar.move(action)
        old_score = score
        nextState = (0, 0, 0);
        
        if ball.directionX and ball.directionY:
            nextState = (ball.x + 1, ball.y + 1, bar.x)
        elif ball.directionX and not ball.directionY:
            nextState = (ball.x + 1, ball.y - 1, bar.x)
        elif not ball.directionX and ball.directionY:
            nextState = (ball.x - 1, ball.y + 1, bar.x)
        else:
            nextState = (ball.x - 1, ball.y - 1, bar.x)

        tempNextState = [0,0,0]

        if nextState[0] > field_x:
            tempNextState[0] = field_x
        elif nextState[0] < 0:
            tempNextState[0] = 0
        else:
            tempNextState[0] = nextState[0]
        
        if nextState[1] > field_y:
            tempNextState[1] = field_y
        elif nextState[1] < 0:
            tempNextState[1] = 0
        else:
            tempNextState[1] = nextState[1]

        if nextState[2] > field_x:
            tempNextState[2] = field_x
        elif nextState[2] < 0:
            tempNextState[2] = 0
        else:
            tempNextState[2] = nextState[2]

        nextState = tuple(tempNextState)


        if not nextRound():
            reward = -200
            qL.learn(state, action, reward, nextState)
            break
        else:
            reward = score - old_score
            qL.learn(state, action, reward, nextState)
            


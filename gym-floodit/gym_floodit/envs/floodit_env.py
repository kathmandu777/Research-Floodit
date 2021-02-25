# FloodItをOpenAIGymのAPI形式に合わせて作成し登録する
# pygameによって作成されたinkspillを改良(https://github.com/asweigart/making-games-with-python-and-pygame/blob/master/inkspill/inkspill.py)

"""
設計：
    環境状態(observation)=self.mainBoard

関数定義:
    def reset(self):
        return observation

    def step(self, action):
        return observation, reward, done, info

*詳しくはREADME.md参照
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random
import copy
import numpy as np
import pygame
import os
from pygame.locals import *


# 各種グローバル変数
SMALLBOXSIZE = 60  # size is in pixels
MEDIUMBOXSIZE = 20
LARGEBOXSIZE = 11

SMALLBOARDSIZE = 6  # size is in boxes
MEDIUMBOARDSIZE = 17
LARGEBOARDSIZE = 30

SMALLMAXLIFE = 10  # number of turns
MEDIUMMAXLIFE = 30
LARGEMAXLIFE = 64

FPS = 60
WINDOWWIDTH = 640
WINDOWHEIGHT = 480
PALETTEGAPSIZE = 10
PALETTESIZE = 45
EASY = 0   # arbitrary but unique value
MEDIUM = 1  # arbitrary but unique value
HARD = 2   # arbitrary but unique value


#          R    G    B
WHITE = (255, 255, 255)
DARKGRAY = (70,  70,  70)
BLACK = (0,   0,   0)
RED = (255,   0,   0)
GREEN = (0, 255,   0)
BLUE = (0,   0, 255)
YELLOW = (255, 255,   0)
ORANGE = (255, 128,   0)
PURPLE = (255,   0, 255)

# The first color in each scheme is the background color, the next six are the palette colors.
COLORSCHEMES = (((150, 200, 255), RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE),
                ((0, 155, 104),  (97, 215, 164),  (228, 0, 69),  (0, 125, 50),
                 (204, 246, 0),   (148, 0, 45),    (241, 109, 149)),
                ((195, 179, 0),  (255, 239, 115), (255, 226, 0), (147, 3, 167),
                 (24, 38, 176),   (166, 147, 0),   (197, 97, 211)),
                ((85, 0, 0),     (155, 39, 102),  (0, 201, 13),  (255, 118, 0),
                 (206, 0, 113),   (0, 130, 9),     (255, 180, 115)),
                ((191, 159, 64), (183, 182, 208), (4, 31, 183),  (167, 184, 45),
                 (122, 128, 212), (37, 204, 7),    (88, 155, 213)),
                ((200, 33, 205), (116, 252, 185), (68, 56, 56),  (52, 238, 83),  (23, 149, 195),  (222, 157, 227), (212, 86, 185)))

# 色の設定 ※気にしなくてよい エラー表示専用
"""
for i in range(len(COLORSCHEMES)):
    assert len(
        COLORSCHEMES[i]) == 7, 'Color scheme %s does not have exactly 7 colors.' % (i)
bgColor = COLORSCHEMES[0][0]
paletteColors = COLORSCHEMES[0][1:]
"""
bgColor = COLORSCHEMES[0][0]
paletteColors = COLORSCHEMES[0][1:]


class FlooditEnv(gym.Env):
    def __init__(self):
        super(FlooditEnv, self).__init__()

        # アクションの数の設定
        self.action_space = gym.spaces.Discrete(6)

        # 状態空間の設定, 観測空間(state)の次元
        # (e.g.)状態が3つの時で、それぞれの状態が正規化されている場合、LOW=[0,0,0]、HIGH=[1,1,1]
        MAP = np.array([[0] * 6 for _ in range(6)])
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=MAP.shape)

        self.isFirstRender = True

        self.reset()

    def reset(self):
        # シミュレータの初期化処理

        # デフォルトのゲームの難易度
        self.difficulty = EASY
        self.maxLife = SMALLMAXLIFE
        self.boardWidth = SMALLBOARDSIZE
        self.boardHeight = SMALLBOARDSIZE
        self.boxSize = SMALLBOXSIZE

        # 盤面とライフの初期化
        self.mainBoard = self.generateRandomBoard(
            self.boardWidth, self.boardHeight, self.difficulty)
        self.life = self.maxLife
        self.lastAction = None
        self.continueing_square = 0

        if (not self.isFirstRender):
            self.render()

        return self.mainBoard

    def step(self, action):
        isWon = False
        isLose = False
        done = False
        reward = 0.0

        changed_square = self.flood_and_count(self.mainBoard, action)
        if (changed_square == 0):
            reward = -0.5
            if (not self.isFirstRender):  # レンダリング中は同じ手を選んでも無視する
                pass
            else:
                done = True
        else:
            reward = changed_square/36
        self.life -= 1

        # 勝利or敗北時の描画
        if self.hasWon(self.mainBoard):
            reward = 1
            done = True
            isWon = True
            if (not self.isFirstRender):
                for i in range(2):
                    self.flashBorderAnimation(WHITE, self.mainBoard)
                pygame.time.wait(200)
        elif self.life == 0:
            reward = -1
            done = True
            isLose = True
            if (not self.isFirstRender):
                for i in range(2):
                    self.flashBorderAnimation(BLACK, self.mainBoard)
                pygame.time.wait(200)

        info = {"isWon": isWon, "isLose": isLose,
                "changed_square": changed_square, "life": self.life}
        #info = {"isWon": isWon, "isLose": isLose,"changed_square": changed_square, "life": self.life, "isQuit": self.checkReset()}

        return np.array(self.mainBoard), reward, done, info

    def render(self, mode='human', close=False):
        # シミュレータ環境の描画処理
        if mode != 'human':
            raise NotImplementedError()

        if (self.isFirstRender):
            self.isFirstRender = False
            # pygame の描画関連
            global FPSCLOCK, LOGOIMAGE, SPOTIMAGE, SETTINGSIMAGE, SETTINGSBUTTONIMAGE, RESETBUTTONIMAGE

            pygame.init()
            FPSCLOCK = pygame.time.Clock()
            self.DISPLAYSURF = pygame.display.set_mode(
                (WINDOWWIDTH, WINDOWHEIGHT))

            # 画像のロード
            current_path = os.path.dirname(__file__)
            LOGOIMAGE = pygame.image.load(os.path.join(
                current_path, 'inkspilllogo.png'))
            SPOTIMAGE = pygame.image.load(
                os.path.join(current_path, 'inkspillspot.png'))
            SETTINGSIMAGE = pygame.image.load(os.path.join(current_path,
                                                           'inkspillsettings.png'))
            SETTINGSBUTTONIMAGE = pygame.image.load(os.path.join(current_path,
                                                                 'inkspillsettingsbutton.png'))
            RESETBUTTONIMAGE = pygame.image.load(os.path.join(current_path,
                                                              'inkspillresetbutton.png'))

            # キャプションとマウスの位置の設定
            pygame.display.set_caption('Ink Spill for machine-learning')

        pygame.event.get()  # 応答なし回避のため

        self.DISPLAYSURF.fill(bgColor)
        self.drawLogoAndButtons()
        self.drawBoard(self.mainBoard)
        self.drawLifeMeter(self.life)
        self.drawPalettes()
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        pygame.time.wait(500)


#### don't care ##############################
    # FloodItの処理


    def flood_and_count(self, board, action, animationSpeed=25):  # マスを選択された色に変換し、変更マス数を返す
        origBoard = copy.deepcopy(board)

        # 先頭の色が連続しているマスの数を数える
        self.continueing_square = 0
        self.countSquare(copy.deepcopy(board), board[0][0], 0, 0)
        self.before_continueing_square = self.continueing_square

        # マスを選択された色に変換
        self.floodFill(board, board[0][0], action, 0, 0)

        # 先頭の色が連続しているマスの数を数える
        self.continueing_square = 0
        self.countSquare(copy.deepcopy(board), board[0][0], 0, 0)
        self.after_continueing_square = self.continueing_square

        return self.after_continueing_square - self.before_continueing_square

    def floodFill(self, board, oldColor, newColor, x, y):  # マスの色を選択された色に変更
        if oldColor == newColor or board[x][y] != oldColor:
            # 先頭のマスの色と選択された色が同じ or 検索しているマスが先頭のマスと違う色だったら
            return

        board[x][y] = newColor  # change the color of the current box

        # Make the recursive call for any neighboring boxes:
        if x > 0:
            self.floodFill(board, oldColor, newColor, x - 1, y)
        if x < self.boardWidth - 1:
            self.floodFill(board, oldColor, newColor, x + 1, y)
        if y > 0:
            self.floodFill(board, oldColor, newColor, x, y - 1)
        if y < self.boardHeight - 1:
            self.floodFill(board, oldColor, newColor, x, y + 1)

    def countSquare(self, board, originColor, x, y):  # 先頭の色と連続した色のマスのカウント
        if (board[x][y] != originColor or board[x][y] == -1):
            # マスの色が原点のマスの色と異なる or 探索済み
            return

        board[x][y] = -1  # 探索済みの印

        self.continueing_square += 1

        if x > 0:
            self.countSquare(board, originColor, x - 1, y)
        if x < self.boardWidth - 1:
            self.countSquare(board, originColor, x + 1, y)
        if y > 0:
            self.countSquare(board, originColor, x, y - 1)
        if y < self.boardHeight - 1:
            self.countSquare(board, originColor, x, y + 1)

    def hasWon(self, board):  # 勝ったかどうか＝＝全マス同じ色か
        for x in range(self.boardWidth):
            for y in range(self.boardHeight):
                if board[x][y] != board[0][0]:
                    return False  # found a different color, player has not won
        return True

    def generateRandomBoard(self, width, height, difficulty=MEDIUM):  # 盤面の初期化
        # Creates a board data structure with random colors for each box.
        board = []
        for x in range(width):
            column = []
            for y in range(height):
                column.append(random.randint(0, len(paletteColors) - 1))
            board.append(column)

        # Make board easier by setting some boxes to same color as a neighbor.

        # Determine how many boxes to change.
        if difficulty == EASY:
            if self.boxSize == SMALLBOXSIZE:
                boxesToChange = 100
            else:
                boxesToChange = 1500
        elif difficulty == MEDIUM:
            if self.boxSize == SMALLBOXSIZE:
                boxesToChange = 5
            else:
                boxesToChange = 200
        else:
            boxesToChange = 0

        # Change neighbor's colors:
        for i in range(boxesToChange):
            # Randomly choose a box whose color to copy
            x = random.randint(1, width-2)
            y = random.randint(1, height-2)

            # Randomly choose neighbors to change.
            direction = random.randint(0, 3)
            if direction == 0:  # change left and up neighbor
                board[x-1][y] = board[x][y]
                board[x][y-1] = board[x][y]
            elif direction == 1:  # change right and down neighbor
                board[x+1][y] = board[x][y]
                board[x][y+1] = board[x][y]
            elif direction == 2:  # change right and up neighbor
                board[x][y-1] = board[x][y]
                board[x+1][y] = board[x][y]
            else:  # change left and down neighbor
                board[x][y+1] = board[x][y]
                board[x-1][y] = board[x][y]
        return board

    def quit(self):
        pygame.quit()

    def checkReset(self):
        for event in pygame.event.get():  # event handling loop
            if event.type == MOUSEBUTTONUP:  # マウスが離されたら
                mousex, mousey = event.pos  # マウスが離れた場所
                if pygame.Rect(WINDOWWIDTH - RESETBUTTONIMAGE.get_width(),
                               WINDOWHEIGHT - SETTINGSBUTTONIMAGE.get_height() - RESETBUTTONIMAGE.get_height(),
                               RESETBUTTONIMAGE.get_width(),
                               RESETBUTTONIMAGE.get_height()).collidepoint(mousex, mousey):  # マウスが離れた場所がリセットボタンのところだったら
                    return True
        return False


##### rendering ##########

    def flashBorderAnimation(self, color, board, animationSpeed=30):
        origSurf = self.DISPLAYSURF.copy()
        flashSurf = pygame.Surface(self.DISPLAYSURF.get_size())
        flashSurf = flashSurf.convert_alpha()
        for start, end, step in ((0, 256, 1), (255, 0, -1)):
            # the first iteration on the outer loop will set the inner loop
            # to have transparency go from 0 to 255, the second iteration will
            # have it go from 255 to 0. This is the "flash".
            for transparency in range(start, end, animationSpeed * step):
                self.DISPLAYSURF.blit(origSurf, (0, 0))
                r, g, b = color
                flashSurf.fill((r, g, b, transparency))
                self.DISPLAYSURF.blit(flashSurf, (0, 0))
                # draw board ON TOP OF the transparency layer
                self.drawBoard(board)
                pygame.display.update()
                FPSCLOCK.tick(FPS)
        self.DISPLAYSURF.blit(origSurf, (0, 0))  # redraw the original surface

    def drawLogoAndButtons(self):
        # draw the Ink Spill logo and Settings and Reset buttons.
        self.DISPLAYSURF.blit(
            LOGOIMAGE, (WINDOWWIDTH - LOGOIMAGE.get_width(), 0))
        self.DISPLAYSURF.blit(SETTINGSBUTTONIMAGE, (WINDOWWIDTH - SETTINGSBUTTONIMAGE.get_width(),
                                                    WINDOWHEIGHT - SETTINGSBUTTONIMAGE.get_height()))
        self.DISPLAYSURF.blit(RESETBUTTONIMAGE, (WINDOWWIDTH - RESETBUTTONIMAGE.get_width(),
                                                 WINDOWHEIGHT - SETTINGSBUTTONIMAGE.get_height() - RESETBUTTONIMAGE.get_height()))

    def drawBoard(self, board, transparency=255):  # 盤面の描画
        # The colored squares are drawn to a temporary surface which is then
        # drawn to the self.DISPLAYSURF surface. This is done so we can draw the
        # squares with transparency on top of self.DISPLAYSURF as it currently is.
        tempSurf = pygame.Surface(self.DISPLAYSURF.get_size())
        tempSurf = tempSurf.convert_alpha()
        tempSurf.fill((0, 0, 0, 0))

        for x in range(self.boardWidth):
            for y in range(self.boardHeight):
                left, top = self.leftTopPixelCoordOfBox(x, y)
                r, g, b = paletteColors[board[x][y]]
                pygame.draw.rect(tempSurf, (r, g, b, transparency),
                                 (left, top, self.boxSize, self.boxSize))
        left, top = self.leftTopPixelCoordOfBox(0, 0)
        pygame.draw.rect(tempSurf, BLACK, (left-1, top-1, self.boxSize *
                                           self.boardWidth + 1, self.boxSize * self.boardHeight + 1), 1)
        self.DISPLAYSURF.blit(tempSurf, (0, 0))

    def drawPalettes(self):  # パレットの描画
        # Draws the six color palettes at the bottom of the screen.
        numColors = len(paletteColors)
        xmargin = int((WINDOWWIDTH - ((PALETTESIZE * numColors) +
                                      (PALETTEGAPSIZE * (numColors - 1)))) / 2)
        for i in range(numColors):
            left = xmargin + (i * PALETTESIZE) + (i * PALETTEGAPSIZE)
            top = WINDOWHEIGHT - PALETTESIZE - 10
            pygame.draw.rect(
                self.DISPLAYSURF, paletteColors[i], (left, top, PALETTESIZE, PALETTESIZE))
            pygame.draw.rect(self.DISPLAYSURF, bgColor,   (left + 2,
                                                           top + 2, PALETTESIZE - 4, PALETTESIZE - 4), 2)

    def drawLifeMeter(self, currentLife):  # 引数として与えられたライフを描画
        lifeBoxSize = int((WINDOWHEIGHT - 40) / self.maxLife)

        # Draw background color of life meter.
        pygame.draw.rect(self.DISPLAYSURF, bgColor,
                         (20, 20, 20, 20 + (self.maxLife * lifeBoxSize)))

        for i in range(self.maxLife):
            if currentLife >= (self.maxLife - i):  # draw a solid red box
                pygame.draw.rect(self.DISPLAYSURF, RED, (20, 20 +
                                                         (i * lifeBoxSize), 20, lifeBoxSize))
            pygame.draw.rect(self.DISPLAYSURF, WHITE, (20, 20 + (i *
                                                                 lifeBoxSize), 20, lifeBoxSize), 1)  # draw white outline

    def leftTopPixelCoordOfBox(self, boxx, boxy):  # マージンの設定
        # Returns the x and y of the left-topmost pixel of the xth & yth box.
        xmargin = int((WINDOWWIDTH - (self.boardWidth * self.boxSize)) / 2)
        ymargin = int((WINDOWHEIGHT - (self.boardHeight * self.boxSize)) / 2)
        return (boxx * self.boxSize + xmargin, boxy * self.boxSize + ymargin)


"""
参考文献(gym環境の構築方法)：
ドキュメント=https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
    https://qiita.com/inoory/items/e63ade6f21766c7c2393
    https://qiita.com/harmegiddo/items/4226df13139d6ba34018
    https://qiita.com/ohtaman/items/edcb3b0a2ff9d48a7def
    https://moriokalab.com/news/57
    https://qiita.com/inoory/items/e63ade6f21766c7c2393
    https://ailog.site/2020/05/06/0506/
    https://note.com/npaka/n/nc90b305f4ca0
    https://qiita.com/pocokhc/items/a8120b0abd5941dd7a9f
    https://kagglenote.com/ml-tips/my-environment-with-gym/
    https://qiita.com/YudaiSadakuni/items/09e111bec5de257a5782
    https://kagglenote.com/ml-tips/my-environment-with-gym/
    https://omedstu.jimdofree.com/2018/05/30/openai-gym%E3%81%A7%E3%82%AA%E3%83%AA%E3%82%B8%E3%83%8A%E3%83%AB%E3%81%AE%E7%92%B0%E5%A2%83%E3%82%92%E4%BD%9C%E3%82%8B/
"""

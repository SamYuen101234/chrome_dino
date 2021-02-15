#Intialize log structures from file if exists else create new
from conf import *
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image
from io import BytesIO
import base64

loss_df = pd.read_csv(args.loss_file_path) if os.path.isfile(args.loss_file_path) else pd.DataFrame(columns =['loss'])
scores_df = pd.read_csv(args.scores_file_path) if os.path.isfile(args.loss_file_path) else pd.DataFrame(columns = ['scores'])
actions_df = pd.read_csv(args.actions_file_path) if os.path.isfile(args.actions_file_path) else pd.DataFrame(columns = ['actions'])
q_values_df =pd.read_csv(args.actions_file_path) if os.path.isfile(args.q_value_file_path) else pd.DataFrame(columns = ['qvalues'])

class DinoAgent:
    def __init__(self,game): #takes game as input for taking actions
        self._game = game
        self.jump() #to start the game, we need to jump once
    def is_running(self):
        return self._game.get_playing()
    def is_crashed(self):
        return self._game.get_crashed()
    def jump(self):
        self._game.press_up()
    def duck(self):
        self._game.press_down()

class Game_sate:
    def __init__(self,agent,game):
        self._agent = agent # the dino agent
        self._game = game # env
        self._display = show_img() #display the processed image on screen using openCV, implemented using python coroutine 
        self._display.__next__() # initiliaze the display coroutine 
    def get_state(self,actions):
        actions_df.loc[len(actions_df)] = actions[1] # storing actions in a dataframe
        score = self._game.get_score() 
        reward = 0.1
        is_over = False #game over
        if actions[1] == 1:
            self._agent.jump()
        image = grab_screen(self._game._driver) 
        self._display.send(image) #display the image on screen
        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score # log the score when game is over
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over #return the Experience tuple

def show_img(graphs = False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
        imS = cv2.resize(screen, (800, 400)) 
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break

def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to Grey Scale
    image = image[:300, :500] #Crop Region of Interest(ROI)
    image = cv2.resize(image, (80,80))
    return  image

def grab_screen(_driver):
    image_b64 = _driver.execute_script(args.getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_img(screen)#processing image as required
    return image
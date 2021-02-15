import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch import optim
import pickle
import time
from collections import deque
import random
from conf import *

''' 
main training module
Parameters:
* model => Pytorch Model to be trained
* game_state => Game State module with access to game environment and dino
* observe => flag to indicate wherther the model is to be trained(weight updates), else just play
'''

loss_df = pd.read_csv(args.loss_file_path) if os.path.isfile(args.loss_file_path) else pd.DataFrame(columns =['loss'])
scores_df = pd.read_csv(args.scores_file_path) if os.path.isfile(args.loss_file_path) else pd.DataFrame(columns = ['scores'])
actions_df = pd.read_csv(args.actions_file_path) if os.path.isfile(args.actions_file_path) else pd.DataFrame(columns = ['actions'])
q_values_df =pd.read_csv(args.actions_file_path) if os.path.isfile(args.q_value_file_path) else pd.DataFrame(columns = ['qvalues'])

class trainNetwork:
    def __init__(self,model,game_state,observe=False, device=torch.device('cpu')):
        self.model = model
        self.game_state = game_state
        self.observe = observe
        self.device = device
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss_fn = nn.MSELoss() # loss function
        init_cache()

    # train function
    def start(self):
        last_time = time.time()
        # store the previous observations in replay memory
        D = load_obj("D") #load from file system
        # get the first state by doing nothing
        do_nothing = np.zeros(args.ACTIONS)
        do_nothing[0] =1 # 0 => do nothing, #1=> jump , which means [1, 0] jump first

        # x_t: image (80x80), r_0: first reward, terminal: gameover or not
        x_t, r_0, terminal = self.game_state.get_state(do_nothing) # get next step after performing the action

        # four frames are stack as one input (80x80x4)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
        s_t = s_t.reshape(1, s_t.shape[2], s_t.shape[0], s_t.shape[1])  # 1*4*80*80 to fit the model input requirements
        # turn numpy array to torch tensor
        s_t = torch.from_numpy(s_t).float()
        initial_state = s_t 

        if self.observe :
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = args.FINAL_EPSILON
            #print ("Now we load weight")
            #model.load_weights("model.h5")
            #adam = Adam(lr=LEARNING_RATE)
            #model.compile(loss='mse',optimizer=adam)
            #print ("Weight load successfully")    
        else:                       #We go to training mode
            OBSERVE = args.OBSERVATION
            epsilon = load_obj("epsilon")
            #model.load_weights("model.h5")
            #adam = Adam(lr=LEARNING_RATE)
            #model.compile(loss='mse',optimizer=adam)

        t = load_obj("time") # resume from the previous time step stored in file system
        while (True): #endless running

            loss = torch.zeros((1))
            Q_sa = torch.zeros((1,2))
            action_index = 0
            r_t = 0 #reward at t
            a_t = np.zeros([args.ACTIONS]) # action at t

            #choose an action epsilon greedy
            if t % args.FRAME_PER_ACTION == 0: #parameter to skip frames for actions
                if  random.random() <= epsilon: #randomly explore an action
                    print("----------Random Action----------")
                    action_index = random.randrange(args.ACTIONS)   # pick a action randomly
                    a_t[action_index] = 1
                else: # predict the output
                    q = self.model(s_t)       #input a stack of 4 images, get the prediction
                    max_Q = torch.argmax(q)         # chosing index with maximum q value
                    action_index = max_Q 
                    a_t[action_index] = 1        # o=> do nothing, 1=> jump
            
            #We reduced the epsilon (exploration parameter) gradually
            if epsilon > args.FINAL_EPSILON and t > OBSERVE:
                epsilon -= (args.INITIAL_EPSILON - args.FINAL_EPSILON) / args.EXPLORE 
            
            # x_t1: image(80*80), r_t: reward, terminal: gameover or not
            x_t1, r_t, terminal = self.game_state.get_state(a_t) # let the agent process the action
            #print('fps: {0}'.format(1 / (time.time()-last_time))) # helpful for measuring frame rate(FPS)
            last_time = time.time()
            x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1]) #1x1x80x80
            s_t1 = np.append(x_t1, s_t[:, :3, :, :], axis=1) # append the new image to input stack and remove the first one
            s_t1 = torch.from_numpy(s_t1)

            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal)) # all related info in time t
            if len(D) > args.REPLAY_MEMORY:
                D.popleft() # remove the first item in the queue
            
            #only train if done observing (number of iterations)
            if t > OBSERVE: 
                #sample a minibatch to train on
                minibatch = random.sample(D, args.BATCH)
                inputs = torch.zeros((args.BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #(16, 4, 80, 80) in pytorch
                targets = torch.zeros((inputs.shape[0], args.ACTIONS))  # (16, 2), actions for each batch
                #Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]   # 4D stack of images
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]   #reward at state_t due to action_t
                    state_t1 = minibatch[i][3]  #next state
                    terminal = minibatch[i][4]   #wheather the agent died or survided due the action
                    inputs[i:i + 1] = state_t    

                    targets[i] = self.model(state_t.to(self.device))  # predicted q values
                    Q_sa = self.model(state_t1.to(self.device))      #predict q values for next step
                    
                    if terminal:
                        targets[i, action_t] = reward_t # if terminated, only equals reward
                    else:
                        targets[i, action_t] = reward_t + args.GAMMA * torch.max(Q_sa)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs.to(self.device), targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            s_t = initial_state if terminal else s_t1  # if gameover, reset the state
            t += 1

            # save progress every 1000 iterations
            if t % 1000 == 0:
                print("Now we save model")
                self.game_state._game.pause() #pause game while saving to filesystem
                torch.save(self.model.state_dict(), 'model.pt')
                save_obj(D,"D") #saving episodes
                save_obj(t,"time") #caching time steps
                save_obj(epsilon,"epsilon") #cache epsilon to avoid repeated randomness in actions
                loss_df.to_csv("./objects/loss_df.csv",index=False)
                scores_df.to_csv("./objects/scores_df.csv",index=False)
                actions_df.to_csv("./objects/actions_df.csv",index=False)
                q_values_df.to_csv(args.q_value_file_path,index=False)
                self.game_state._game.resume()
            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + args.EXPLORE:
                state = "explore"
            else:
                state = "train"

            print('TIMESTEP: {}, STATE: {}, EPSILON: {:.6f}, ACTION: {}, REWARD: {}, Q_MAX: {:.4f}, Loss: {:.4f}'
                          .format(t, state, epsilon, action_index, r_t, torch.max(Q_sa).item(), loss.item()))

        print("Episode finished!")
        print("************************")



# training variables saved as checkpoints to filesystem to resume training from the same step
def init_cache():
    """initial variable caching, done only once"""
    save_obj(args.INITIAL_EPSILON,"epsilon")
    t = 0
    save_obj(t,"time")
    D = deque()
    save_obj(D,"D")

def save_obj(obj, name):
    with open('objects/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
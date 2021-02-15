from Game import *
from setup import *
from model import CNN_Model
from conf import *
from train import trainNetwork

import torch



if __name__ == "__main__":
    # choosing cpu/cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device:",device)

    # open chome to set up a game env
    game = Game()
    # set the agent with different and intilize/start the game
    dino = DinoAgent(game)
    game_state = Game_sate(dino,game)
    model = CNN_Model()    # return resnet18
    #model.achitect_summary((4, 80, 80))

    try:
        process = trainNetwork(model.to(device),game_state,observe=False,device=device)
        # train function
        process.start()
    except StopIteration:
        game.end()

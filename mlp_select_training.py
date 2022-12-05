from asyncore import write
from copy import deepcopy
import os
import random
from datetime import datetime
from tabnanny import verbose

import gym
import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tensorflow as tf

from rrc_2022_datasets.sim_env import SimTriFingerCubeEnv

device = torch.device("cuda")

pre_action_idx = list(range(24, 33))  # lift: 24~33; push: 3~12
goal_idx = list(range(33, 57)) # lift: 33~57; push: 12~15
pose_idx = list(range(57, 120)) + list(range(121, 139))  # lift: 57~120, 121~139, push: 15~78, 79~97
selected_idx = pre_action_idx + goal_idx + pose_idx

def data_prepare():
    h5path = os.path.expanduser("~/.rrc_2022_datasets/trifinger-cube-lift-real-mixed-v0.hdf5")
    # push real expert: trifinger-cube-push-real-expert-v0.hdf5;
    # push real mixed: trifinger-cube-push-real-mixed-v0.hdf5;
    # lift real expert: trifinger-cube-lift-real-expert-v0.hdf5
    # lift real mixed: trifinger-cube-lift-real-mixed-v0.hdf5
    data_dict = {}
    with h5py.File(h5path, "r") as dataset_file:
        for k in tqdm(dataset_file.keys(), desc="Loading datafile"):
            data_dict[k] = dataset_file[k][:]

    # push sim
    # k = {'episode_length': 750, 'difficulty': 1, 'keypoint_obs': True, 'obs_action_delay': 10}
    # push real
    # k = {'episode_length': 750, 'difficulty': 1, 'keypoint_obs': True, 'obs_action_delay': 10, 'visualization': True}

    # lift real
    k = {'episode_length': 1500, 'difficulty': 4, 'keypoint_obs': True, 'obs_action_delay': 2, 'visualization': True}


    sim_env = SimTriFingerCubeEnv(**k)

    _orig_obs_space = sim_env.observation_space
    orig_flat_obs_space = gym.spaces.flatten_space(_orig_obs_space)

    selected_rows = data_dict["observations"][:, 58] < 0.5  # lift: 58, push: 16
    data_dict["observations"] = data_dict["observations"][selected_rows, :]
    data_dict["actions"] = data_dict["actions"][selected_rows, :]

    data_dict["observations"] = data_dict["observations"].clip(
        min=orig_flat_obs_space.low,
        max=orig_flat_obs_space.high,
        dtype=orig_flat_obs_space.dtype,
    )

    data_dict["observations"] = data_dict["observations"][:, selected_idx]

    data_dict["observations"] = data_dict["observations"].astype(np.float32)
    data_dict["actions"] = data_dict["actions"].astype(np.float32)

    train_num = int(0.99 * len(data_dict['observations']))
    idx = list(range(len(data_dict['observations'])))
    random.shuffle(idx)

    train_obs, test_obs = data_dict["observations"][idx[0: train_num]], data_dict["observations"][idx[train_num: ]]
    train_action, test_action = data_dict["actions"][idx[0: train_num]], data_dict["actions"][idx[train_num: ]]


    return (train_obs, train_action), (test_obs, test_action)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self._obs = data[0]
        self._action = data[1]

    def __len__(self):
        return len(self._obs)-1

    def __getitem__(self, idx):
        return self._obs[idx], self._action[idx]


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(81 + 9 + 24, 512)  # lift: 81 + 9 + 24; push: 81 + 9 + 3
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 9)
    
    def forward(self, obs): 
        emb = F.relu(self.fc1(obs))
        emb = F.relu(self.fc2(emb))
        emb = F.relu(self.fc3(emb))

        action = torch.tanh(self.fc4(emb)) * 0.397

        return action


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    tot_loss = 0
    for batch_idx, (obs, action) in enumerate(train_loader):
        obs, action = obs.to(device), action.to(device)
        optimizer.zero_grad()
        out_action = model(obs)
        loss = F.l1_loss(out_action, action)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, tot_loss / (batch_idx+1)))
    return tot_loss / (batch_idx+1)


def test(model, device, test_loader):
    model.eval()
    action_loss = 0
    with torch.no_grad():
        for batch_idx,  (obs, action) in enumerate(test_loader):
            obs, action = obs.to(device), action.to(device)
            out_action = model(obs)
            action_loss += F.l1_loss(out_action, action, reduction="sum").item()

    action_loss /= len(test_loader.dataset) * 9  # * 5000

    print('Test set: action loss: {:.6f}\n'.format(action_loss))
    return action_loss


def main():
    train_data, test_data = data_prepare()
    train_dataset = Dataset(train_data)
    test_dataset = Dataset(test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=6)

    model = Network().to(device)
    # model = torch.jit.load("model/20220811174139/action_100.pt")
    optimizer = optim.Adam(model.parameters(), 0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, cooldown=1, min_lr=1e-5, verbose=True)

    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = "model/" + current_time
    os.makedirs(model_dir)

    # model_dir = "model/20220811174139"
    for epoch in range(1, 21):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step(test_loss)
        with open("{}/log.txt".format(model_dir), "a") as f:
            f.write("epoch: {}, train loss: {}, test loss {}, lr {}\n".format(epoch, train_loss, test_loss, optimizer.param_groups[0]['lr']))

        model_scripted = torch.jit.script(model)
        model_scripted.save('{}/action_{}.pt'.format(model_dir, epoch))



if __name__ == "__main__":
    main()
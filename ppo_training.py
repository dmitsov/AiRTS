import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.optim as optim
from gym.spaces import MultiDiscrete
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecVideoRecorder
from torch.distributions.categorical import Categorical
import wandb
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import random
from datetime import datetime, timedelta
from collections import OrderedDict


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], mask_value=None):
        logits = torch.where(masks.bool(), logits, mask_value)
        super(CategoricalMasked, self).__init__(probs, logits, validate_args)


class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init_xavier(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.xavier_uniform_(layer.weight, nn.init.calculate_gain('relu'))
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# class Agent(nn.Model):
#     def __init__(self, env, mapsize=16 * 16):
#         super(Agent, self).__init__()

#         self.mapsize = 16*16
#         self.env = env

#     @abstractmethod
#     def get_action_and_value(self, x, action=None, invalid_action_masks=None):
#         pass

#     @abstractmethod
#     def get_value(self, x):
#         pass

class ConvAgent(nn.Module):
    def __init__(self, envs, mapsize=16 * 16):
        super(ConvAgent, self).__init__()
        self.mapsize = mapsize
        self.envs =envs
        self.action_space_num = self.envs.action_plane_space.nvec.sum()

        h, w, c = envs.observation_space.shape
        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, 78, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )
        self.register_buffer("mask_value", torch.tensor(-1e8))

    def get_action_and_value(self, x, action=None, invalid_action_masks=None, device=None, hidden_states=None, c_values=None):
        size = x.size()

        if x is not None:

            if len(size) > 4:
                batch_size, _, H, W, C = size
                x = x.squeeze(dim = 1)
            else:
                batch_size, H, W, C = size
                
        else:
            batch_size = 1

        hidden = self.encoder(x)
        logits = self.actor(hidden)
        grid_logits = logits.view(batch_size, -1, self.action_space_num) 
        #print("grid_logits ", grid_logits.shape)

        split_logits = torch.split(grid_logits, self.envs.action_plane_space.nvec.tolist(), dim=2)

        if action is None:
            invalid_action_masks = invalid_action_masks.view(batch_size, -1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks, self.envs.action_plane_space.nvec.tolist(), dim=2)
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            invalid_action_masks = invalid_action_masks.view(batch_size, -1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks, self.envs.action_plane_space.nvec.tolist(), dim=2)
            action = action.T
            
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
        #print("action ", action.shape)
        
      

        logprob = torch.stack([categorical.log_prob(a if batch_size == 1 else a.squeeze(dim=1).permute((1, 0))) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])

        num_predicted_parameters = len(self.envs.action_plane_space.nvec)
        #print("old logprob ", logprob.shape)
        #print("old logprob T", logprob.T.shape)
        logprob = logprob.T.reshape(batch_size, -1, self.mapsize, num_predicted_parameters)
        action = action.T.reshape(batch_size, -1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.reshape(batch_size, -1, self.mapsize, num_predicted_parameters)

        
        value = self.critic(hidden)

        #print("value ", value.shape)    

   

        if batch_size == 1:
            action = action[0]
            logprob = logprob[0]
            value = value[0]
            logprob = logprob.sum(1).sum(1)
            entropy = entropy.sum(1).sum(1)
        else:
            logprob = logprob.sum(1).sum(1).sum(1).view(batch_size, 1)
            entropy = entropy.sum(1).sum(1).sum(1).view(batch_size, 1)
            value = value.view(batch_size, 1)

        return action, logprob, entropy, value, torch.zeros((1, 512)), torch.zeros((1, 512))

    def get_value(self, x,  hidden_states=None, c_values=None):
        return self.critic(self.encoder(x)), (torch.zeros((1, 512)), torch.zeros((1, 512)))

    def add_weights_to_histogram(self, writer, global_step):
        writer.add_histogram('conv2d_0', self.encoder[1].weight, global_step=global_step)
        
        writer.add_histogram('conv2d_1', self.encoder[4].weight, global_step=global_step)
        
        writer.add_histogram('actor_0', self.actor[0].weight, global_step=global_step)
        writer.add_histogram('actor_1', self.actor[2].weight, global_step=global_step)
        writer.add_histogram('critic_0', self.critic[1].weight, global_step=global_step)
        writer.add_histogram('critic_1', self.critic[3].weight, global_step=global_step)


class LSTMEncoder(nn.Module):
    def __init__(self, env, device, layer_count=1, mapsize=16 * 16):
        super(LSTMEncoder, self).__init__()
        self.mapsize = 16*16
        self.envs = env
        _, _, c = env.observation_space.shape
        self.hidden_size = 512
        self.layer_count = layer_count
        self.output_size = 512
        self.input_size = 1024
        self.hidden_value = None
        self.c_value = None
        self.conv_transpose_in_channels = 32
        self.device = device

        self.CNNs = nn.Sequential(
            Transpose((0, 3, 1, 2)),
            layer_init_xavier(nn.Conv2d(c, out_channels=32, kernel_size=4, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.LayerNorm((32, 8, 8)),
            layer_init_xavier(nn.Conv2d(32, out_channels=64, kernel_size=4, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            nn.LayerNorm((64, 4, 4)),
            nn.Flatten(),
            layer_init_xavier(nn.Linear(in_features=self.input_size, out_features=self.hidden_size)),
            nn.ReLU(),
            nn.LayerNorm((self.hidden_size)),
        )

        self.action_space_num = self.envs.action_plane_space.nvec.sum()

        self.lstmCore = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.layer_count, batch_first=True)
        

        self.action_head = nn.Sequential(
            layer_init_xavier(nn.ConvTranspose2d(in_channels=self.conv_transpose_in_channels, out_channels = 64, kernel_size=4,stride=2, padding=1)),
            nn.ReLU(),
            layer_init_xavier(nn.ConvTranspose2d(in_channels=64, out_channels = self.action_space_num , kernel_size=4, stride=2, padding=1)),
            nn.ReLU(),
            Transpose((0, 2, 3, 1)),
        )

        self.critic_head = nn.Sequential(
            layer_init_xavier(nn.Linear(in_features=self.hidden_size, out_features=64)),
            nn.ReLU(),
            nn.LayerNorm(64),
            layer_init_xavier(nn.Linear(in_features=64, out_features=1))
        )


        self.register_buffer("mask_value", torch.tensor(-1e8))

    def forward(self, x: torch.Tensor):
        states = x["states"]
        invalid_action_masks = x["invalid_action_masks"]
        actions = x["actions"]

        return self.get_action_and_value(states, action=actions, invalid_action_masks=invalid_action_masks)

    def add_weights_to_histogram(self, writer, global_step):
        writer.add_histogram('conv2d_0', self.CNNs[1].weight, global_step=global_step)
        
        writer.add_histogram('conv2d_1', self.CNNs[5].weight, global_step=global_step)
        # writer.add_histogram('lstm_ih_0', self.lstmCore.weight_ih_l[0], global_step=global_step)
        # writer.add_histogram('lstm_hh_0', self.lstmCore.weight_hh_l[0], global_step=global_step)
        # writer.add_histogram('lstm_bias_ih_0', self.lstmCore.bias_ih_l[0], global_step=global_step)
        # writer.add_histogram('lstm_bias_hh_0', self.lstmCore.bias_hh_l[0], global_step=global_step)
        # writer.add_histogram('lstm_hr_0', self.lstmCore.weight_hr_l[0], global_step=global_step)
        
        writer.add_histogram('actor_0', self.action_head[0].weight, global_step=global_step)
        writer.add_histogram('actor_1', self.action_head[2].weight, global_step=global_step)
        writer.add_histogram('critic_0', self.critic_head[0].weight, global_step=global_step)
        writer.add_histogram('critic_1', self.critic_head[3].weight, global_step=global_step)

    def set_use_preserved_hidden_vector(self,value):
        self.use_preserved_hidden_vector = value

    #  def select_action(self, state):
    #     probs = self.policy_old(state)
    #     m = Categorical(probs)
    #     action = m.sample()
    #     return action.item(), m.log_prob(action)

    def get_action_and_value(self, x, action=None, invalid_action_masks=None, hidden_states=None, c_values=None):
        size = x.size()

        if x is not None:

            if len(size) > 4:
                batch_size, _, H, W, C = size
                x = x.squeeze(dim = 1)
            else:
                batch_size, H, W, C = size
                
        else:
            batch_size = 1

        # if action is not None:
        #     #print("x.size ", x.size)

        #print("x ", x.shape)
    

        encoded_value = self.CNNs(x)
        encoded_value = encoded_value.view(batch_size, 1, -1)

        #print("encoded value ", encoded_value.shape)

        out = None #could be used later
       

        if hidden_states is None:
            out, (h_out, c_out) = self.lstmCore(encoded_value)
        else:
            hidden_states = hidden_states.view(self.layer_count, batch_size, -1).to(self.device)
            c_values = c_values.view(self.layer_count, batch_size, -1).to(self.device)
            out, (h_out, c_out) = self.lstmCore(encoded_value, (hidden_states, c_values))
            out = out[-1]
        #print("h_out ", h_out.shape)
        
        out = out + encoded_value

        encoded_hidden = out.view(batch_size,self.conv_transpose_in_channels, 4, 4)

        
        logits = self.action_head(encoded_hidden)
        grid_logits = logits.view(batch_size, -1, self.action_space_num) 

        split_logits = torch.split(grid_logits, self.envs.action_plane_space.nvec.tolist(), dim=2)

        if action is None:
            invalid_action_masks = invalid_action_masks.view(batch_size, -1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks, self.envs.action_plane_space.nvec.tolist(), dim=2)
            #print("split_invalid_action_mask length", len(split_invalid_action_masks))

            #print("split_invalid_action_mask ", split_invalid_action_masks[0].shape)
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            #print("old action shape ", action.shape)
        
            invalid_action_masks = invalid_action_masks.view(batch_size, -1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks, self.envs.action_plane_space.nvec.tolist(), dim=2)
            #print("split_invalid_action_mask length", len(split_invalid_action_masks))
            
            multi_categoricals = [
                CategoricalMasked(logits=logits, masks=iam, mask_value=self.mask_value)
                for (logits, iam) in zip(split_logits, split_invalid_action_masks)
            ]

        if batch_size > 1:
            print("logits ", logits.shape)
            print("split_logits ", split_logits[0].shape)
            print("invalid_action_masks ", invalid_action_masks.shape)

            print("split_invalid_action_mask ", split_invalid_action_masks[0].shape)
            
            print("multi_categoricals length ", len(multi_categoricals))
            print("action ", action.shape)
            test_action = torch.split(action, 1)
            print("test_action len ", len(test_action))
            print("test_action shape ", test_action[0].shape)

        #print("action ", action.shape)
        ##print("categorical ", multi_categoricals[0].shape)
        logprob = None
        if batch_size == 1:
            logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        else:
            logprob = torch.stack([categorical.log_prob(a.T.squeeze(dim=1)) for a, categorical in zip(action.T, multi_categoricals)])
            print("logprob shape ", logprob.shape)
        
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])

        num_predicted_parameters = len(self.envs.action_plane_space.nvec)
        #print("old logprob ", logprob.shape)
        #print("old logprob T", logprob.T.shape)
        logprob = logprob.T.reshape(batch_size, -1, self.mapsize, num_predicted_parameters)
        action = action.T.reshape(batch_size, -1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.reshape(batch_size, -1, self.mapsize, num_predicted_parameters)


        #print("action ", action.shape)
        #print("logprob ", logprob.shape)    

        value = self.critic_head(out)

        #print("value ", value.shape)    

        if batch_size == 1:
            action = action[0]
            logprob = logprob[0]
            value = value[0]
            logprob = logprob.sum(1).sum(1)
            entropy = entropy.sum(1).sum(1)
        else:
            logprob = logprob.sum(1).sum(1).sum(1).view(batch_size, 1)
            entropy = entropy.sum(1).sum(1).sum(1).view(batch_size, 1)
            value = value.view(batch_size, 1)

        return action, logprob, entropy, value, h_out, c_out

    def get_value(self, x, hidden_state, c_value):
        batch_size, C, H, W = x.size()

        encoded_value = self.CNNs(x)
        encoded_value = encoded_value.view(1, batch_size, -1).to(self.device)
        hidden_state = hidden_state.view(self.layer_count, batch_size, -1)
        c_value = c_value.view(self.layer_count, batch_size, -1)

        out, (h_out, c_out) = self.lstmCore(encoded_value, (hidden_state, c_value))
        out = out + encoded_value
        return self.critic_head(out), (h_out, c_out)



class RtsDataset(Dataset):
    
    def __init__(self, states, actiona, log_probs, rewards, next_states, dones, values, invalid_action_masks, hidden_states, c_values, transform=None):
        self.actions = actiona
        self.states = states
        self.rewards = rewards
        self.log_probs = log_probs
        self.next_states = next_states
        self.dones = dones
        self.values = values
        #self.advantages = advanages
        self.invalid_action_masks = invalid_action_masks
        self.hidden_states = hidden_states
        self.c_values = c_values
        #self.returns = returns
        self.transform = transform

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        batch_actions = self.actions[idx]
        batch_states = self.states[idx]
        batch_rewards = self.rewards[idx]
        batch_log_probs = self.log_probs[idx]
        batch_next_states = self.next_states[idx]
        batch_dones = self.dones[idx]
        batch_values = self.values[idx]
        #batch_advantages = self.advantages[idx]
        batch_invalid_action_masks = self.invalid_action_masks[idx]
        batch_hidden_states = self.hidden_states[idx]
        batch_c_values = self.c_values[idx]
        #batch_returns = self.returns[idx]

        sample = (batch_actions, batch_states, batch_rewards, batch_log_probs, batch_next_states, batch_dones, batch_values, batch_invalid_action_masks, batch_hidden_states, batch_c_values, idx)

        if self.transform:
            sample = self.transform(sample)

        return sample

class PPO:
    def __init__(self, device, experiment_name, gae_lambda=0.95, lr=1e-3, gamma=0.99, clip_epsilon=0.1, num_epochs=4, num_steps=256, batch_size=32):
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.learning_rate = lr
        self.clip_epsilon = clip_epsilon
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device
        self.experiment_name = experiment_name
        self.hidden_state_size = 512
        
        # saving actions
        self.actions = None
        self.states = None
        self.rewards = None
        self.log_probs = None
        self.next_states = None
        self.dones = None
        self.values = None
        self.advantages = None
        self.invalid_action_masks = None
        self.hidden_states = None
        self.c_values = None
        self.returns = None
        self.e_hidden_value = None
        self.e_c_value = None

    def store_transition(self, step_number, state, action, log_prob, reward, next_state, done, value, invalid_action_mask, hidden_value, c_value):
        self.states[step_number] = state
        self.actions[step_number] = action
        self.log_probs[step_number] = log_prob
        self.rewards[step_number] = reward
        self.next_states[step_number] = next_state
        self.dones[step_number] = done
        self.values[step_number] = value
        self.hidden_states[step_number] = hidden_value
        self.c_values[step_number] = c_value

    def initialize_transitions(self,env, layer_count, mapsize):
        action_space_shape = (mapsize, len(envs.action_plane_space.nvec))
        invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum())

        self.states = torch.zeros((self.num_steps, 1) + env.observation_space.shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, 1) + action_space_shape).to(self.device)
        self.rewards = torch.zeros((self.num_steps, 1)).to(self.device)
        self.log_probs = torch.zeros((self.num_steps, 1)).to(self.device)
        self.next_states = torch.zeros((self.num_steps, 1) + env.observation_space.shape).to(self.device)
        self.dones = torch.zeros((self.num_steps, 1)).to(self.device)
        self.values = torch.zeros((self.num_steps, 1)).to(self.device)
        self.advantages = torch.zeros((self.num_steps, 1)).to(self.device)
        self.returns = torch.zeros((self.num_steps)).to(self.device)
        self.invalid_action_masks = torch.zeros((self.num_steps, 1) + invalid_action_shape).to(self.device)
        self.hidden_states = torch.zeros((self.num_steps, layer_count, self.hidden_state_size)).to(self.device)
        self.c_values = torch.zeros((self.num_steps, layer_count, self.hidden_state_size)).to(self.device)

    def prepare_loader(self, env, layer_count):
        # blank_frame = torch.zeros(env.observation_space.shape).to(device)
        # initial_hidden_values = []
        # initial_c_values = []
        # for i in range(layer_count-1):
        #     blank_part_hidden = [torch.zeros_like(blank_frame) for j in range(layer_count - 1 - i)]
        #     blank_part_c = [torch.zeros_like(blank_frame) for j in range(layer_count - 1 - i)]
        #     frame_part_hidden = self.hidden_states[0:i]
        #     frame_part_c = self.c_values[0:i]
        #     h_entry = torch.stack([*blank_part_hidden, *frame_part_hidden])
        #     c_entry = torch.stack([*blank_part_c, *frame_part_c])

        #     initial_hidden_values.append(h_entry)
        #     initial_c_values.append(c_entry)


        # hidden_states = [self.hidden_states[i:i+layer_count] for i in range(len(self.hidden_states) - (layer_count - 1))]
        # c_values = [self.c_values[i:i+layer_count] for i in range(len(self.c_values) - (layer_count - 1))]

        # hidden_states = [*initial_hidden_values, *hidden_states]
        # c_values = [*initial_c_values, *c_values]

        rtsDataset = RtsDataset(self.states, self.actions, self.log_probs, self.rewards, self.next_states, self.dones, self.values, self.invalid_action_masks, self.hidden_states, self.c_values)
        dataloader = DataLoader(rtsDataset, batch_size=self.batch_size,
                        shuffle=True, num_workers=0)
        
        return dataloader
    
    def update_agent(self, env, agent, optimizer, writer):

        #self.calculate_advantages(agent)
        steps_loader = self.prepare_loader(env, agent)
        value_loss = None
        action_loss = None
        print("updating agent")
        agent.train()
        update_step = 0
        for i in range(self.num_epochs):
            self.calculate_advantages_epoch(agent)

            for _, sample_batches in enumerate(steps_loader):

                actions, states, log_probs, rewards, next_states, _, values, invalid_action_masks, hidden_states, c_values, idx = sample_batches

                states.to(self.device)
                actions.to(self.device)
                log_probs.to(self.device)
                rewards.to(self.device)
                next_states.to(self.device)
                values.to(self.device)
                hidden_states.to(self.device)
                c_values.to(self.device)
                
                returns = self.returns[idx]
                advantages = self.advantages[idx]
                
                update_step += len(states)

                # #print("States ", states.shape)
                # #print("Actions ", actions.shape)
                # #print("Log_probs ", log_probs.shape)
                # #print("Rewards ", rewards.shape)
                # #print("Next states ", next_states.shape)
                # #print("Values ", values.shape)                
                # #print("Advantages ", advantages.shape)
                

                #normalizing advanages
                writer.add_histogram('advanatages', advantages, global_step=self.global_step)

                # advantages = (advantages - advantages.mean()) /(advantages.std() + 1e-8)
                _, batch_logprob, batch_entropy, batch_values, _, _ = agent.get_action_and_value(states, action=actions, invalid_action_masks=invalid_action_masks, hidden_states=hidden_states, c_values=c_values)

                # ratio = (batch_logprob - log_probs).exp()
                # action_loss = (-advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)).mean()
                # value_loss = 0.5 * ((batch_values - values) ** 2)


                if batch_logprob.shape != log_probs.shape:
                    raise ValueError(f"Shape mismatch: batch_logprob {batch_logprob.shape} vs log_probs {log_probs.shape}")

                print("Batch logprob shape ", batch_logprob.shape)
                print("Batch logprob shape ", batch_logprob.shape)

                # Calculate the ratio
                ratio = (batch_logprob - log_probs).exp()
                #ratio.retain_grad() 
                # Ensure advantages shape matches ratio shape
                if advantages.shape != ratio.shape:
                    raise ValueError(f"Shape mismatch: advantages {advantages.shape} vs ratio {ratio.shape}")

                # Calculate action loss
                action_loss1 = -advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                #action_loss1.retain_grad()
                action_loss2 = -advantages * ratio
                #action_loss2.retain_grad()
                action_loss = torch.max(action_loss1, action_loss2).mean()
                #action_loss.retain_grad()
                # Ensure batch_values shape matches values shape
                if batch_values.shape != values.shape:
                    raise ValueError(f"Shape mismatch: batch_values {batch_values.shape} vs values {values.shape}")

                # Calculate value loss
                value_loss_unclipped = (batch_values - returns) ** 2
                v_clipped = values + torch.clamp(batch_values - values, -self.clip_epsilon, self.clip_epsilon)
                value_loss_clipped = (v_clipped - returns) ** 2
                value_loss_max = torch.max(value_loss_clipped, value_loss_unclipped)
                value_loss_max.retain_grad()
                entropy_loss = batch_entropy.mean()

                value_loss = 0.5 * value_loss_max.mean()
                value_loss.retain_grad()

                loss = action_loss - 0.01 * entropy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()

                # writer.add_histogram('log loss grad', ratio.grad, global_step=self.global_step)
                # writer.add_histogram('action_loss1 grad', action_loss1.grad, global_step=self.global_step)
                # writer.add_histogram('action_loss2 grad', action_loss2.grad, global_step=self.global_step)
                # writer.add_histogram('value_loss grad', value_loss.grad, global_step=self.global_step)

                # writer.add_histogram('actor_0', self.action_head[0].grad, global_step=global_step)
                # writer.add_histogram('actor_1', self.action_head[2].grad, global_step=global_step)
                
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

                optimizer.step()

                # if update_step % 2048 == 0:
                #     writer.add_graph(agent, {"states": states, "invalid_action_masks": invalid_action_masks, "actions":actions})

        return action_loss,  value_loss
    
   

    # def calculate_advantages(self, agent):
    #     self.advantages = torch.zeros_like(self.rewards)
    #     with torch.no_grad():
    #         self.new_values = torch.zeros_like(self.rewards).to(self.device)
    #         final_h = self.hidden_states[-1].view(1, 1, -1).to(self.device)
    #         final_c = self.c_values[-1].view(1, 1, -1).to(self.device)

    #         _, (next_h, next_c) = agent.get_value(self.states[-1], final_h, final_c)
    #         last_value, _ = agent.get_value(self.next_states[-1], next_h, next_c)
    #         last_done = self.dones[-1]
    #         lastgaelam = 0
    #         for s in reversed(range(self.num_steps)):
    #             next_nonterminal = 1.0 - self.dones[s + 1] if s < self.num_steps - 1 else 1.0 - last_done
                
    #             next_value = self.new_values[s + 1] if s < self.num_steps - 1 else last_value

    #             delta = self.rewards[s] + self.gamma * next_nonterminal * next_value - self.values[s]
    #             self.advantages[s] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            
    #         self.returns = self.advantages + self.values

    def calculate_advantages_epoch(self, agent):
        self.advantages = torch.zeros_like(self.rewards)
        with torch.no_grad():
            self.new_values = torch.zeros_like(self.rewards).to(self.device)
            self.values = torch.zeros_like(self.rewards).to(self.device)

            for i in range(len(self.states)):
                s = self.states[i].to(self.device)
                h = self.hidden_states[i].view(1, 1, -1).to(self.device)
                c = self.c_values[i].view(1, 1, -1).to(self.device)

                v, _ = agent.get_value(s, hidden_state=h, c_value=c)
                self.values[i] = v
            
            
            lastgaelam = 0
            for s in reversed(range(self.num_steps - 1)):
                next_nonterminal = 1.0 - self.dones[s + 1]
                
                next_value = self.values[s + 1]
                delta = self.rewards[s] + self.gamma * next_nonterminal * next_value - self.values[s]
                self.advantages[s] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            
            self.returns = self.advantages + self.values


    def gather_episode_data(self, agent, env, mapsize, writer, run_name, hyperparameters):
        agent.eval()
        #env.reset()    
        next_state = torch.Tensor(env.reset()).to(self.device)

        if self.e_hidden_value is None:# or self.global_step % 2000 == 0:
            self.e_hidden_value = torch.zeros((agent.layer_count,512)).to(self.device)
        
        if self.e_c_value is None:# or self.global_step % 2000 == 0:
            self.e_c_value = torch.zeros((agent.layer_count,512)).to(self.device)

        self.initialize_transitions(env, agent.layer_count, mapsize)

        for s in range(0, self.num_steps):
            self.global_step += 1
            invalid_action_mask = torch.tensor(env.get_action_mask()).to(self.device)
            with torch.no_grad():
                state = next_state
                action, logprob, _, value, h_out, c_out = agent.get_action_and_value(state, action=None, invalid_action_masks=invalid_action_mask, hidden_states=self.e_hidden_value, c_values=self.e_c_value )
                old_hidden_value = self.e_hidden_value
                old_c_value = self.e_c_value 

                next_state, reward, done, infos = env.step(action.cpu().numpy().reshape(envs.num_envs, -1))
                if done:
                    self.e_hidden_value = torch.zeros((agent.layer_count,512)).to(self.device)
                else:
                    self.e_hidden_value = h_out.view(agent.layer_count, 512)
                
                self.e_c_value = c_out.view(agent.layer_count, 512)
                
                next_state, reward, done = torch.Tensor(next_state).to(self.device), torch.Tensor(reward).to(self.device), torch.Tensor(done).to(self.device)
                for info in infos:
                    if "episode" in info.keys():
                        print(f"global_step={self.global_step}, episodic_return={info['episode']['r']}")
                        
                        metrics = {
                            "charts/episodic_return": info["episode"]["r"],
                            "charts/episodic_length": info["episode"]["l"]
                        }

                        for key in info["microrts_stats"]:
                           metrics[f"charts/episodic_return/{key}"] = info["microrts_stats"][key]
                        
                        writer.add_hparams(hyperparameters, metrics, run_name=run_name, global_step=self.global_step)
                       
                        break

                #print("Gathered logprob ", logprob.shape)
                #state, action, log_prob, reward, next_state, done, value
                self.store_transition(s, state, action, logprob, reward, next_state, done, value, invalid_action_mask, old_hidden_value, old_c_value)

    def save_agent(self, agent, update_num):
        models_path = f"models/{self.experiment_name}"
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        agent_path = f"models/{self.experiment_name}/agent_{update_num}.pt"
        torch.save(agent.state_dict(), agent_path)
        #wandb.save(agent_path, base_path=f"models/{self.experiment_name}", policy="now")


    def train_agent(self, env, agent, num_of_iterations, mapsize):
        
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate) #we could try AdamW and 
        writer = SummaryWriter(f"runs/{self.experiment_name}")
        lr = lambda f: f * self.learning_rate

        self.global_step = 0
        hyperparameters = dict({
            'num_epochs': self.num_epochs,
            'num_steps': self.num_steps,
            'seed': 1234,
            'clip_epsilon': self.clip_epsilon,
            'gamma': self.gamma,
            'num_of_iterations': num_of_iterations
        })

        run_name = "PPO LSTM encoder"      

        print("Starting training")
        startTime = datetime.now()
        for i in range(num_of_iterations):
            print(f"Experiment name: {self.experiment_name} Gathering episode data for update number: ", i)
            
            self.gather_episode_data(agent, env, mapsize, writer, run_name, hyperparameters)
            frac = 1.0 - i / num_of_iterations
            lrnow = lr(frac)
            optimizer.param_groups[0]["lr"] = lrnow
            
            action_loss, value_loss = self.update_agent(env, agent, optimizer, writer)

            agent.add_weights_to_histogram(writer, self.global_step)
            if i % 10 == 0:
                print(f"Experiment name:{self.experiment_name} Saving model {i}")
                self.save_agent(agent, i)
              
            metrics = {
                "charts/learning_rate": optimizer.param_groups[0]["lr"],
                "losses/value_loss": value_loss.detach().item(),
                "losses/policy_loss": action_loss.detach().item()
            }

            writer.add_hparams(hyperparameters, metrics, run_name=run_name, global_step=self.global_step)
        

        endTime = datetime.now()

        timeSpan = endTime - startTime
        
        writer.close()
        print("Training finished")
        print("Full-time: ", timeSpan)


class MicroRTSStatsRecorder(VecEnvWrapper):
    def __init__(self, env, gamma=0.99) -> None:
        super().__init__(env)
        self.gamma = gamma

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]
        self.ts = np.zeros(self.num_envs, dtype=np.float32)
        self.raw_discount_rewards = [[] for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos[:])
        for i in range(len(dones)):
            self.raw_rewards[i] += [infos[i]["raw_rewards"]]
            self.raw_discount_rewards[i] += [
                (self.gamma ** self.ts[i])
                * np.concatenate((infos[i]["raw_rewards"], infos[i]["raw_rewards"].sum()), axis=None)
            ]
            self.ts[i] += 1
            if dones[i]:
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)
                raw_names = [str(rf) for rf in self.rfs]
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(0)
                raw_discount_names = ["discounted_" + str(rf) for rf in self.rfs] + ["discounted"]
                info["microrts_stats"] = dict(zip(raw_names, raw_returns))
                info["microrts_stats"].update(dict(zip(raw_discount_names, raw_discount_returns)))
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.ts[i] = 0
                newinfos[i] = info
        return obs, rews, dones, newinfos


if __name__ == "__main__":
    seed = 1234

    # run = wandb.init(
    #     project="test-run",
    #     #entity="ppo_entity",
    #     #sync_tensorboard=True,
    #     #ame="test_ppo",
    #     monitor_gym=True,
    #     save_code=True,
    # )

  #  wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"runs/ppo_test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    envs = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        partial_obs=False,
        max_steps=2000,
        render_theme=2,
        # ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs - 6)]
        # + [microrts_ai.randomBiasedAI for _ in range(min(args.num_bot_envs, 2))]
        # + [microrts_ai.lightRushAI for _ in range(min(args.num_bot_envs, 2))]
        # + [microrts_ai.workerRushAI for _ in range(min(args.num_bot_envs, 2))],
        ai2s = [microrts_ai.workerRushAI],
        map_paths = ["maps/16x16/basesWorkers16x16A.xml"],
        cycle_maps= ["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )

    envs = MicroRTSStatsRecorder(envs)
    envs = VecMonitor(envs)
    assert isinstance(envs.action_space, MultiDiscrete), "only MultiDiscrete action space is supported"
    envs.reset()    

    agent = LSTMEncoder(envs, device, layer_count=16).to(device)
    convAgent = ConvAgent(envs).to(device)
    runs = [dict(
        {
            'name': 'ppo_lstm_7_multi_layer',
            'batch_size': 512,
            'num_steps': 2048,
            'num_updates': 1000,
            'agent': agent,
        }),
        # dict({
        #     'name': 'ppo_conv_0',
        #     'batch_size': 512,
        #     'num_steps': 2048,
        #     'num_updates': 1000,
        #     'agent': convAgent
        # }),
        
    ]

    for _, run in enumerate(runs):
        run_name = run['name']
        batch_size = run['batch_size']
        num_steps = run['num_steps']
        ppo = PPO(device=device,experiment_name=run_name, batch_size=batch_size, num_steps=num_steps)
        ppo.train_agent(envs, run['agent'], run['num_updates'], 16*16)
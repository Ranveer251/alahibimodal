import itertools
import copy
import math

import numpy as np
import torch

import trajnetplusplustools

from .modules import Hidden2Normal, InputEmbedding

from .. import augmentation
from .utils import center_scene
from scipy.optimize import minimize
import torch.nn as nn

NAN = float('nan')

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


def generate_pooling_inputs(obs2, obs1, hidden_cell_state, track_mask, batch_split):
    hidden_states_to_pool = torch.stack(hidden_cell_state[0]).clone() # detach?
    hidden_dim = hidden_states_to_pool.size(-1)
    # tensor for pooling; filled with nan-mask [bs, max # neighbor, 2]
    max_num_neighbor = (batch_split[1:] - batch_split[:-1]).max()   # number of agents in a scene minus the primary
    batch_size = len(batch_split) - 1
    curr_positions = torch.empty(batch_size, max_num_neighbor, 2).fill_(float('nan')).to(obs1.device) # placeholder
    prev_positions = torch.empty(batch_size, max_num_neighbor, 2).fill_(float('nan')).to(obs1.device) # placeholder
    curr_hidden_state = torch.empty(batch_size, max_num_neighbor, hidden_dim).fill_(float('nan')).to(obs1.device) # placeholder
    track_mask_positions = torch.empty(batch_size, max_num_neighbor).fill_(False).bool().to(obs1.device)  # placeholder

    for i in range(batch_size):
        curr_positions[i, :batch_split[i+1]-batch_split[i]] = obs2[batch_split[i]:batch_split[i+1]]
        prev_positions[i, :batch_split[i+1]-batch_split[i]] = obs1[batch_split[i]:batch_split[i+1]]
        curr_hidden_state[i, :batch_split[i+1]-batch_split[i]] = hidden_states_to_pool[batch_split[i]:batch_split[i+1]]
        track_mask_positions[i, :batch_split[i+1]-batch_split[i]] = track_mask[batch_split[i]:batch_split[i+1]].bool()

    return curr_positions, prev_positions, curr_hidden_state, track_mask_positions

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# SELayer Source: https://github.com/HuangxingLin123/A2Net-Adjacent-Aggregation-Networks-for-Image-Raindrop-Removal

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False, intent_pool = False):
        """ Initialize the LSTM forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        goal_dim : Embedding dimension of the unit vector pointing towards the goal
        goal_flag: Bool
            if True, the embedded goal vector is concatenated to the input embedding of LSTM 
        """

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input

        ## Location
        scale = 4.0
        self.input_embedding = InputEmbedding(2, self.embedding_dim, scale)

        ## Goal
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0

        ## Pooling
        self.intent_pool = intent_pool
        pooling_dim = 0
        if pool is not None and self.pool_to_input:
            pooling_dim = self.pool.out_dim 
        
        ## LSTMs
        self.encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        
        ## Channel-Wise Attention (CNN)
        self.ca1 = SEBasicBlock(128,128)
        self.ca2 = SEBasicBlock(128,128)
        self.ca3 = SEBasicBlock(128,128)

        self.mlp = nn.Linear(8,1)
        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals, batch_split, obs_first = None):
        """Do one step of prediction: two inputs to one normal prediction.
        
        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_cell_state : tuple (hidden_state, cell_state)
            Current hidden_cell_state of the pedestrians
        obs1 : Tensor [num_tracks, 2]
            Previous x-y positions of the pedestrians
        obs2 : Tensor [num_tracks, 2]
            Current x-y positions of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        
        Returns
        -------
        hidden_cell_state : tuple (hidden_state, cell_state)
            Updated hidden_cell_state of the pedestrians
        normals : Tensor [num_tracks, 5]
            Parameters of a multivariate normal of the predicted position 
            with respect to the current position
        """
        num_tracks = len(obs2)
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0

        ## Masked Hidden Cell State
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        ## Mask current velocity & embed
        curr_velocity = obs2 - obs1
        curr_velocity = curr_velocity[track_mask]
        input_emb = self.input_embedding(curr_velocity)

        ## Mask Goal direction & embed
        if self.goal_flag:
            ## Get relative direction to goals (wrt current position)
            norm_factors = (torch.norm(obs2 - goals, dim=1))
            goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1)
            goal_direction[norm_factors == 0] = torch.tensor([0., 0.], device=obs1.device)
            goal_direction = goal_direction[track_mask]
            goal_emb = self.goal_embedding(goal_direction)
            input_emb = torch.cat([input_emb, goal_emb], dim=1)

        ## Mask & Pool per scene
        if self.pool is not None:
            if obs_first is not None:
              obs1_temp = obs_first
            else:
              obs1_temp = obs2
            curr_positions, prev_positions, curr_hidden_state, track_mask_positions = \
                generate_pooling_inputs(obs2, obs1_temp, hidden_cell_state, track_mask, batch_split)
            pool_sample = self.pool(curr_hidden_state, prev_positions, curr_positions)
            pooled = pool_sample[track_mask_positions.view(-1)]

            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_cell_stacked[0] += pooled

        # LSTM step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal(hidden_cell_stacked[0])

        # unmask [Update hidden-states and next velocities of pedestrians]
        normal = torch.full((track_mask.size(0), 10), NAN, device=obs1.device)
        mask_index = [i for i, m in enumerate(track_mask) if m]
        for i, h, c, n in zip(mask_index,
                              hidden_cell_stacked[0],
                              hidden_cell_stacked[1],
                              normal_masked):
            hidden_cell_state[0][i] = h
            hidden_cell_state[1][i] = c
            normal[i] = n

        return hidden_cell_state, normal

    def forward(self, observed, goals, batch_split, prediction_truth=None, n_predict=None):
        """Forecast the entire sequence 
        
        Parameters
        ----------
        observed : Tensor [obs_length, num_tracks, 2]
            Observed sequences of x-y coordinates of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene        
        prediction_truth : Tensor [pred_length - 1, num_tracks, 2]
            Prediction sequences of x-y coordinates of the pedestrians
            Helps in teacher forcing wrt neighbours positions during training
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        rel_pred_scene : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        pred_scene : Tensor [pred_length, num_tracks, 2]
            Predicted positions of pedestrians i.e. absolute positions
        """

        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict - 1)]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        num_tracks = observed.size(1)
        hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
        )

        ## Reset LSTMs of Interaction Encoders.
        if self.pool is not None:
            max_num_neighbor = (batch_split[1:] - batch_split[:-1]).max() - 1
            batch_size = len(batch_split) - 1
            self.pool.reset(batch_size * (max_num_neighbor+1), max_num_neighbor, device=observed.device)

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        if self.intent_pool:
          obs_first = observed[0]
        else:
          obs_first = None

          h_stack = []
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_cell_state, normal = self.step(self.encoder, hidden_cell_state, obs1, obs2, goals, batch_split, obs_first)

            # concat predictions
            h_temp = torch.stack(hidden_cell_state[0], dim=0)
            # print(h_temp.size())
            h_stack.append(h_temp)
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean
        h_stack_t = torch.stack(h_stack, dim=2)
        # print(h_stack_t.size())
        h_stack_t = self.ca1(h_stack_t)
        h_stack_t = self.ca2(h_stack_t)
        h_stack_t = self.ca3(h_stack_t)
        # print(h_stack_t.size())
        h_new = self.mlp(h_stack_t)
        # print(h_new.size())

        hcs_list = list(torch.unbind(torch.squeeze(h_new), dim = 0))
        hidden_cell_state = (hcs_list, hidden_cell_state[1])

        # initialize predictions with last position to form velocity. DEEP COPY !!!
        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth)
        )))

        normals_dist = []
        # decoder, predictions
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            else:
                for primary_id in batch_split[:-1]:
                    obs1[primary_id] = positions[-2][primary_id].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            else:
                for primary_id in batch_split[:-1]:
                    obs2[primary_id] = positions[-1][primary_id].detach()  # DETACH!!!
            hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2, goals, batch_split, obs_first)

            # concat predictions
            normals.append(normal)
            normals_dist.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean

        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)

        # print("NORMALS actual shape: ", len(normals))
        # print("OBS: ", observed.size())
        return rel_pred_scene, pred_scene, normals_dist

class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)
    def joint_pdf(self, xy,mu1, mu2, s1, s2, rho):
        x1, x2 = xy[:,0], xy[:,1]
        # mu1, mu2, s1, s2, rho = self.mu1x, self.mu1y, self.s1x, self.s1y, self.rho1
        norm1 = x1 - mu1
        norm2 = x2 - mu2

        sigma1sigma2 = s1 * s2

        z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / sigma1sigma2

        numerator = np.exp(-z / (2 * (1 - rho ** 2)))
        denominator = 2 * math.pi * sigma1sigma2 * np.sqrt(1 - rho ** 2)

        return numerator/(0.01 + denominator)

    def joint_pdf1(self, xy):
        x1, x2 = xy[:,0], xy[:,1]
        mu1, mu2, s1, s2, rho = self.mu1x, self.mu1y, self.s1x, self.s1y, self.rho1
        norm1 = x1 - mu1
        norm2 = x2 - mu2

        sigma1sigma2 = s1 * s2

        z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / sigma1sigma2

        numerator = torch.exp(-z / (2 * (1 - rho ** 2)))
        denominator = 2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2)

        return numerator/denominator
      
    def joint_pdf2(self, xy):
        x1, x2 = xy[:,0], xy[:,1]
        mu1, mu2, s1, s2, rho = self.mu2x, self.mu2y, self.s2x, self.s2y, self.rho2
        norm1 = x1 - mu1
        norm2 = x2 - mu2

        sigma1sigma2 = s1 * s2

        z = (norm1 / s1) ** 2 + (norm2 / s2) ** 2 - 2 * rho * norm1 * norm2 / sigma1sigma2

        numerator = torch.exp(-z / (2 * (1 - rho ** 2)))
        denominator = 2 * math.pi * sigma1sigma2 * torch.sqrt(1 - rho ** 2)

        return numerator/denominator
      
    def bimodal_joint_pdf(self,xy):
        return self.joint_pdf1(xy) + self.joint_pdf2(xy)

    # Define the negative of the bimodal joint PDF function for maximization
    def negative_bimodal_joint_pdf(self,xy):
        return -self.bimodal_joint_pdf(xy)

    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        self.model.eval()
        # self.model.train()
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            # xy = augmentation.add_noise(xy, thresh=args.thresh, ped=args.ped_type)
            batch_split = [0, xy.shape[1]]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)
            batch_split = torch.Tensor(batch_split).long()

            multimodal_outputs = {}
            # md_outputs = 
            # for num_p in range(modes):
                # _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, xy[obs_length:-1].clone())
            _, output_scenes, normals = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)
            output_scenes = output_scenes.numpy()
            # print(output_scenes)
            md_outputs = np.stack([output_scenes for i in range(8)])
            # print(len(md_outputs))
            # print(len(normals))
            # print(md_outputs[0].shape)
            # print(normals[0].shape)
            # normals = normals.numpy()
            split = 0
            for t in range(len(normals)):
              normal = normals[t]
              normal = normal.numpy()
              curr_pos = output_scenes.shape[0] - len(normals) + t
              prev_pos = curr_pos - 1

              # print(curr_pos, prev_pos)

              # for p in range(normal.size(0)):
              #   mode1 = joint_pdf(normal[p])

              p1 = normal[:,:2]
              p2 = normal[:,5:7]

              prob1 = self.joint_pdf(p1,normal[:,0],normal[:,1],normal[:,2],normal[:,3],normal[:,4])
              prob2 = self.joint_pdf(p2,normal[:,5],normal[:,6],normal[:,7],normal[:,8],normal[:,9])

              prob1 = np.nanmean(prob1)
              prob2 = np.nanmean(prob2)
              # print(prob1, prob2)
              # continue

              if abs(prob1-prob2) <= 20 and split<3:
                split += 1
                if split == 1:
                  md_outputs[0:4, curr_pos] = md_outputs[0:4, prev_pos] + p1
                  md_outputs[4:8, curr_pos] = md_outputs[4:8, prev_pos] + p2
                elif split == 2:
                  md_outputs[0:2, curr_pos] = md_outputs[0:2, prev_pos] + p1
                  md_outputs[2:4, curr_pos] = md_outputs[2:4, prev_pos] + p2
                  md_outputs[4:6, curr_pos] = md_outputs[4:6, prev_pos] + p1
                  md_outputs[6:8, curr_pos] = md_outputs[6:8, prev_pos] + p2
                elif split == 3:
                  md_outputs[0:8:2, curr_pos] = md_outputs[0:8:2,prev_pos] + p1
                  md_outputs[1:8:2, curr_pos] = md_outputs[1:8:2, prev_pos] + p2
              else:
                # print("Here")
                p = p1
                if prob2 > prob1:
                  p = p2
                
                # output_scenes[curr_pos] = output_scenes[prev_pos]
                # print(md_outputs[::, curr_pos])
                # print(md_outputs[::, prev_pos])
                md_outputs[::, curr_pos] = md_outputs[::, prev_pos] + p

              # initial_point_mode1 = output_scenes[curr_pos]
              # initial_point_mode2 = 2*output_scenes[prev_pos] - output_scenes[curr_pos]

              # result_mode1 = minimize(self.negative_bimodal_joint_pdf, initial_point_mode1)

              # # Perform the optimization for mode 2
              # result_mode2 = minimize(self.negative_bimodal_joint_pdf, initial_point_mode2)

              # maximum_point_mode1 = result_mode1.x
              # maximum_point_mode2 = result_mode2.x

              # print(maximum_point_mode1)


            # if args.normalize_scene:
            #     output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)

            for i in range(md_outputs.shape[0]):

              output_primary = md_outputs[i, -n_predict:, 0]
              output_neighs = md_outputs[i, -n_predict:, 1:]
            ## Dictionary of predictions. Each key corresponds to one mode
              multimodal_outputs[i] = [output_primary, output_neighs]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs

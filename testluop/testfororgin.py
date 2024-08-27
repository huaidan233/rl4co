from rl4co.envs.urbanplan.cityplan import init
import torch
from tensordict.tensordict import TensorDict
import torch

from rl4co.envs import landuseOptEnv
from rl4co.envs.routing import TSPEnv
from rl4co.models import AttentionModel, AttentionModelPolicy
from rl4co.models.nn.env_embeddings.context import LOPContext
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.nn.env_embeddings.init import lopInitEmbedding
from rl4co.models.zoo.amppo.model import AMPPO
from rl4co.utils import RL4COTrainer
from rl4co.utils.decoding import random_policy, rollout

''' neighbour to list '''
polygoncount = 221
neighbourlist = init.getneighbourlist('../Data/queen.csv', polygoncount)
basiclanduse = init.readlanduselist('../Data/baseParcels.shp', polygoncount)
landusePalette = {'Commercial': 'coral',
                  'Residential': 'peachpuff',
                  'Office': 'indianred',
                  'Residential&Commercial': 'lightsalmon',
                  'Green Space': 'lightgreen',
                  'Education': 'lightskyblue',
                  'Hospital': 'royalblue',
                  'SOHO': 'lightcoral'
                  }

landtype = ['Commercial', 'Residential', 'Office', 'Residential&Commercial', 'Green Space', 'Education', 'Hospital',
            'SOHO']
adj_matrix = init.get_adjacency_matrix('../Data/queen.csv', polygoncount)
import math
import numpy as np
shapefile = '../Data/Parcels.shp'


arealist = init.readarealist(shapefile, polygoncount)
locs_list = init.normalizeloc(shapefile)
locs = torch.tensor(locs_list, dtype=torch.float32).unsqueeze(0)
areas = torch.tensor(init.normalizearea(arealist), dtype=torch.float32).unsqueeze(0)
init_plan = torch.tensor(init.map_to_num(basiclanduse, landtype), dtype=torch.int64).unsqueeze(0)
fixed_mask = torch.ones_like(init_plan, dtype=torch.bool)
fixed_mask[(init_plan == 4) | (init_plan == 6)] = 0
neighbourlist = torch.tensor(adj_matrix).unsqueeze(0)
distances = torch.tensor(init.calculate_distance_matrix(locs_list), dtype=torch.float32).unsqueeze(0)
td = TensorDict(
    {
        "locs": locs,
        "areas": areas,
        "init_plan": init_plan,
        "fixed_mask": fixed_mask,
        "adjacency_list": neighbourlist,
        "distances": distances,
    },
    batch_size=1,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = landuseOptEnv(generator_params=dict(num_loc=221))
# Instantiate policy with the embeddings we created above
emb_dim = 128
policy = AttentionModelPolicy(env_name=env.name, # this is actually not needed since we are initializing the embeddings!
                              embed_dim=emb_dim,
                              init_embedding=lopInitEmbedding(emb_dim),
                              context_embedding=LOPContext(emb_dim),
                              dynamic_embedding=StaticEmbedding(emb_dim)
)
# Model: default is AM with REINFORCE and greedy rollout baseline
model = AMPPO(env,
                       policy=policy,
                       train_data_size=10,
                       val_data_size=10)

td_init = env.reset(td=td, batch_size=[1]).to(device)
policy = model.policy.to(device)
out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)

actions_untrained = out['actions'].cpu().detach()
rewards_untrained = out['reward'].cpu().detach()

for i in range(1):
    print(f"Problem {i+1} | Cost: {rewards_untrained[i]:.3f}")
    plan = env.render(td_init[i], actions_untrained[i], planout=True)
    init.visuallanduse(shapefile, plan, landusePalette, "temp.png", "reset")




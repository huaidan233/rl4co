import torch
from rl4co.envs.urbanplan.cityplan import init
from rl4co.envs import landuseOptEnv
from rl4co.models import AttentionModel, AttentionModelPolicy
from rl4co.models.nn.env_embeddings.context import LOPContext
from rl4co.models.nn.env_embeddings.dynamic import StaticEmbedding
from rl4co.models.nn.env_embeddings.init import lopInitEmbedding
from rl4co.utils import RL4COTrainer
from rl4co.utils.decoding import random_policy, rollout

batch_size = 3


# env = landuseOptEnv(generator_params=dict(num_loc=50))
# reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
# env.render(td, actions)
# env = TSPEnv(generator_params=dict(num_loc=20))
# reward, td, actions = rollout(env, env.reset(batch_size=[batch_size]), random_policy)
# env.render(td, actions)

# Instantiate our environment
env = landuseOptEnv(generator_params=dict(num_loc=50))

# Instantiate policy with the embeddings we created above
emb_dim = 128
policy = AttentionModelPolicy(env_name=env.name, # this is actually not needed since we are initializing the embeddings!
                              embed_dim=emb_dim,
                              init_embedding=lopInitEmbedding(emb_dim),
                              context_embedding=LOPContext(emb_dim),
                              dynamic_embedding=StaticEmbedding(emb_dim)
)


# Model: default is AM with REINFORCE and greedy rollout baseline
model = AttentionModel(env,
                       policy=policy,
                       baseline='rollout',
                       train_data_size=100_000,
                       val_data_size=10_000)

new_dataset = env.dataset(512, filename='tests/luopt_100.pkl.npz')
dataloader = model._dataloader(new_dataset, batch_size=512)

landtype = ['Commercial', 'Residential', 'Office', 'Residential&Commercial', 'Green Space', 'Education', 'Hospital',
            'SOHO']
# Greedy rollouts over untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Greedy rollouts over trained policy (same states as previous plot, with 20 nodes)
init_states = next(iter(dataloader))[:1]
td_init_generalization = env.reset(init_states).to(device)


policy = model.policy.to(device)
out = policy(td_init_generalization.clone(), env, phase="test", decode_type="greedy", return_actions=True)

# Plotting
print(f"Tour lengths: {[f'{-r.item():.2f}' for r in out['reward']]}")
for td, actions in zip(td_init_generalization, out['actions'].cpu()):
    env.render(td, actions)

td_init = td_init_generalization.clone()
print(td_init[0])
print(td_init[0].get("locs"))
print(td_init[0].get("areas"))
print(td_init[0].get("init_plan"))
print(td_init[0].get("adjacency_list"))
print(td_init[0].get("distances").tolist())
print(init.map_to_strings(td_init[0].get("init_plan").tolist(), landtype))
# policy = model.policy.to(device)
# out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
#
# actions_untrained = out['actions'].cpu().detach()
# rewards_untrained = out['reward'].cpu().detach()
#
# for i in range(3):
#     print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.3f}")
#     env.render(td_init[i], actions_untrained[i])

# # We use our own wrapper around Lightning's `Trainer` to make it easier to use
# trainer = RL4COTrainer(max_epochs=3, device=device)
#
# trainer.fit(model)
#
# # Greedy rollouts over trained policy (same states as previous plot)
# policy = model.policy.to(device)
# out = policy(td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True)
# actions_trained = out['actions'].cpu().detach()
#
# torch.__version__()
# # Plotting
# import matplotlib.pyplot as plt
# for i, td in enumerate(td_init):
#     fig, axs = plt.subplots(1,2, figsize=(11,5))
#     env.render(td, actions_untrained[i], ax=axs[0])
#     env.render(td, actions_trained[i], ax=axs[1])
#     axs[0].set_title(f"Untrained | Cost = {-rewards_untrained[i].item():.3f}")
#     axs[1].set_title(r"Trained $\pi_\theta$" + f"| Cost = {-out['reward'][i].item():.3f}")
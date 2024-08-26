from rl4co.envs.urbanplan.cityplan.env import landuseOptEnv
from rl4co.tasks import eval

env = landuseOptEnv(generator_params=dict(num_loc=100))
new_dataset = env.dataset(512, filename='luopt_100.pkl.npz')
dataloader = model._dataloader(new_dataset, batch_size=512)

eval(
    model, dataloader, env, num_rollouts=10, render=True)

--problem landuseOptEnv --data-path 'luopt_50.pkl.npz' --model AttentionModel --ckpt-path 'tests/checkpoints50/last.ckpt' --method "greedy"
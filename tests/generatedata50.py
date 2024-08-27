from rl4co.data.generate_data import generate_dataset

generate_dataset(filename="./luopt_50_10test1.pkl.npz",
                 data_dir="./data50",
                 problem="luop",
                 dataset_size=10,
                 graph_sizes=[50],
                 )

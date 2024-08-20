from rl4co.data.generate_data import generate_dataset

generate_dataset(filename="./luopt_100.pkl",
                 data_dir="./data50",
                 problem="luop",
                 dataset_size=100000,
                 graph_sizes=[100],
                 )

import os
import pickle
import torch
from torch_geometric.utils import to_undirected, is_undirected
from torch_geometric.seed import seed_everything
from trainer import Trainer
from args_parser import parse_args
from gnn_models import get_model


args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

print(device)


def main():
    args = parse_args()
    args.unlearning_model = 'original'
    args.epochs = 2000
    args.valid_freq = 200
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset, args.gnn, args.unlearning_model,
                                       str(args.random_seed))
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    seed_everything(args.random_seed)

    # Dataset
    with open(os.path.join(args.data_dir, args.dataset, f'd_{args.random_seed}.pkl'), 'rb') as f:
        dataset, data = pickle.load(f)
    print('Directed dataset:', dataset, data)

    args.in_dim = dataset.num_features
    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    train_pos_edge_index = to_undirected(data.train_pos_edge_index)
    data.train_pos_edge_index = train_pos_edge_index
    data.dtrain_mask = torch.ones(data.train_pos_edge_index.shape[1], dtype=torch.bool)
    assert is_undirected(data.train_pos_edge_index)

    print('Undirected dataset:', data)

    # Model
    model = get_model(args, num_nodes=data.num_nodes, num_edge_type=args.num_edge_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    trainer = Trainer(args)
    trainer.train(model, data, optimizer, args)

    # Test
    trainer.test(model, data)
    trainer.save_log()


if __name__ == "__main__":
    main()

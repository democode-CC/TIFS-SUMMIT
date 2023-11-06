import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device",
                        type=str,
                        default='cuda:0',
                        help="The GPU is used.")

    parser.add_argument('--attack', type=bool, default=False,
                        help='whether to MI attack')

    # MooGIB (our method)
    parser.add_argument('--lmd', type=str, default='moo',
                        help='0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1 and moo')

    # Ablation
    parser.add_argument('--HFE', type=int, default=1,
                        help='FFE')
    parser.add_argument('--RMB', type=int, default=1,
                        help='RMB')

    # Model
    parser.add_argument('--unlearning_model', type=str, default='SUMMIT',
                        help='original, retrain, SUMMIT')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN architecture, gcn, gat, gin')
    parser.add_argument('--in_dim', type=int, default=128,
                        help='input dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden dimension')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='output dimension')

    # Data
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='dataset. Cora, PubMed, DBLP, CS. case sensitive!')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='data dir')
    parser.add_argument('--df', type=str, default='none',
                        help='Df set to use, in or out')
    parser.add_argument('--df_idx', type=str, default='none',
                        help='indices of data to be deleted')
    parser.add_argument('--df_size', type=float, default=5,
                        help='Df size')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed 42 21 13 87 100')


    parser.add_argument('--batch_size', type=int, default=8192,
                        help='batch size for GraphSAINTRandomWalk sampler')
    parser.add_argument('--walk_length', type=int, default=2,
                        help='random walk length for GraphSAINTRandomWalk sampler')
    parser.add_argument('--num_steps', type=int, default=32,
                        help='number of steps for GraphSAINTRandomWalk sampler')

    # Training
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train, default=3000')
    parser.add_argument('--valid_freq', type=int, default=50,
                        help='# of epochs to do validation')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint',
                        help='checkpoint folder')

    parser.add_argument('--shadow_dir', type=str, default='',
                        help='shadow model folder')
    parser.add_argument('--attack_dir', type=str, default='',
                        help='attack model folder')

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha in loss function')
    parser.add_argument('--neg_sample_random', type=str, default='non_connected',
                        help='type of negative samples for randomness')
    parser.add_argument('--loss_fct', type=str, default='mse_mean',
                        help='loss function. one of {mse, kld, cosine}')


    # Evaluation
    parser.add_argument('--topk', type=int, default=500,
                        help='top k for evaluation')
    parser.add_argument('--eval_on_cpu', type=bool, default=False,
                        help='whether to evaluate on CPU')

    # KG
    parser.add_argument('--num_edge_type', type=int, default=None,
                        help='number of edges types')

    args = parser.parse_args()




    if args.unlearning_model in ['original', 'retrain']:
        args.epochs = 2000
        args.valid_freq = 500

    return args

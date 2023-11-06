import os
import time
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, to_dense_adj

import random
import dgl

from trainer import Trainer
from evaluation import *
from utils import *

from gnn_models import GCN, GAT, GIN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SummitTrainer(Trainer):
    def train(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None, attack_model_sub=None):
        return self.train_fullbatch(model, data, optimizer, args, logits_ori, attack_model_all, attack_model_sub)

    def gather_rows(self, input, index):
        """Gather the rows specificed by index from the input tensor"""
        return torch.gather(input, 0, index.unsqueeze(-1).expand((-1, input.shape[1])))

    def log_transformation_HFE(self, x):
        x = torch.exp(-torch.log(0.1 + x))
        return x

    def HFE_(self, args, mu, sigma, u_index, v_index, sp):
        mu_u = self.gather_rows(mu, u_index)
        mu_v = self.gather_rows(mu, v_index)

        sigma_u = self.gather_rows(sigma, u_index)
        sigma_v = self.gather_rows(sigma, v_index)

        diff_uv = mu_u - mu_v
        ob = torch.abs(diff_uv).mean()
        ratio_vu = sigma_v / sigma_u
        kld = 0.5 * (
                ratio_vu.sum(axis=-1)
                + (diff_uv ** 2 / sigma_u).sum(axis=-1)
                - args.hidden_dim / 2
                - torch.log(ratio_vu).sum(axis=-1)
        )

        # shortest path is obtained by DGL shortest_list function, older version may not support this function.
        sp = sp[u_index][v_index]

        return kld @ sp

    def RMB_(self, args, mu, sigma, mu_moo, sigma_moo):
        diff = mu_moo - mu
        ratio = sigma / sigma_moo
        kld = 0.5 * torch.sum(
            ratio.sum(axis=-1) + (diff ** 2 / sigma_moo).sum(axis=-1) - (args.hidden_dim / 2) - torch.log(ratio).sum(
                axis=-1))
        return kld

    def freeze_param(self, model):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            self.freeze_param(child)

    def reparameterize(self, mu, logvar, training_status):
        if training_status:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def get_model(self, args, data):
        mask_1hop = data.sdf_node_1hop_mask
        mask_2hop = data.sdf_node_2hop_mask
        num_nodes = data.num_nodes
        num_edge_type = args.num_edge_type
        model_mapping = {'gcn': GCN, 'gat': GAT, 'gin': GIN}

        return model_mapping[args.gnn](args, mask_1hop=mask_1hop, mask_2hop=mask_2hop, num_nodes=num_nodes,
                                       num_edge_type=num_edge_type)

    def train_fullbatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None,
                        attack_model_sub=None):

        loss_fct = nn.MSELoss(reduction='mean')
        model.eval()

        g = dgl.graph(tuple(data.edge_index.tolist()))

        sp = dgl.shortest_dist(g, root=0)


        if self.args.dataset == 'Cora' and self.args.gnn == 'gin':
            model.cpu()
            with torch.no_grad():
                _, mu, sigma = model.encode(data.x, data.train_pos_edge_index[:, data.dr_mask])
                z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])
            z = z.to(self.args.device)
            mu = mu.to(self.args.device)
            sigma = sigma.to(self.args.device)
            model = model.to(self.args.device)  # learned
            data = data.to(self.args.device)
        else:
            model = model.to(self.args.device)  # learned
            data = data.to(self.args.device)
            with torch.no_grad():
                _, mu, sigma = model.encode(data.x, data.train_pos_edge_index[:, data.dr_mask])
                z = model(data.x, data.train_pos_edge_index[:, data.dr_mask])

        z_ori = z

        # random edge index
        random_neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.df_mask.sum())

        moo = self.get_model(args, data).to(self.args.device)
        moo.load_state_dict(model.state_dict())
        optimizer_moo = torch.optim.Adam(moo.parameters(), lr=args.lr)

        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

        model.eval()
        self.freeze_param(model)


        best_metric = 0
        if attack_model_all is not None:
            mi_logit_all_before, mi_sucrate_all_before = member_infer_attack(model, attack_model_all, data)
            self.trainer_log['mi_logit_all_before'] = mi_logit_all_before
            self.trainer_log['mi_sucrate_all_before'] = mi_sucrate_all_before
        if attack_model_sub is not None:
            mi_logit_sub_before, mi_sucrate_sub_before = member_infer_attack(model, attack_model_sub, data)
            self.trainer_log['mi_logit_sub_before'] = mi_logit_sub_before
            self.trainer_log['mi_sucrate_sub_before'] = mi_sucrate_sub_before

        self.trainer_log['train_loss_all'] = []
        for epoch in trange(args.epochs, desc='Unlearning'):
            moo.train()
            start_time = time.time()
            total_step = 0
            total_loss = 0

            neg_edge_index = negative_sampling(
                edge_index=data.train_pos_edge_index[:, data.dr_mask],
                num_nodes=data.num_nodes,
                num_neg_samples=data.dr_mask.sum())

            with torch.no_grad():
                logits_y_r = model.decode(z, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)

            u_index = data.train_pos_edge_index[:, data.df_mask][0]  # deleted node-level index u
            v_index = data.train_pos_edge_index[:, data.df_mask][1]  # deleted node-level index v
            _, mu_moo, sigma_moo = moo.encode(data.x, data.train_pos_edge_index[:, data.dr_mask])
            loss_fgt_hfe = torch.clamp(
                self.log_transformation_HFE(torch.mean(self.HFE_(args, mu_moo, sigma_moo, u_index, v_index, sp))), min=0)  #
            z_moo = moo(data.x, data.train_pos_edge_index[:, data.dr_mask])
            training_status = moo.training


            # GA
            logits_y_f_moo = moo.decode(z_moo, data.train_pos_edge_index[:, data.df_mask])
            label = torch.ones_like(logits_y_f_moo, dtype=torch.float, device=self.args.device)
            loss_fgt_ga = - F.binary_cross_entropy_with_logits(logits_y_f_moo, label)
            loss_fgt_ga = torch.exp(-torch.log(-loss_fgt_ga + 0.1))

            # Loss Con
            cat_embed_ori = torch.cat([z_ori[random_neg_edge_index[0]], z_ori[random_neg_edge_index[1]]], dim=0)
            cat_embed_unl = torch.cat([z_moo[data.train_pos_edge_index[:, data.df_mask][0]],
                                       z_moo[data.train_pos_edge_index[:, data.df_mask][1]]], dim=0)
            loss_fgt_con = loss_fct(cat_embed_unl, cat_embed_ori)  # z:unl z_ori:lrn

            ## loss_fgt
            lft1 = self.args.HFE
            lft2 = 1
            loss_fgt = lft1 * loss_fgt_hfe + lft2 * loss_fgt_ga + loss_fgt_con
            print('\nloss_fgt_hfe\n', loss_fgt_hfe)
            print('loss_fgt_ga\n', loss_fgt_ga)
            print('loss_fgt_con\n', loss_fgt_con)
            # ------------------------------------------

            # -------------remembering loss-------------
            loss_rmb_t1 = torch.mean(self.RMB_(args, mu, sigma, mu_moo, sigma_moo))

            ## loss_rmb_t2

            logits_y_r_moo = moo.decode(z_moo, data.train_pos_edge_index[:, data.dr_mask], neg_edge_index)
            loss_rmb_t2 = kl_loss(F.log_softmax(logits_y_r_moo, dim=-1), F.log_softmax(logits_y_r,
                                                                                       dim=-1))
            lrm1 = self.args.RMB
            lrm2 = 0
            loss_rmb = lrm1 * loss_rmb_t1 + lrm2 * loss_rmb_t2
            print('loss_rmb_t1\n', lrm1 * loss_rmb_t1)
            print('loss_rmb_t2\n', lrm2 * loss_rmb_t2)
            # ------------------------------------------

            if args.lmd == 'moo':
                if loss_fgt <= 0:
                    lmd = torch.abs(loss_fgt) / (torch.abs(loss_fgt) + loss_rmb)
                else:
                    lmd = torch.abs(loss_fgt) / (torch.abs(loss_fgt) + loss_rmb)
            else:
                lmd = float(args.lmd)

            # --------Final unlearning loss---------
            print('fgt loss:', loss_fgt)
            print('rmb loss:', loss_rmb)
            print('lmd', lmd)
            loss = lmd * loss_fgt + (1 - lmd) * loss_rmb
            self.trainer_log['train_loss_all'].append(loss.item())
            print()

            print('loss', loss)
            loss.backward()
            optimizer_moo.step()
            optimizer_moo.zero_grad()

            total_step += 1
            total_loss += loss.item()

            end_time = time.time()
            epoch_time = end_time - start_time

            step_log = {
                'Epoch': epoch,
                'train_loss': loss.item(),
                'train_time': epoch_time
            }

            msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in step_log.items()]
            tqdm.write(' | '.join(msg))

            if (epoch + 1) % self.args.valid_freq == 0:
                valid_loss, dt_auc, dt_aup, df_auc, df_aup, df_con_auc, df_con_mse, df_logit, logit_all_pair, valid_log = self.eval(
                    moo, data, 'val')
                valid_log['epoch'] = epoch

                train_log = {
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_time': epoch_time,
                }

                for log in [train_log, valid_log]:
                    # wandb.log(log)
                    msg = [f'{i}: {j:>4d}' if isinstance(j, int) else f'{i}: {j:.4f}' for i, j in log.items()]
                    tqdm.write(' | '.join(msg))
                    self.trainer_log['log'].append(log)

                metric = dt_auc + df_auc
                # metric = dt_auc
                if metric > best_metric:
                    cat_embed_unl_non = torch.cat([z_moo[random_neg_edge_index[0]], z_ori[random_neg_edge_index[1]]],
                                                  dim=0).detach().cpu().numpy()
                    cat_embed_unl_df = torch.cat([z_moo[data.train_pos_edge_index[:, data.df_mask][0]],
                                                  z_moo[data.train_pos_edge_index[:, data.df_mask][1]]],
                                                 dim=0).detach().cpu().numpy()

                    logtis_unl_non = moo.decode(z_moo, random_neg_edge_index).detach().cpu().numpy()
                    logits_unl_df = moo.decode(z_moo, data.train_pos_edge_index[:, data.df_mask]).detach().cpu().numpy()

                    np.save(f'tsne/cat_{args.dataset}_{args.gnn}_non', cat_embed_unl_non)
                    np.save(f'tsne/cat_{args.dataset}_{args.gnn}_df', cat_embed_unl_df)
                    np.save(f'tsne/logits_{args.dataset}_{args.gnn}_non', logtis_unl_non)
                    np.save(f'tsne/logits_{args.dataset}_{args.gnn}_df', logits_unl_df)

                    best_metric = metric
                    best_epoch = epoch

                    print(f'Save best checkpoint at epoch {epoch:04d}. Valid loss = {valid_loss:.4f}')
                    ckpt = {
                        'model_state': moo.state_dict(),
                        'optimizer_state': optimizer_moo.state_dict(),
                    }
                    if args.lmd == 'moo':
                        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_best.pt'))
                    else:
                        torch.save(ckpt, os.path.join(args.checkpoint_dir, f'model_best_{args.lmd}.pt'))

        self.trainer_log['training_time'] = time.time() - start_time

        # Save
        ckpt = {
            'model_state': moo.state_dict(),

            'optimizer_state': optimizer_moo.state_dict(),
        }
        if args.lmd == 'moo':
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'model_final.pt'))
        else:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, f'model_final_{args.lmd}.pt'))

        print(f'Training finished. Best checkpoint at epoch = {best_epoch:04d}, best metric = {best_metric:.4f}')

        self.trainer_log['best_epoch'] = best_epoch
        self.trainer_log['best_metric'] = best_metric

    def train_minibatch(self, model, data, optimizer, args, logits_ori=None, attack_model_all=None,
                        attack_model_sub=None):
        pass


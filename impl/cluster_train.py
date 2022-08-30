import torch
from torch_sparse import SparseTensor
import time
def cluster_train_routine(dsname, mod, opt, trn_ds, val_ds, tst_ds, epoch, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    trn_ds.pos1 = trn_ds.pos1.to(torch.long)
    val_ds.pos1 = val_ds.pos1.to(torch.long)
    tst_ds.pos1 = tst_ds.pos1.to(torch.long)
    batch_size = val_ds.y.shape[0]
    vprint(f"batch size{batch_size}")

    best_val = 0
    tst_score = 0
    early_stop = 0
    early_stop_thd = 800
    for i in range(epoch):
        train_idx = 0
        t0 = time.time()
        loss, trn_score, train_idx = train(mod, opt, trn_ds, batch_size, train_idx)
        t1 = time.time()
        val_score = test(mod, val_ds)
        vprint(f"epoch: {i:03d}, trn: time {t1 - t0:.2f} s, loss {loss:.4f}, trn {trn_score:.4f}, val {val_score:.4f}",
               end=" ")
        early_stop += 1
        if val_score > best_val:
            early_stop = 0
            best_val = val_score
            if verbose:
                t0 = time.time()
                tst_score = test(mod, tst_ds, True)
                t1 = time.time()
                #vprint(f"time:{t1-t0:.4f}")
            vprint(f"tst {tst_score:.4f}")
        else:
            vprint()
        if early_stop > early_stop_thd:
            break
    vprint(f"end test {tst_score:.3f}")
    if verbose:
        with open(f'./records/{dsname}_auc_record.txt', 'a') as f:
            f.write('AUC:' + str(round(tst_score, 4)) + '   ' + 'Time:' + str(
                    round(t1 - t0, 4)) + '   ' + '\n')
    return best_val

def train(model, predictor, clusterloader, optimizer, device):
    model.train()

    total_loss = total_examples = 0
    for data in clusterloader:
        data = data.to(device)
        optimizer.zero_grad()

        h = model(data.x, data.edge_index)

        src, dst = data.edge_index
        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.x.size(0), src.size(),
                                dtype=torch.long, device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, device):
    predictor.eval()
    x = data.x.numpy()
    adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
    adj = adj.set_diag()
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj = adj.to_scipy(layout='csr')

    h = torch.from_numpy(model(x, adj)).to(device)

    def test_split(split):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        from torch.utils.data import DataLoader
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr

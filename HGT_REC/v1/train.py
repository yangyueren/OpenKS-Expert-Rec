import dgl
import torch.nn as nn
import pickle
import numpy as np
import argparse
from model import *
from torch.utils.data import DataLoader
from dataset import NSFDataset
from tqdm import tqdm
from metrics import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training HGT')

    parser.add_argument('--n_epoch', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--n_hid', type=int, default=256)
    parser.add_argument('--n_dim', type=int, default=256)
    parser.add_argument('--clip', type=int, default=5.0)
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='initial learning rate')
    parser.add_argument('--max_project', type=int, default=5,
                        help='max number of projects the project related to')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use GPU for training')
    parser.add_argument('--topk', type=int, default=10,
                        help="compute metrics@top_k")
    parser.add_argument('--decay', type=float, default=0.98,
                        help='learning rate decay rate')
    parser.add_argument('--decay_step', type=int, default=1,
                        help='learning rate decay step')
    parser.add_argument('--log_step', type=int, default=1e2,
                        help='log print step')
    parser.add_argument('--patience', type=int, default=10,
                        help='extra iterations before early-stopping')
    parser.add_argument('--save', type=str, default='model/',
                        help='path to save the final model')

    args = parser.parse_args()
    args.save = args.save + '_n_epoch{}'.format(args.n_epoch)
    args.save = args.save + '_n_hid{}'.format(args.n_hid)
    args.save = args.save + '_n_dim{}'.format(args.n_dim)
    args.save = args.save + '_lr{}'.format(args.lr)
    args.save = args.save + 'clip{}_model.pt'.format(args.clip)
    return args

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def Calculate_Similarity(project_text_emb, proj_text_emb, args):
    project_text_emb = project_text_emb.unsqueeze(1)
    score = torch.sum(project_text_emb * proj_text_emb, -1)
    _, indices = torch.topk(score, args.max_project)
    return indices

def train(model, G, train_projects_text_emb, train_data_loader, valid_data_loader, test_data_loader, device, args):
    best_ndcg = 0.0
    best_epoch = -1
    n = len(train_data_loader)
    loss_fn = nn.BCELoss().to(device)
    proj_text_emb = torch.Tensor(np.array(list(train_projects_text_emb.values())))
    proj_text_id = torch.LongTensor(np.array(list(train_projects_text_emb.keys()))).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)

    # eval(model, args, valid_data_loader, proj_text_emb, proj_text_id)
    for epoch in np.arange(args.n_epoch) + 1:
        print('Start epoch: ', epoch)
        model.train()
        for step, batch_data in tqdm(enumerate(train_data_loader), total = n):
            project_id, project_text_emb, pos_person, neg_person_list = batch_data
            batch_size = project_id.shape[0]
            pos_label = torch.ones(batch_size).to(device)
            neg_label = torch.zeros(batch_size).to(device)

            indices = Calculate_Similarity(project_text_emb, proj_text_emb, args)

            project_emb = model(G, 'project')
            person_emb = model(G, 'person')

            cur_emb = torch.zeros(batch_size, args.n_dim).to(device)
            for i in range(args.max_project):
                similar_id = proj_text_id[indices[:,i]]
                cur_emb += project_emb[similar_id]

            cur_emb /= args.max_project

            pos_score = torch.sigmoid(torch.sum(cur_emb * person_emb[pos_person], -1))
            neg_score = torch.sigmoid(torch.sum(cur_emb * person_emb[neg_person_list[0]], -1))
            pos_loss = loss_fn(pos_score, pos_label)
            neg_loss = loss_fn(neg_score, neg_label)
            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        scheduler.step()
        mean_p, mean_r, mean_h, mean_ndcg = eval(model, args, valid_data_loader, proj_text_emb, proj_text_id)
        print(f'Valid:\tprecision@{args.topk}:{mean_p:.6f}, recall@{args.topk}:{mean_r:.6f}, '
              f'hr@{args.topk}:{mean_h:.6f}, ndcg@{args.topk}:{mean_ndcg:.6f}')
        if mean_ndcg > best_ndcg:
            best_epoch = epoch
            best_ndcg = mean_ndcg
            model.save(args.save)
            print('Model save for higher ndcg %f in %s' % (best_ndcg, args.save))
        if epoch - best_epoch >= args.patience:
            print('Stop training after %i epochs without improvement on validation.' % args.patience)
            break
    model.load(args.save)
    mean_p, mean_r, mean_h, mean_ndcg = eval(model, args, test_data_loader, proj_text_emb, proj_text_id)
    print(f'Test:\tprecision@{args.topk}:{mean_p:.6f}, recall@{args.topk}:{mean_r:.6f}, '
          f'hr@{args.topk}:{mean_h:.6f}, ndcg@{args.topk}:{mean_ndcg:.6f}')


def eval(model, args, eval_data_loader, proj_text_emb, proj_text_id):
    model.eval()
    eval_p = []
    eval_r = []
    eval_h = []
    eval_ndcg = []
    eval_len = []
    with torch.no_grad():
        for step, batch_data in enumerate(eval_data_loader):
            project_id, project_text_emb, pos_person, neg_person_list = batch_data
            batch_size = project_id.shape[0]
            neg_person_list = torch.transpose(torch.stack(neg_person_list), 0, 1)
            pos_person = pos_person.unsqueeze(1)

            indices = Calculate_Similarity(project_text_emb, proj_text_emb, args)

            project_emb = model(G, 'project')
            person_emb = model(G, 'person')

            cur_emb = torch.zeros(batch_size, args.n_dim)
            for i in range(args.max_project):
                similar_id = proj_text_id[indices[:, i]]
                cur_emb += project_emb[similar_id]
            cur_emb /= args.max_project

            person_list = torch.cat((pos_person, neg_person_list), dim=1)
            cur_emb = cur_emb.unsqueeze(1)
            score = torch.sigmoid(torch.sum(cur_emb * person_emb[person_list], -1))
            pred_person_index = torch.topk(score, args.topk)[1].tolist()

            for i in range(batch_size):
                p_at_k = getP(pred_person_index[i], [0])
                r_at_k = getR(pred_person_index[i], [0])
                h_at_k = getHitRatio(pred_person_index[i], [0])
                ndcg_at_k = getNDCG(pred_person_index[i], [0])
                eval_p.append(p_at_k)
                eval_r.append(r_at_k)
                eval_h.append(h_at_k)
                eval_ndcg.append(ndcg_at_k)
                eval_len.append(1)
            if (step % args.log_step == 0) and step > 0:
                print('Valid epoch:[{}/{} ({:.0f}%)]\t Recall: {:.6f}, AvgRecall: {:.6f}'.format(step, len(eval_data_loader),
                                                            100. * step / len(eval_data_loader), r_at_k, np.mean(eval_r)))
        mean_p = np.mean(eval_p)
        mean_r = np.mean(eval_r)
        mean_h = np.sum(eval_h) / np.sum(eval_len)
        mean_ndcg = np.mean(eval_ndcg)
        return mean_p, mean_r, mean_h, mean_ndcg

if __name__ == '__main__':
    torch.manual_seed(0)
    args = parse_args()
    root_path = '/home/disk1/ls/project/zheda_nsf/'
    index_path = root_path + 'data/index.pkl'
    dgl_data_path = root_path + 'data/dgl_data.pkl'
    emb_data_path = root_path + 'data/projects_text_emb.pkl'
    train_data_path = root_path + 'data/train_dataset.pkl'
    valid_data_path = root_path + 'data/valid_dataset.pkl'
    test_data_path = root_path + 'data/test_dataset.pkl'

    print('start')

    with open(index_path, 'rb') as f:
        project2index = pickle.load(f)
        index2project = pickle.load(f)
        paper2index = pickle.load(f)
        index2paper = pickle.load(f)
        person2index = pickle.load(f)
        index2person = pickle.load(f)

    with open(dgl_data_path, 'rb') as f:
        project_main_row = pickle.load(f)
        person_main_col = pickle.load(f)
        project_co_row = pickle.load(f)
        person_co_col = pickle.load(f)
        paper_ref_row = pickle.load(f)
        paper_ref_col = pickle.load(f)
        paper_auther_row = pickle.load(f)
        author_col = pickle.load(f)

    with open(emb_data_path, "rb") as f:
        train_projects_text_emb = pickle.load(f)

    with open(train_data_path, "rb") as f:
        train_data = pickle.load(f)

    with open(valid_data_path, "rb") as f:
        valid_data = pickle.load(f)

    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    use_cuda = torch.cuda.is_available() and args.cuda
    # device = torch.device('cuda' if use_cuda else 'cpu')
    # print(device)
    device = torch.device('cpu')

    train_data_loader = DataLoader(
        dataset=NSFDataset(train_data, args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    valid_data_loader = DataLoader(
        dataset=NSFDataset(valid_data, args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    test_data_loader = DataLoader(
        dataset=NSFDataset(test_data, args),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    G = dgl.heterograph({
        ('project', 'investigated-by', 'person'): (project_main_row, person_main_col),
        ('person', 'investigate', 'project'): (person_main_col, project_main_row),
        ('project', 'co-investigated-by', 'person'): (project_co_row, person_co_col),
        ('person', 'co-investigate', 'project'): (person_co_col, project_co_row),
        ('paper', 'writed-by', 'person'): (paper_auther_row, author_col),
        ('person', 'write', 'paper'): (author_col, paper_auther_row),
    })

    # G = dgl.heterograph({
    #     ('project', 'investigated-by', 'person'): (project_main_row, person_main_col),
    #     ('person', 'investigate', 'project'): (person_main_col, project_main_row),
    #     ('project', 'co-investigated-by', 'person'): (project_co_row, person_co_col),
    #     ('person', 'co-investigate', 'project'): (person_co_col, project_co_row),
    #     ('paper', 'cite', 'paper'): (paper_ref_row, paper_ref_col),
    #     ('paper', 'cited-by', 'paper'): (paper_ref_col, paper_ref_row),
    #     ('paper', 'writed-by', 'person'): (paper_auther_row, author_col),
    #     ('person', 'write', 'paper'): (author_col, paper_auther_row),
    # })
    print(G)

    node_dict = {}
    edge_dict = {}
    print(G.ntypes)
    print(G.etypes)

    for ntype in G.ntypes:
        node_dict[ntype] = len(node_dict)
    for etype in G.etypes:
        edge_dict[etype] = len(edge_dict)
        # G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

    # Random initialize input feature
    for ntype in G.ntypes:
        emb = nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), args.n_dim)) #, requires_grad=False
        nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['inp'] = emb

    G = G.to(device)

    hgt_model = HGT(G,
                node_dict, edge_dict,
                n_inp=args.n_dim,
                n_hid=args.n_hid,
                n_out=args.n_dim,
                n_layers=2,
                n_heads=4,
                use_norm=True).to(device)


    print('Training HGT with #param: %d' % (get_n_params(hgt_model)))
    train(hgt_model, G, train_projects_text_emb, train_data_loader, valid_data_loader, test_data_loader, device, args)




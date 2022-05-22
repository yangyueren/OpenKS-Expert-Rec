import dgl
import torch.nn as nn
import pickle
import json
import numpy as np
import argparse
from model import *
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from dataset import NSFDataset
from tqdm import tqdm
from metrics import *


def parse_args():
    parser = argparse.ArgumentParser(description='Training HGT')

    parser.add_argument('--n_epoch', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--n_hid', type=int, default=192)
    parser.add_argument('--n_dim', type=int, default=768)
    parser.add_argument('--clip', type=int, default=1.0)
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('--max_project', type=int, default=3,
                        help='max number of projects the project related to')
    parser.add_argument('--n_max_neigh', type=int, default=[5, 5],
                        help='max number of neighs for each layer')
    parser.add_argument('--n_neigh_layer', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--n_head', type=int, default=4,
                        help='number of head')
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
    args.save = args.save + '_max_project{}'.format(args.max_project)
    args.save = args.save + '_n_head{}'.format(args.n_head)
    args.save = args.save + '_clip{}_model.pt'.format(args.clip)
    return args

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def graph_collate(batch):
    project_id = default_collate([item[0] for item in batch])
    batch_sub_g = dgl.batch([item[1] for item in batch])
    similar_id = default_collate([item[2] for item in batch])
    emb_weight = default_collate([item[3] for item in batch])
    pos_person = default_collate([item[4] for item in batch])
    neg_person_list = default_collate([item[5] for item in batch])
    return project_id, batch_sub_g, similar_id, emb_weight, pos_person, neg_person_list

def train(model, train_data_loader, valid_data_loader, test_data_loader, device, args):
    best_ndcg = 0.0
    best_epoch = -1
    n = len(train_data_loader)
    loss_fn = nn.BCELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay)

    # eval(model, args, valid_data_loader)
    for epoch in np.arange(args.n_epoch) + 1:
        print('Start epoch: ', epoch)
        model.train()
        for step, batch_data in tqdm(enumerate(train_data_loader), total = n):
            project_id, sub_g, similar_id, emb_weight, pos_person, neg_person_list = batch_data
            batch_size = project_id.shape[0]
            pos_label = torch.ones(batch_size).to(device)
            neg_label = torch.zeros(batch_size).to(device)
            sub_g = sub_g.to(device)

            emb_weight = nn.functional.softmax(emb_weight,dim=1)

            project_emb, person_emb = model(sub_g, 'project', 'person')


            cur_emb = torch.zeros(batch_size, args.n_dim).to(device)
            for i in range(batch_size):
                for j in range(args.max_project):
                    cur_emb[i] += emb_weight[i][j] * project_emb[similar_id[i][j].item()]

            pos_person_emb = []
            for i in range(batch_size):
                pos_person_emb.append(person_emb[pos_person[i].item()])
            pos_person_emb = torch.stack(pos_person_emb)
            neg_person_emb = []
            for i in range(batch_size):
                neg_person_emb.append(person_emb[neg_person_list[i][0].item()])
            neg_person_emb = torch.stack(neg_person_emb)

            pos_score = torch.sigmoid(torch.sum(cur_emb * pos_person_emb, -1))
            neg_score = torch.sigmoid(torch.sum(cur_emb * neg_person_emb, -1))
            # pos_score = torch.sigmoid(torch.sum(cur_emb * person_emb[pos_person], -1))
            # neg_score = torch.sigmoid(torch.sum(cur_emb * person_emb[neg_person_list[0]], -1))
            pos_loss = loss_fn(pos_score, pos_label)
            neg_loss = loss_fn(neg_score, neg_label)
            loss = pos_loss + neg_loss
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
        scheduler.step()
        mean_p, mean_r, mean_h, mean_ndcg = eval(model, args, valid_data_loader)
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
    mean_p, mean_r, mean_h, mean_ndcg = eval(model, args, test_data_loader)
    print(f'Test:\tprecision@{args.topk}:{mean_p:.6f}, recall@{args.topk}:{mean_r:.6f}, '
          f'hr@{args.topk}:{mean_h:.6f}, ndcg@{args.topk}:{mean_ndcg:.6f}')


def eval(model, args, eval_data_loader):
    model.eval()
    eval_p = []
    eval_r = []
    eval_h = []
    eval_ndcg = []
    eval_len = []
    n = len(eval_data_loader)
    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(eval_data_loader), total = n):
            project_id, sub_g, similar_id, emb_weight, pos_person, neg_person_list = batch_data
            sub_g = sub_g.to(device)
            batch_size = project_id.shape[0]
            emb_weight = nn.functional.softmax(emb_weight, dim=1)

            project_emb, person_emb = model(sub_g, 'project', 'person')

            cur_emb = torch.zeros(batch_size, args.n_dim).to(device)
            for i in range(batch_size):
                for j in range(args.max_project):
                    cur_emb[i] += emb_weight[i][j] * project_emb[similar_id[i][j].item()]

            neg_person_list = torch.transpose(torch.stack(neg_person_list), 0, 1)
            pos_person = pos_person.unsqueeze(1)
            person_list = torch.cat((neg_person_list, pos_person), dim=1)
            cur_emb = cur_emb.unsqueeze(1)

            eval_person_emb = []
            for i in range(batch_size):
                temp =[]
                for j in range(person_list.size(1)):
                    temp.append(person_emb[person_list[i][j].item()])
                temp = torch.stack(temp)
                eval_person_emb.append(temp)
            eval_person_emb = torch.stack(eval_person_emb)

            score = torch.sigmoid(torch.sum(cur_emb * eval_person_emb, -1))
            pred_person_index = torch.topk(score, args.topk)[1].tolist()
            for i in range(batch_size):
                p_at_k = getP(pred_person_index[i], [99])
                r_at_k = getR(pred_person_index[i], [99])
                h_at_k = getHitRatio(pred_person_index[i], [99])
                ndcg_at_k = getNDCG(pred_person_index[i], [99])
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

def inference_expert(model, args, infer_data):
    model.eval()
    project_id, sub_g, similar_id, person_list = infer_data
    sub_g = sub_g.to(device)
    batch_size = project_id.shape[0]
    project_emb, person_emb = model(sub_g, 'project', 'person')

    cur_emb = torch.zeros(batch_size, args.n_dim).to(device)
    for i in range(batch_size):
        for j in range(args.max_project):
            cur_emb[i] += project_emb[similar_id[i][j].item()]
    cur_emb /= args.max_project

    cur_emb = cur_emb.unsqueeze(1)

    eval_person_emb = []
    for i in range(batch_size):
        temp =[]
        for j in range(person_list.size(1)):
            temp.append(person_emb[person_list[i][j].item()])
        temp = torch.stack(temp)
        eval_person_emb.append(temp)
    eval_person_emb = torch.stack(eval_person_emb)

    score = torch.sigmoid(torch.sum(cur_emb * eval_person_emb, -1))
    pred_person_index = torch.max(score)[1]
    return pred_person_index

if __name__ == '__main__':
    torch.manual_seed(0)
    args = parse_args()
    root_path = '/home/disk1/ls/project/zheda_nsf/data/'
    # root_path = '/home1/ls/zheda_nsf_10_129_server/'
    # index_path = root_path + 'data/index.pkl'
    # dgl_data_path = root_path + 'data/dgl_data.pkl'
    # emb_data_path = root_path + 'data/projects_text_emb.pkl'
    # train_data_path = root_path + 'data/train_dataset.pkl'
    # valid_data_path = root_path + 'data/valid_dataset.pkl'
    # test_data_path = root_path + 'data/test_dataset.pkl'
    with open(root_path + 'entities_paper.pkl', 'rb') as f:
        papers = pickle.load(f)
    with open(root_path + 'entities_person.pkl', 'rb') as f:
        persons = pickle.load(f)
    with open(root_path + 'entities_project.pkl', 'rb') as f:
        project = pickle.load(f)
    with open(root_path + 'train_rel_is_principal_investigator_of.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(root_path + 'val_rel_is_principal_investigator_of.pkl', 'rb') as f:
        valid_data = pickle.load(f)
    with open(root_path + 'test_rel_is_principal_investigator_of.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open(root_path + 'project_emb_bert.pkl', 'rb') as f:
        pro_emb_data = pickle.load(f)
    with open(root_path + 'paper_emb_bert.pkl', 'rb') as f:
        paper_emb_data = pickle.load(f)

    train_project = []
    for train_one in train_data:
        train_project.append(train_one[2])

    projects = [json.loads(p) for p in project]
    projects = sorted(projects, key=lambda x: x['AwardID'])
    project2index = {}
    index2project = {}
    for index in range(len(projects)):
        project2index[projects[index]['AwardID']] = index
        index2project[index] = projects[index]['AwardID']

    papers = sorted(papers, key=lambda x: x['_id'])
    paper2index = {}
    index2paper = {}
    for index in range(len(papers)):
        paper2index[papers[index]['_id']] = index
        index2paper[index] = papers[index]['_id']

    person2index = {}
    index2person = {}
    index = 0
    for id in persons:
        person2index[id] = index
        index2person[index] = id
        index += 1

    project_main_row = []
    person_main_col = []
    project_co_row = []
    person_co_col = []
    for project in projects:
        if project['AwardID'] in train_project:
            for role in project['Investigator']:
                if role['RoleCode'] == 'Principal Investigator':
                    project_main_row.append(project2index[project['AwardID']])
                    person_main_col.append(person2index[role['uid']])
                elif role['RoleCode'] == 'Co-Principal Investigator':
                    project_co_row.append(project2index[project['AwardID']])
                    person_co_col.append(person2index[role['uid']])

    paper_auther_row = []
    author_col = []
    paper_ref_row = []
    paper_ref_col = []
    for paper in papers:
        for author in paper['authors']:
            paper_auther_row.append(paper2index[paper['_id']])
            author_col.append(person2index[author['_id']])
        try:
            # sometimes no references or no information about it
            temp = paper['references']
        except:
            # print(paper)
            continue
        for ref in temp:
            try:
                paper_ref_col.append(paper2index[ref])
            except:
                continue
            paper_ref_row.append(paper2index[paper['_id']])

    pro_id = list(project2index.keys())
    train_projects_text_emb = {}
    assert len(pro_id) == len(pro_emb_data), 'error'
    for i in range(len(pro_emb_data)):
        if index2project[i] in train_project:
            # print(pro_emb_data[i])
            train_projects_text_emb[i] = pro_emb_data[i].numpy()

    train_dataset = []
    for index in range(len(train_data)):
        project_id = project2index[train_data[index][2]]
        pos_person = person2index[train_data[index][1]]
        project_text_emb = pro_emb_data[project_id].numpy()
        neg_person = []
        for i in range(len(train_data[index][4])):
            neg_person.append(person2index[train_data[index][4][i]])
        train_dataset.append((project_id, project_text_emb, pos_person, neg_person))

    valid_dataset = []
    for index in range(len(valid_data)):
        project_id = project2index[valid_data[index][2]]
        pos_person = person2index[valid_data[index][1]]
        project_text_emb = pro_emb_data[project_id].numpy()
        neg_person = []
        for i in range(len(valid_data[index][4])):
            neg_person.append(person2index[valid_data[index][4][i]])
        valid_dataset.append((project_id, project_text_emb, pos_person, neg_person))

    test_dataset = []
    for index in range(len(test_data)):
        project_id = project2index[test_data[index][2]]
        pos_person = person2index[test_data[index][1]]
        project_text_emb = pro_emb_data[project_id].numpy()
        neg_person = []
        for i in range(len(test_data[index][4])):
            neg_person.append(person2index[test_data[index][4][i]])
        test_dataset.append((project_id, project_text_emb, pos_person, neg_person))

    print('start')
    # use_cuda = torch.cuda.is_available() and args.cuda
    # device = torch.device('cuda' if use_cuda else 'cpu')
    # print(device)
    device = torch.device('cpu')

    G = dgl.heterograph({
        ('project', 'investigated-by', 'person'): (project_main_row, person_main_col),
        ('person', 'investigate', 'project'): (person_main_col, project_main_row),
        # ('project', 'co-investigated-by', 'person'): (project_co_row, person_co_col),
        # ('person', 'co-investigate', 'project'): (person_co_col, project_co_row),
        ('paper', 'cite', 'paper'): (paper_ref_row, paper_ref_col),
        ('paper', 'cited-by', 'paper'): (paper_ref_col, paper_ref_row),
        ('paper', 'writed-by', 'person'): (paper_auther_row, author_col),
        ('person', 'write', 'paper'): (author_col, paper_auther_row),
    })

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
    node_emb = {}
    for ntype in G.ntypes:
        G.nodes[ntype].data['id'] = torch.arange(0, G.number_of_nodes(ntype))

    emb = nn.Embedding(G.number_of_nodes('person'), args.n_dim)
    node_emb['person'] = emb.to(device)
    emb = nn.Embedding.from_pretrained(pro_emb_data,freeze=False)
    node_emb['project'] = emb.to(device)
    emb = nn.Embedding.from_pretrained(paper_emb_data,freeze=False)
    node_emb['paper'] = emb.to(device)
    # G = G.to(device)
    train_data_loader = DataLoader(
        dataset=NSFDataset(G, train_dataset, train_projects_text_emb, args, 'train'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=graph_collate,
        pin_memory=True
    )
    valid_data_loader = DataLoader(
        dataset=NSFDataset(G, valid_dataset, train_projects_text_emb, args, 'valid'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=graph_collate,
        pin_memory=True
    )
    test_data_loader = DataLoader(
        dataset=NSFDataset(G, test_dataset, train_projects_text_emb, args, 'test'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=graph_collate,
        pin_memory=True
    )

    print(len(train_data_loader))
    print(len(valid_data_loader))
    print(len(test_data_loader))

    hgt_model = HGT(node_emb,
                node_dict, edge_dict,
                n_inp=args.n_dim,
                n_hid=args.n_hid,
                n_out=args.n_dim,
                n_layers=args.n_neigh_layer,
                n_heads=args.n_head,
                use_norm=True).to(device)

    del papers, persons, project, train_data, valid_data, test_data, pro_emb_data, paper_emb_data, projects
    torch.cuda.empty_cache()
    print('Training HGT with #param: %d' % (get_n_params(hgt_model)))
    train(hgt_model, train_data_loader, valid_data_loader, test_data_loader, device, args)




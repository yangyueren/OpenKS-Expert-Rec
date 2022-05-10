import torch
import numpy as np
import dgl
from torch.utils.data import Dataset
from collections import defaultdict

class NSFDataset(Dataset):
    def __init__(self, G, data, projects_text_emb, args, label):
        super(NSFDataset, self).__init__()
        self.data = data
        self.args = args
        self.G = G
        self.proj_text_emb = np.array(list(projects_text_emb.values()))
        self.proj_text_id = np.array(list(projects_text_emb.keys()))
        self.label = label

    def Calculate_Similarity(self, project_text_emb):
        score = np.sum(project_text_emb * self.proj_text_emb, axis=1)
        indices = np.argpartition(score, -self.args.max_project)
        #top_n
        # similar_id = np.array(list(self.proj_text_id[indices[-self.args.max_project:]]))
        emb_weight = score[indices[-self.args.max_project:]]
        similar_id = self.proj_text_id[indices[-self.args.max_project:]]
        return similar_id, emb_weight

    def get_subgraph_from_heterograph(self, similar_id, person_list):

        subgraph_in = dgl.sampling.sample_neighbors(self.G, nodes={'project':similar_id, 'person':person_list}, fanout=self.args.n_max_neigh[0],
                                                        edge_dir='in')
        nodes_subgraph = defaultdict(set)
        nodes_subgraph['project'].update(similar_id)
        nodes_subgraph['person'].update(person_list)

        for layer in range(1, self.args.n_neigh_layer):
            subgraph = subgraph_in
            new_adj_nodes = defaultdict(set)
            for node_type_1, edge_type, node_type_2 in subgraph.canonical_etypes:
                nodes_id_1, nodes_id_2 = subgraph.all_edges(etype=edge_type)
                new_adj_nodes[node_type_1].update(set(nodes_id_1.numpy()).difference(nodes_subgraph[node_type_1]))
                new_adj_nodes[node_type_2].update(set(nodes_id_2.numpy()).difference(nodes_subgraph[node_type_2]))
                nodes_subgraph[node_type_1].update(new_adj_nodes[node_type_1])
                nodes_subgraph[node_type_2].update(new_adj_nodes[node_type_2])

            new_adj_nodes = {key: list(value) for key, value in new_adj_nodes.items()}

            subgraph_in = dgl.sampling.sample_neighbors(self.G, nodes=dict(new_adj_nodes),
                                                        fanout=self.args.n_max_neigh[layer],
                                                        edge_dir='in')
        nodes_sampled = {}
        for node_type in nodes_subgraph.keys():
            nodes_sampled[node_type] = torch.LongTensor(list(nodes_subgraph[node_type]))
        sub_g = self.G.subgraph(nodes_sampled)
        # print(sub_g.ndata[dgl.NID])
        # print(sub_g.nodes['project'].data['id'])

        return sub_g

    def __getitem__(self, index):
        project_id = self.data[index][0]
        project_text_emb = self.data[index][1]
        pos_person = self.data[index][2]
        neg_person_list = self.data[index][3]
        if self.label == 'train':
            similar_id = [project_id]
            emb_weight = 1
        else:
            similar_id, emb_weight = self.Calculate_Similarity(project_text_emb)

        person_list = torch.LongTensor([pos_person] + neg_person_list)
        similar_id = torch.LongTensor(similar_id)
        sub_g = self.get_subgraph_from_heterograph(similar_id, person_list)

        return project_id, sub_g, similar_id, emb_weight, pos_person, neg_person_list

    def __len__(self):
        # if self.label == 'train':
        #     return 20000
        # else:
        #     return 800
        return len(self.data)
        # return 1000

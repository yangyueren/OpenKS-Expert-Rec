import torch
import numpy as np
import dgl
from torch.utils.data import Dataset

class NSFDataset(Dataset):
    def __init__(self, G, data, projects_text_emb, args):
        super(NSFDataset, self).__init__()
        self.data = data
        self.args = args
        self.G = G
        self.proj_text_emb = np.array(list(projects_text_emb.values()))
        self.proj_text_id = np.array(list(projects_text_emb.keys()))

    def Calculate_Similarity(self, project_text_emb):
        score = np.sum(project_text_emb * self.proj_text_emb, axis=1)
        indices = np.argpartition(score, -self.args.max_project)
        #top_n
        # similar_id = np.array(list(self.proj_text_id[indices[-self.args.max_project:]]))
        similar_id = self.proj_text_id[indices[-self.args.max_project:]]
        return similar_id

    def get_subgraph_from_heterograph(self, similar_id, person_list):

        subgraph_in = dgl.sampling.sample_neighbors(self.G, nodes={'project':similar_id, 'person':person_list}, fanout=self.args.n_max_neigh,
                                                        edge_dir='in')
        print(subgraph_in)
        return subgraph_in

    def __getitem__(self, index):
        project_id = self.data[index][0]
        project_text_emb = self.data[index][1]
        pos_person = self.data[index][2]
        neg_person_list = self.data[index][3]

        similar_id = self.Calculate_Similarity(project_text_emb)

        person_list = torch.LongTensor([pos_person] + neg_person_list)
        similar_id = torch.LongTensor(similar_id)
        sub_g = self.get_subgraph_from_heterograph(similar_id, person_list)

        return project_id, sub_g, similar_id, pos_person, neg_person_list

    def __len__(self):
        return len(self.data)
        # return 100

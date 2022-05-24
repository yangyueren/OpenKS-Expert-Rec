

import ujson as json
import yaml
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from util import load_from_disk, Dict2Obj
from mylogger import logger

from graph_util import sampler_of_graph
import random
random.seed(37)

class NSFDataset(Dataset):
    def __init__(self, config, data_path, is_train=True) -> None:
        super().__init__()
        self.config = config
        self.is_train = is_train
        self.num_person = -1
        
        self.triple = load_from_disk(data_path)
        if config.debug == 1:
            logger.info('debug using 100 pieces of data!!!')
            self.triple = load_from_disk(data_path)[:100]
        
        self.data = None

    def create_graph(self, config):
        logger.info('creating graph...')
        project_emb = load_from_disk(config.project_emb)
        paper_emb = load_from_disk(config.paper_emb)
        data = HeteroData()
        
        entities_person = load_from_disk(config.entities_person)
        self.num_person = len(entities_person)
        data['person'].x = torch.randn((len(entities_person), project_emb.shape[1])) # [num_persons, num_features_person]
        data['person'].n_id = torch.arange(data['person'].num_nodes)

        data['project'].x = project_emb # [num_institutions, num_features_institution]
        data['project'].n_id = torch.arange(data['project'].num_nodes)

        logger.info('remove project nodes.')
        data['paper'].x = paper_emb  # [num_papers, num_features_paper]
        data['paper'].n_id = torch.arange(data['paper'].num_nodes)
        
        
        def get_edge_index(path):
            edge = load_from_disk(path)
            logger.info(f'{path} has edges of {len(edge)}')
            s2t = torch.tensor(edge).transpose(1,0).contiguous()
            t2s = s2t[[1,0], :].contiguous()
            return s2t, t2s

        s2t, t2s = get_edge_index(config.rel_reference)
        data['paper', 'cited_by', 'paper'].edge_index = s2t # [2, num_edges_cites]

        s2t, t2s = get_edge_index(config.rel_published_by)
        data['paper', 'published_by', 'person'].edge_index = s2t # [2, num_edges_writes]

        s2t, t2s = get_edge_index(config.rel_cooperate)
        data['person', 'cooperate', 'person'].edge_index = s2t # [2, num_edges_affiliated]
        
        s2t, t2s = get_edge_index(config.rel_co_author)
        data['person', 'co_author', 'person'].edge_index = s2t # [2, num_edges_affiliated]
        
        # s2t, t2s = get_edge_index(config.rel_pricipal_investigator_by)
        # data['project', 'investigate_by', 'person'].edge_index = s2t # [2, num_edges_topic]

        # s2t, t2s = get_edge_index(config.rel_common_investigator_by)
        # data['project', 'common_investigate_by', 'person'].edge_index = s2t # [2, num_edges_topic]

        
        # transform = T.Compose([T.ToUndirected(merge=True), T.AddSelfLoops()])
        # transform = T.Compose([ T.AddSelfLoops()])
        # data = transform(data)
        return data

    def get_graph(self):
        if self.data is None:
            self.data = self.create_graph(self.config)
        return self.data

    def get_metadata(self):
        if self.data is None:
            self.data = self.create_graph(self.config)
        return self.data.metadata()
    
    def __getitem__(self, index):
        _, project_id, person_id, year, neg_persons = self.triple[index]
        if self.is_train:
            neg = set()
            while len(neg) < self.config.neg_persons_num_train:
                pid = random.randint(0, self.num_person-1)
                if pid != person_id:
                    neg.add(pid)
            persons = [person_id] + list(neg)
        else:
            persons = [person_id] + neg_persons
        return persons, project_id, 1

    def __len__(self):
        return len(self.triple)

if __name__ == '__main__':
    with open('./config.yml', 'r') as f:
        config = yaml.safe_load(f)
        config = Dict2Obj(config).config
    dataset = NSFDataset(config, config.train_data, is_train=True)
    data = dataset.get_graph()
    print(data)
    print(dataset.get_metadata())
    
    



# num_papers = 100
# num_persons = 100
# num_projects = 100
# feat_dim = 768


# class NSFDataset(Dataset):
#     def __init__(self, config, data_path, is_train=True) -> None:
#         super().__init__()
#         # self.config = config
#         # person = load_from_disk(config.entities_person)
#         # project = load_from_disk(config.entities_project)
#         # self.triple = load_from_disk(data_path)
#         # if config.debug == 1:
#         #     logger.info('debug using 100 pieces of data!!!')
#         #     self.triple = load_from_disk(data_path)[:100]

#         # self.person2id = {}
#         # for p in sorted(person):
#         #     self.person2id[p] = len(self.person2id)

#         # self.personid_set = list()
#         # for k,v in self.person2id.items():
#         #     self.personid_set.append(v)
        
#         # self.project2id = {}
#         # self.id2project = {}
#         # for p in project:
#         #     p = json.loads(p)
#         #     self.project2id[p['AwardID']] = len(project)
#         #     self.id2project[self.project2id[p['AwardID']]] = p
        
#         # self.data = []
#         # self.pos_data = []
#         # self.neg_data = []

#         # if self.is_train:
#         #     for trip in self.triple:
#         #         _, person, project_award, year, neg_persons = trip
#         #         assert person in self.person2id
#         #         self.pos_data.append((self.person2id[person], project_award, 1))

#         self.data = self.create_graph()
#         self.triples = []
#         for i in range(500):
#             person1 = random.randint(0,num_persons-1)
#             person2 = random.randint(0,num_persons-1)
#             project = random.randint(0,num_projects-1)
#             self.triples.append(((person1, person2), project, 1))

#     def create_graph(self):
#         data = HeteroData()
#         data['paper'].x = torch.randn((num_papers, feat_dim)) # [num_papers, num_features_paper]
#         data['paper'].n_id = torch.arange(data['paper'].num_nodes)
        
#         data['person'].x = torch.randn((num_persons, feat_dim)) # [num_persons, num_features_person]
#         data['person'].n_id = torch.arange(data['person'].num_nodes)

#         data['project'].x = torch.randn((num_projects, feat_dim)) # [num_institutions, num_features_institution]
#         data['project'].n_id = torch.arange(data['project'].num_nodes)
        
#         def get_edge_index(a, b):
#             src = torch.randint(0, a, (2000,))
#             dst = torch.randint(0, b, (2000,))
#             s2d = torch.vstack([src, dst])
#             d2s = torch.vstack([dst, src])
#             return s2d, d2s

#         s2d, d2s = get_edge_index(num_papers, num_papers)
#         data['paper', 'cite', 'paper'].edge_index = s2d # [2, num_edges_cites]
#         data['paper', 'cite_reverse', 'paper'].edge_index = d2s # [2, num_edges_cites]
        
#         s2d, d2s = get_edge_index(num_persons, num_papers)
#         data['person', 'publish', 'paper'].edge_index = s2d # [2, num_edges_writes]
#         data['paper', 'publish_reverse', 'person'].edge_index = d2s # [2, num_edges_writes]
        
#         s2d, d2s = get_edge_index(num_persons, num_persons)
#         data['person', 'cooperate', 'person'].edge_index = torch.cat([s2d, d2s], dim=1) # [2, num_edges_affiliated]
        
#         s2d, d2s = get_edge_index(num_persons, num_projects)
#         data['person', 'investigate', 'project'].edge_index = s2d # [2, num_edges_topic]
#         data['project', 'investigate_reverse', 'person'].edge_index = d2s
#         return data


#     def get_graph(self):
#         return self.data

#     def get_metadata(self):
#         return self.data.metadata()
    

#     def __getitem__(self, index):
#         # project_id, person_id, label
#         person, project, label = self.triples[index]
#         return person, project, label

#     def __len__(self):
#         return len(self.triples)

# if __name__ == '__main__':
#     dataset = NSFDataset(None, None)
#     data = dataset.create_graph()
#     print(data)
#     nodeids = [1,2,3,4]
#     name = 'project'
#     print(data['investigate_reverse'])
#     sample_data = sampler_of_graph(nodeids, name, data)
#     import pdb;pdb.set_trace()
#     print(sample_data)
    
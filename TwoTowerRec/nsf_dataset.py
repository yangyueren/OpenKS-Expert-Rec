
import ujson as json
import torch
from torch.utils.data import Dataset

from util import load_from_disk

class NSFDataset(Dataset):
    def __init__(self, config, data_path) -> None:
        super().__init__()
        self.config = config
        person = load_from_disk(config.entities_person)
        project = load_from_disk(config.entities_project)
        self.data = load_from_disk(data_path)

        self.person2id = {}
        for p in person:
            self.person2id[p] = len(self.person2id)
        self.project2id = {}
        self.id2project = {}
        for p in project:
            p = json.loads(p)
            self.project2id[p['AwardID']] = len(project)
            self.id2project[self.project2id[p['AwardID']]] = p
        
        for triple in self.data:
            _, person, project_award, year, neg_persons = triple
            assert person in self.person2id
        

    def __getitem__(self, index):
        triple = self.data[index]
        _, person, project_award, year, neg_persons = triple
        assert person not in neg_persons, 'neg persons error'

        persons = [self.person2id[person]]
        for p in neg_persons:
            persons.append(self.person2id[p])
        persons = sorted(persons)
        idx = persons.index(self.person2id[person])

        pid = self.project2id[project_award]
        abstract_of_project = self.id2project[pid]['AbstractNarration']


        return persons, idx, abstract_of_project

    def __len__(self):
        return len(self.data)
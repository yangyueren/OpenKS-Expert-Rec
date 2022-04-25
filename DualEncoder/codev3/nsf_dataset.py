

import ujson as json
import torch
from torch.utils.data import Dataset

from util import load_from_disk
from mylogger import logger


import random
random.seed(37)

class NSFDataset(Dataset):
    def __init__(self, config, data_path, is_train=True) -> None:
        super().__init__()
        self.config = config
        self.is_train = is_train
        person = load_from_disk(config.entities_person)
        project = load_from_disk(config.entities_project)
        self.triple = load_from_disk(data_path)
        if config.debug == 1:
            logger.info('debug using 100 pieces of data!!!')
            self.triple = load_from_disk(data_path)[:100]

        self.person2id = {}
        for p in sorted(person):
            self.person2id[p] = len(self.person2id)

        self.personid_set = list()
        for k,v in self.person2id.items():
            self.personid_set.append(v)
        
        self.project2id = {}
        self.id2project = {}
        for p in project:
            p = json.loads(p)
            self.project2id[p['AwardID']] = len(project)
            self.id2project[self.project2id[p['AwardID']]] = p
        
        self.data = []
        self.pos_data = []
        self.neg_data = []

        if self.is_train:
            for trip in self.triple:
                _, person, project_award, year, neg_persons = trip
                assert person in self.person2id
                self.pos_data.append((self.person2id[person], project_award, 1))

        else:
            for trip in self.triple:
                _, person, project_award, year, neg_persons = trip
                assert person in self.person2id
                person_ids = [self.person2id[person]]
                for neg_p in neg_persons:
                    person_ids.append(self.person2id[neg_p])
                self.data.append(( person_ids , project_award, 0))


    def __getitem__(self, index):
        if self.is_train:
            triple = self.pos_data[index]
            personid, project_award, label = triple
            pid = self.project2id[project_award]
            abstract_of_project = self.id2project[pid]['AbstractNarration']

            neg_personid = set()
            while len(neg_personid) < self.config.neg_persons_num_train:
                p = random.choice(self.personid_set)
                if p != personid:
                    neg_personid.add(p)
            return [personid] + list(neg_personid), abstract_of_project, label        # 人，项目，1或0
        else:
            triple = self.data[index]
            personid, project_award, label = triple
            pid = self.project2id[project_award]
            abstract_of_project = self.id2project[pid]['AbstractNarration']
            return personid, abstract_of_project, label

    def __len__(self):
        if self.is_train:
            return len(self.pos_data)
        else:
            return len(self.data)
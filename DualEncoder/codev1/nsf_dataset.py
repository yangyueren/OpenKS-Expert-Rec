
import ujson as json
import torch
from torch.utils.data import Dataset

from util import load_from_disk

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
        # self.triple = load_from_disk(data_path)[:100]

        # if is_train:
        #     self.triple = load_from_disk(data_path)[:100]

        self.person2id = {}
        for p in person:
            self.person2id[p] = len(self.person2id)
        self.project2id = {}
        self.id2project = {}
        for p in project:
            p = json.loads(p)
            self.project2id[p['AwardID']] = len(project)
            self.id2project[self.project2id[p['AwardID']]] = p
        
        self.data = []
        if self.is_train:
            for trip in self.triple:
                _, person, project_award, year, neg_persons = trip
                assert person in self.person2id
                self.data.append((self.person2id[person], project_award, 1))

                neg_person_ids = []
                for neg_p in neg_persons:
                    neg_person_ids.append(self.person2id[neg_p])
                self.data.append((neg_person_ids, project_award, 0))
        else:
            for trip in self.triple:
                _, person, project_award, year, neg_persons = trip
                assert person in self.person2id
                person_ids = [self.person2id[person]]
                for neg_p in neg_persons:
                    person_ids.append(self.person2id[neg_p])
                self.data.append(( person_ids , project_award, 0))


    def __getitem__(self, index):
        triple = self.data[index]
        personid, project_award, label = triple
        pid = self.project2id[project_award]
        abstract_of_project = self.id2project[pid]['AbstractNarration']
        if self.is_train and label == 0:
            personid = random.choice(personid)

        return personid, abstract_of_project, label # 人，项目，1或0

    def __len__(self):
        return len(self.data)
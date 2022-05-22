from tqdm import tqdm
import pickle
import json

root_path = '/home/disk1/ls/project/zheda_nsf/data/'
with open(root_path + 'entities_paper.pkl', 'rb') as f:
    papers = pickle.load(f)
with open(root_path + 'entities_person.pkl', 'rb') as f:
    persons = pickle.load(f)
with open(root_path + 'entities_project.pkl', 'rb') as f:
    projects = pickle.load(f)
with open(root_path + 'train_rel_is_principal_investigator_of.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(root_path + 'val_rel_is_principal_investigator_of.pkl', 'rb') as f:
    valid_data = pickle.load(f)
with open(root_path + 'test_rel_is_principal_investigator_of.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open(root_path + 'project_emb_all_mpnet_base_v2.pkl', 'rb') as f:
    emb_data = pickle.load(f)

train_project = []
for train_one in train_data:
    train_project.append(train_one[2])

#
# print(len(projects))
# for i in range(10,30):
#     print(projects[i])
# print(papers[0])

# project2index = {}
# index2project = {}
# for index in range(len(projects)):
#     projects[index] = json.loads(projects[index])
#     project2index[projects[index]['AwardID']] = index
#     index2project[index] = projects[index]['AwardID']

project2index = {}
index2project = {}
for index in range(len(projects)):
    projects[index] = json.loads(projects[index])
    project2index[projects[index]['AwardID']] = index
    index2project[index] = projects[index]['AwardID']

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
        # else:
        #     print('find new role!')
        #     print(project['Investigator'])
        #     print(role['RoleCode'])
        #     break


paper_auther_row = []
author_col = []
paper_ref_row = []
paper_ref_col = []
for paper in papers:
    for author in paper['authors']:
        paper_auther_row.append(paper2index[paper['_id']])
        author_col.append(person2index[author['_id']])
    try:
        #sometimes no references or no information about it
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

projects_text_emb = {}
train_projects_text_emb ={}
for i in range(len(emb_data)):
    if emb_data[i][2] is not None:
        projects_text_emb[project2index[emb_data[i][0]]] = emb_data[i][2]
        if emb_data[i][0] in train_project:
            train_projects_text_emb[project2index[emb_data[i][0]]] = emb_data[i][2]

train_dataset = []
for index in range(len(train_data)):
    project_id = project2index[train_data[index][2]]
    pos_person = person2index[train_data[index][1]]
    project_text_emb = projects_text_emb[project_id]
    neg_person = []
    for i in range(len(train_data[index][4])):
        neg_person.append(person2index[train_data[index][4][i]])
    train_dataset.append((project_id, project_text_emb, pos_person, neg_person))

valid_dataset = []
for index in range(len(valid_data)):
    project_id = project2index[valid_data[index][2]]
    pos_person = person2index[valid_data[index][1]]
    project_text_emb = projects_text_emb[project_id]
    neg_person = []
    for i in range(len(valid_data[index][4])):
        neg_person.append(person2index[valid_data[index][4][i]])
    valid_dataset.append((project_id, project_text_emb, pos_person, neg_person))

test_dataset = []
for index in range(len(test_data)):
    project_id = project2index[test_data[index][2]]
    pos_person = person2index[test_data[index][1]]
    project_text_emb = projects_text_emb[project_id]
    neg_person = []
    for i in range(len(test_data[index][4])):
        neg_person.append(person2index[test_data[index][4][i]])
    test_dataset.append((project_id, project_text_emb, pos_person, neg_person))

with open(root_path + '/index.pkl', 'wb') as f:
    pickle.dump(project2index, f)
    pickle.dump(index2project, f)
    pickle.dump(paper2index, f)
    pickle.dump(index2paper, f)
    pickle.dump(person2index, f)
    pickle.dump(index2person, f)

with open(root_path + '/dgl_data.pkl', 'wb') as f:
    pickle.dump(project_main_row, f)
    pickle.dump(person_main_col, f)
    pickle.dump(project_co_row, f)
    pickle.dump(person_co_col, f)
    pickle.dump(paper_ref_row, f)
    pickle.dump(paper_ref_col, f)
    pickle.dump(paper_auther_row, f)
    pickle.dump(author_col, f)

with open(root_path + '/train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

with open(root_path + '/valid_dataset.pkl', 'wb') as f:
    pickle.dump(valid_dataset, f)

with open(root_path + '/test_dataset.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)

with open(root_path + '/projects_text_emb.pkl', 'wb') as f:
    pickle.dump(train_projects_text_emb, f)

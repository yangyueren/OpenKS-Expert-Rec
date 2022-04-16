# NSF + DBLP for Recommendation

## Process NSF and DBLP data

- Put `dblpv13.json` and `NSF-data/` in `raw` folder;
- Run:
```bash
python analysis_v1-1.py
python analysis_v1-2.py
python analysis_v1-3.py
python analysis_v1-4.py
```
- Check md5 of generated files:
  - https://www.notion.so/md5sum-ff8177dce9194a1a8838a43e071f9bfb


## 数据格式

### entities_paper.pkl

list

每个元素是一个dict

- _id 是paper的唯一标识
- title 是paper的标题
- authors 是一个list，里面是每个作者对id，与entities_person里面的id对应
- year 是发表年份
- abstract 是简介

样例：
```json
{'_id': '53e99784b7602d9701f3e151', 'title': 'A solution to the problem of touching and broken characters.', 'authors': [], 'venue': {'_id': '53a72a4920f7420be8bfa51b', 'name_d': 'International Conference on Document Analysis and Recognition', 'type': 0, 'raw': 'ICDAR-1'}, 'year': 1993, 'keywords': ['handwriting recognition', 'prototypes', 'image segmentation', 'computer science', 'expert systems', 'knowledge base', 'pattern recognition', 'usability', 'optical character recognition', 'shape', 'feature extraction'], 'fos': ['Intelligent character recognition', 'Pattern recognition', 'Computer science', 'Feature (computer vision)', 'Document processing', 'Handwriting recognition', 'Optical character recognition', 'Feature extraction', 'Feature (machine learning)', 'Artificial intelligence', 'Intelligent word recognition'], 'n_citation': 17, 'page_start': '602', 'page_end': '605', 'lang': 'en', 'volume': '', 'issue': '', 'issn': '', 'isbn': '', 'doi': '10.1109/ICDAR.1993.395663', 'pdf': None, 'url': ['http://dx.doi.org/10.1109/ICDAR.1993.395663'], 'abstract': '', 'references': ['53e99cf5b7602d97025ace63', '557e8a7a6fee0fe990caa63d', '53e9a96cb7602d97032c459a', '53e9b929b7602d9704515791', '557e59ebf6678c77ea222447']}
```
  
### entities_person.pkl
list

每个元素是一个str，是一个person的uid

### entities_project.pkl
list

每个元素是一个json格式的str，请用json.loads()加载为dict

- AwardID 每个project的唯一标识
- AwardTitle 项目的title
- AbstractNarration 项目的简介
- Investigator 是一个list，list里的每个元素是一个dict
  - uid 代表person
  - RoleCode 代表角色，有发起人，参与者 等角色


```json
'{"AwardID":"0000000","AwardTitle":"Regulation of Sn-Glycerol-3-Phosphate Metabolism in         Escherichia Coli by the GLPR - Encoded Repressor","AwardEffectiveDate":"07\\/01\\/1986","AbstractNarration":"","Investigator":[{"uid":"nsfuid149896","RoleCode":"Principal Investigator"}]}'
```

### rel_co_author.pkl
list

每个元素是一个四元组: ('co_author', au1, au2, year)，代表paper中作者的合作关系

```json
('co_author', 'nsfuid057285', 'nsfuid083244', 2010)
```

### rel_cooperate.pkl
list

每个元素是一个四元组: ('cooperate', au1['uid'], au2['uid'], year)，代表nsf project中人员的合作关系

```json
('cooperate', 'nsfuid019256', 'nsfuid047540', 2000)
```


### rel_is_publisher_of.pkl
list

每个元素是一个四元组: ('is_publisher_of', auid, paper['_id'], year)，代表作者和论文的发表关系

```json
('is_publisher_of', 'nsfuid030161', '53e99784b7602d9701f3f600', 2004)
```

### rel_reference.pkl
list

每个元素是一个四元组: ('reference', paper['_id'], ref, year)，代表paper与paper之间的引用关系

```json
('reference', '53e99784b7602d9701f3e151', '53e99cf5b7602d97025ace63', 1993)
```
### train_is_principal_investigator_of.pkl   test_is_principal_investigator_of.pkl   val_is_principal_investigator_of.pkl

list

每个元素是一个五元组: ('is_principal_investigator_of', pricipal_uid, AwardID, year,list(neg_persons)，代表发起人和nsf project的关系，第五个元素是一个list，代表99个负样本（人员）

```json
('is_principal_investigator_of', 'nsfuid149896', '0000000', 1986, ['nsfuid103499', 'nsfuid072250', 'nsfuid109364', 'nsfuid036193', 'nsfuid068021', 'nsfuid019003', 'nsfuid010617', 'nsfuid011254', 'nsfuid024879', 'nsfuid099860', 'nsfuid140053', 'nsfuid100258', 'nsfuid012169', 'nsfuid089256', 'nsfuid023663', 'nsfuid052699', 'nsfuid091999', 'nsfuid055189', 'nsfuid129487', 'nsfuid111935', 'nsfuid074517', 'nsfuid123844', 'nsfuid065468', 'nsfuid028244', 'nsfuid062851', 'nsfuid150507', 'nsfuid136640', 'nsfuid107918', 'nsfuid050946', 'nsfuid063883', 'nsfuid138162', 'nsfuid014519', 'nsfuid038096', 'nsfuid039940', 'nsfuid093165', 'nsfuid062580', 'nsfuid148251', 'nsfuid122410', 'nsfuid116073', 'nsfuid020715', 'nsfuid005189', 'nsfuid074624', 'nsfuid056690', 'nsfuid150653', 'nsfuid000859', 'nsfuid117343', 'nsfuid011170', 'nsfuid137861', 'nsfuid148008', 'nsfuid017370', 'nsfuid107427', 'nsfuid094998', 'nsfuid037995', 'nsfuid053865', 'nsfuid046374', 'nsfuid060237', 'nsfuid114245', 'nsfuid121349', 'nsfuid078994', 'nsfuid006998', 'nsfuid032798', 'nsfuid066220', 'nsfuid072376', 'nsfuid069510', 'nsfuid021890', 'nsfuid123077', 'nsfuid035668', 'nsfuid022336', 'nsfuid124219', 'nsfuid112635', 'nsfuid112244', 'nsfuid084688', 'nsfuid047400', 'nsfuid076600', 'nsfuid096054', 'nsfuid135275', 'nsfuid050908', 'nsfuid113417', 'nsfuid003370', 'nsfuid103648', 'nsfuid033771', 'nsfuid127417', 'nsfuid044902', 'nsfuid037020', 'nsfuid047773', 'nsfuid007560', 'nsfuid019992', 'nsfuid018176', 'nsfuid075435', 'nsfuid037623', 'nsfuid042360', 'nsfuid045174', 'nsfuid095752', 'nsfuid037569', 'nsfuid088148', 'nsfuid105089', 'nsfuid033846', 'nsfuid116669', 'nsfuid134080'])
```

- train数据集是2015年之前的project
- val和test是2015年及之后的数据，val和test是按照3:7的比例随机划分的。

### rel_is_investigator_of.pkl
暂时无用

### rel_is_principal_investigator_of.pkl
暂时无用

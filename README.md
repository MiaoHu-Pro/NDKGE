PyTorch implementation of paper [Knowledge Graph Representation Learning via Generated Descriptions](https://pure.qub.ac.uk/en/publications/knowledge-graph-representation-learning-via-generated-description)([NLDB2023]) accepted by NLDB 2023.
# Knowledge Graph Representation Learning via Generated Descriptions
Knowledge graph representation learning (KGRL) aims to project the entities and relations into a continuous low-dimensional knowledge graph space to be used for knowledge graph completion and detecting new triples. Using textual descriptions for entity representation learning has been a key topic. However, the current work has two major constraints: (1) some entities do not have any associated descriptions; (2) the associated descriptions are usually phrases, and they do not contain enough information. This paper presents a novel KGRL method for learning effective embeddings by generating meaningful descriptive sentences from entities’ connections. The experiments using four public datasets and a new proposed dataset show that the New Description-Embodied Knowledge Graph Embedding (NDKGE for short) approach introduced in this paper outperforms most of the existing work in the task of link prediction.

### Install python virtual environment

```bash
# install virtual environment module
python3 -m pip install --user virtualenv

# create virtual environment
python3 -m venv env_name
source env_name/bin/activate
# install python packages

pip install torch
pip install openke

```

### Run the code


```
python3 train_transe_ndkge.py --dataset_path ./benchmarks/WN18RR/  --dataset WN18RR --setting name --id_dim 20 --word_dim 50  --nbatches 100  --margin 2 --num_epochs 20000 --learning_rate 0.5 --model_name wn18rr_name_epochs-20000_margin-2_lr-1_id_dim-20_word_dim-50
```

`--dataset_path`: the path of dataset.
 
 `--dataset`: dataset name.
 
 `--setting`: using different setting, such as 'name','mention',and 'description'

  `--id_dim`: the dimention of id.
  
  `--word_dim`: the dimention of token.
  
  `--nbatches`: batch size for training.
  
  `--learning_rate`: learning rate for training.
  
  `--num_epochs`: number of epoches for training.
  
  `--model_name`: model name for output.
  
  `--margin`: margin for loss function.
   

### Data
Download data from [here](https://qubstudentcloud-my.sharepoint.com/:f:/g/personal/40305887_ads_qub_ac_uk/En2kzLSxom1IuOdqQPMi6voBg9SdIATfIhmOFSlTEM-Wug?e=6q95GY)
### Contact:
This paper has been accepted by the 28th International Conference on Natural Language & Information Systems (NLDB 2023). The published version can be viewed by this [link](https://pure.qub.ac.uk/en/publications/knowledge-graph-representation-learning-via-generated-description). If you use any code from our repo in your paper, pls cite:
```buildoutcfg
@InProceedings{10.1007/978-3-031-35320-8_26,
author="Hu, Miao
and Lin, Zhiwei
and Marshall, Adele",
editor="M{\'e}tais, Elisabeth
and Meziane, Farid
and Sugumaran, Vijayan
and Manning, Warren
and Reiff-Marganiec, Stephan",
title="Knowledge Graph Representation Learning via Generated Descriptions",
booktitle="Natural Language Processing and Information Systems",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="365--378"
}
```

Feel free to contact MiaoHu ([mhu05@qub.ac.uk](mhu05@qub.ac.uk)),  if you have any further questions.


import openke
import numpy as np
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from utilities import read_new_init_embs ,read_transe_out_embs
from create_description.ERDse import ERDes

from argparse import ArgumentParser


parser = ArgumentParser("NDKGE")
parser.add_argument("--dataset_path", default="./benchmarks/FB15K237/", help="Name of the dataset.")
parser.add_argument("--dataset", default="FB15K237", help="Name of the dataset.")

parser.add_argument("--setting", default="des", help="Name of the setting.")

parser.add_argument("--learning_rate", default=1, type=float, help="Learning rate")
parser.add_argument("--nbatches", default=100, type=int, help="Number of batches")
parser.add_argument("--num_epochs", default=10000, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='FB15K237', help="")
parser.add_argument('--neg_num', default=2, type=int, help='')
parser.add_argument('--hidden_size', type=int, default=50, help='')
parser.add_argument('--num_of_filters', type=int, default=64, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='')
parser.add_argument('--save_steps', type=int, default=1000, help='')
parser.add_argument('--valid_steps', type=int, default=50, help='')
parser.add_argument("--lmbda", default=0.2, type=float, help="")
parser.add_argument("--lmbda2", default=0.01, type=float, help="")
parser.add_argument("--checkpoint_path", default=None, type=str)
parser.add_argument("--test_file", default="", type=str)
parser.add_argument("--optim", default='adagrad', help="")
parser.add_argument('--id_dim', default=200, type=int, help='')
parser.add_argument('--word_dim', default=300, type=int, help='')
parser.add_argument('--margin', default=5, type=float, help='')


args = parser.parse_args()

print(args)

"""
pythons train_transe_ndkge.py --dataset_path ./benchmarks/FB15K237/ --id_dim 200 --word_dim 300  --nbatches 100  --margin 5 --num_epochs 10000 --learning_rate 1 

--model_name fb15k237_epochs-10000_margin-5_lr-1_id_dim-200_word_dim-300
"""

out_transE_entity_emb = args.dataset_path + 'out_transE_entity_embedding_dim' + str(args.id_dim) + '.txt'
out_transE_relation_emb = args.dataset_path + 'out_transE_relation_embedding_dim' + str(args.id_dim) + '.txt'

pre_train_entity_id, pre_train_rel_id = read_transe_out_embs(out_transE_entity_emb ,out_transE_relation_emb)
print("out_transE_entity_emb " ,pre_train_entity_id)
print("out_transE_relation_emb " ,pre_train_rel_id)


# entity_id = './benchmarks/FB15K/randomly_entity_id50.txt'
# relation_id = './benchmarks/FB15K/randomly_relation_id50.txt'
# out_entity_embs, out_rel_embs = read_new_init_embs(entity_id,relation_id)
# print("out_transE_entity_emb ",entity_id)
# print("out_transE_relation_emb ",relation_id)


# new_entity_embs_path = './benchmarks/FB15K/new_init_entity_embedding_mention_description_id0_des300.txt'
# new_rel_embs_path = './benchmarks/FB15K/new_init_relation_embedding_mention_description_id0_des300.txt'
# print("entity_embs ",new_entity_embs_path)
# print("relation_embs ",new_rel_embs_path)

# entity_embs, rel_embs = read_new_init_embs(new_entity_embs_path ,new_rel_embs_path)

file_path = args.dataset_path
Paras = {
	'num_neighbours': args.neg_num,
	'num_step': 1,
	'word_dim': args.word_dim,
	'file_path': file_path,
	'all_triples_path': file_path + 'train.tsv',
	'entity2Obj_path': file_path + 'ID_Name_Mention.txt',
	'entity2id_path': file_path + 'entity2id.txt',
	'relation2id_path': file_path + 'relation2id.txt',
	'word_bag_path': file_path + 'word_bag.txt',
	'entity_description_des_path': file_path + 'entity2new_description_des_all_nums_1step.txt',
	'entity_mention_des_path': file_path + 'entity2new_mention_des_all_nums_1step.txt',
}

en_rel_des = ERDes(_Paras=Paras)
en_rel_des.pre_process()

random_ent_embeddings = None
random_rel_embeddings = None

entity_embs = None
rel_embs = None


if args.setting == "name":
	print("setting is :", args.setting)

	ent_init_embedding, rel_init_embedding = en_rel_des.get_name_embedding()
	print("ent_init_embedding.shape ", ent_init_embedding.shape)
	print("rel_init_embedding.shape " ,rel_init_embedding.shape)

	new_entity_embs_path = file_path + 'entity_name_init_embedding_'+ str(args.word_dim)+'.txt'

	new_rel_embs_path = file_path + 'relation_name_init_embedding_' + str(args.word_dim)+'.txt'

	entity_embs, rel_embs = read_new_init_embs(new_entity_embs_path,new_rel_embs_path)

	print("entity_embs.shape ", entity_embs.shape)
	print("rel_embs.shape ", rel_embs.shape)

elif args.setting == "mention":
	print("setting is :", args.setting)

	ent_init_embedding, rel_init_embedding = en_rel_des.get_mention_embeddings()
	print("ent_init_embedding.shape ", ent_init_embedding.shape)
	print("rel_init_embedding.shape " ,rel_init_embedding.shape)

	new_entity_embs_path = file_path + 'entity_mention_init_embedding_'+ str(args.word_dim)+'.txt'

	new_rel_embs_path = file_path + 'relation_mention_init_embedding_' + str(args.word_dim)+'.txt'

	entity_embs, rel_embs = read_new_init_embs(new_entity_embs_path,new_rel_embs_path)

	print("entity_embs.shape ", entity_embs.shape)
	print("rel_embs.shape ", rel_embs.shape)

elif args.setting == "des":
	print("setting is :", args.setting)

	ent_init_embedding, rel_init_embedding = en_rel_des.get_description_embeddings()
	# print("ent_init_embedding.shape ", ent_init_embedding.shape)
	# print("rel_init_embedding.shape ",rel_init_embedding.shape)

	new_entity_embs_path = file_path + 'entity_description_init_embedding_'+ str(args.word_dim)+'.txt'

	new_rel_embs_path = file_path + 'relation_description_init_embedding_' + str(args.word_dim)+'.txt'

	entity_embs, rel_embs = read_new_init_embs(new_entity_embs_path,new_rel_embs_path)

	print("entity_embs.shape ", entity_embs.shape)
	print("rel_embs.shape ", rel_embs.shape)

	print("setting is des ... \n")

import time
time.sleep(1)


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = args.dataset_path,
	nbatches = args.nbatches,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# name + mention + new_des
train_dataloader.set_init_embeddings(entity_embs, rel_embs)

# pre-trained entity id and relation id
train_dataloader.set_trains_out_embeddings(pre_train_entity_id, pre_train_rel_id)

# dataloader for test
test_dataloader = TestDataLoader(args.dataset_path, "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	init_en_embed = train_dataloader.get_entity_embedding(),
	init_rel_embed = train_dataloader.get_rel_embedding(),
	per_out_ent_embed= train_dataloader.get_transe_out_entity_embedding(),
	per_out_rel_embed = train_dataloader.get_transe_out_rel_embedding(),
	id_dim = args.id_dim,
	dim =args.word_dim,
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = args.margin),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = args.num_epochs, alpha =args.learning_rate , use_gpu=True)

time.sleep(1)


print("trainer_run .... " + '\n')

trainer.run()
transe.save_checkpoint('./checkpoint/' + args.model_name + '.ckpt')

# test the model
print("tester_run .... " + '\n')
# print("test 1-1: \n")
transe.load_checkpoint('./checkpoint/' + args.model_name + '.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False) 

acc, thread = tester.run_triple_classification()

print("Triples Classification: ")
print("Accuracy: ", acc)
print("Thread: ", thread)


"""
pythons train_transe_ndkge_hadoop.py --dataset_path ./benchmarks/hadoop_data/  --dataset hadoop_data  --id_dim 100 --word_dim 300  --nbatches 100  --margin 5 --num_epochs 10000 --learning_rate 1 

--model_name hadoop_epochs-10000_margin-5_lr-1_id_dim-200_word_dim-300  --setting "men" 

"""

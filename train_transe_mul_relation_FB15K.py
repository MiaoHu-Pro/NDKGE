

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
from utilities import read_new_init_embs,read_transe_out_embs


out_transE_entity_emb = './benchmarks/FB15K/out_transE_entity_embedding100.txt'
out_transE_relation_emb = './benchmarks/FB15K/out_transE_relation_embedding100.txt'

out_entity_embs, out_rel_embs = read_transe_out_embs(out_transE_entity_emb,out_transE_relation_emb)
print("out_transE_entity_emb ",out_transE_entity_emb)
print("out_transE_relation_emb ",out_transE_relation_emb)


# entity_id = './benchmarks/FB15K/randomly_entity_id50.txt'
# relation_id = './benchmarks/FB15K/randomly_relation_id50.txt'
# out_entity_embs, out_rel_embs = read_new_init_embs(entity_id,relation_id)
# print("out_transE_entity_emb ",entity_id)
# print("out_transE_relation_emb ",relation_id)


new_entity_embs_path = './benchmarks/FB15K/new_init_entity_embedding_no_symbol_no_description_id0_des300.txt'
new_rel_embs_path = './benchmarks/FB15K/new_init_relation_embedding_no_symbol_no_description_id0_des300.txt'
print("entity_embs ",new_entity_embs_path)
print("relation_embs ",new_rel_embs_path)

entity_embs, rel_embs = read_new_init_embs(new_entity_embs_path,new_rel_embs_path)

print("entity_embs.shape ", entity_embs.shape)
print("rel_embs.shape ",rel_embs.shape)
import time
time.sleep(1)


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB15K/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

train_dataloader.set_init_embeddings(entity_embs, rel_embs)
train_dataloader.set_trains_out_embeddings(out_entity_embs, out_rel_embs)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB15K/", "link")

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	init_en_embed = train_dataloader.get_entity_embedding(),
	init_rel_embed = train_dataloader.get_rel_embedding(),
	per_out_ent_embed= train_dataloader.get_transe_out_entity_embedding(),
	per_out_rel_embed = train_dataloader.get_transe_out_rel_embedding(),
	dim =300, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 1.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 20000, alpha =1 , use_gpu = True)

time.sleep(1)

# print("trainer_run .... " + '\n')
# trainer.run()
# transe.save_checkpoint('./checkpoint/transe_no_symbol_no_description_id100_des300_e20000.ckpt')

# test the model
print("tester_run .... " + '\n')
# print("test 1-1: \n")
transe.load_checkpoint('./checkpoint/transe_no_symbol_no_description_id100_des300_e20000.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False) 

acc, thread = tester.run_triple_classification()

print("Triples Classification: ")
print("Accuracy: ", acc)
print("Thread: ", thread)






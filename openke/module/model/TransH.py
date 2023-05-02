import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import logging

logger = logging.getLogger(__name__)


class TransH(Model):

	def __init__(self, ent_tot, rel_tot, init_en_embed, init_rel_embed, per_out_ent_embed, per_out_rel_embed, id_dim, dim, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransH, self).__init__(ent_tot, rel_tot)
		
		self.dim = dim # the dimension of word_embedding
		self.id_dim = id_dim # the dimension of pre-trained embedding
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		# self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		# self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		# self.dim + self.id_dim
		self.norm_vector = nn.Embedding(self.rel_tot, self.dim + self.id_dim)

		# self.norm_vector = nn.Embedding(self.rel_tot, self.dim)

		# # pre-trained embedding
		# self.pre_out_ent_embeddings = nn.Embedding(self.ent_tot,self.id_dim)
		# self.pre_out_rel_embeddings = nn.Embedding(self.rel_tot,self.id_dim)

		# pre-out + word embedding
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim + self.id_dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim + self.id_dim)

		if margin == None or epsilon == None:

			logger.info("  Init entity and relation embedding ")
			entity_embedding = torch.cat([per_out_ent_embed, init_en_embed],dim=1)
			relation_embedding = torch.cat([per_out_rel_embed, init_rel_embed],dim=1)
			self.ent_embeddings.weight.data = entity_embedding
			self.rel_embeddings.weight.data = relation_embedding

			logger.info(" entity.shape ", self.ent_embeddings.weight.data.shape)
			logger.info(" relation.shape ", self.rel_embeddings.weight.data.shape)


			# print("\nInit fun ....")
			# print("load word embedding data....")
			# print("self.ent_embeddings: ",self.dim)
			# self.ent_embeddings.weight.data = init_en_embed   # give value init_ent_embs comes from init_entity_embedding.txt
			# self.rel_embeddings.weight.data = init_rel_embed   # give value
			#
			# # print("load random data....")
			# # nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			# # nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			#
			# print("load pre-out embeddings....")
			#
			# self.pre_out_ent_embeddings.weight.data = per_out_ent_embed
			# self.pre_out_rel_embeddings.weight.data = per_out_rel_embed
			# print("per_out_rel_embed \n",per_out_rel_embed.shape)

			# nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			# nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			# nn.init.xavier_uniform_(self.norm_vector.weight.data)
		else:

			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.norm_vector.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)

		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _calc(self, h, t, r, mode):
		if self.norm_flag:
			h = F.normalize(h, 2, -1)
			r = F.normalize(r, 2, -1)
			t = F.normalize(t, 2, -1)
		if mode != 'normal':
			h = h.view(-1, r.shape[0], h.shape[-1])
			t = t.view(-1, r.shape[0], t.shape[-1])
			r = r.view(-1, r.shape[0], r.shape[-1])
		if mode == 'head_batch':
			score = h + (r - t)
		else:
			score = (h + r) - t
		score = torch.norm(score, self.p_norm, -1).flatten()
		return score

	def _transfer(self, e, norm):
		norm = F.normalize(norm, p = 2, dim = -1)
		if e.shape[0] != norm.shape[0]:
			e = e.view(-1, norm.shape[0], e.shape[-1])
			norm = norm.view(-1, norm.shape[0], norm.shape[-1])
			e = e - torch.sum(e * norm, -1, True) * norm
			return e.view(-1, e.shape[-1])
		else:
			return e - torch.sum(e * norm, -1, True) * norm

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		mode = data['mode']

		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# h_des = self.ent_embeddings(batch_h)
		# t_des = self.ent_embeddings(batch_t)
		# r_des = self.rel_embeddings(batch_r)
		#
		# # print(h_des)
		# # print("h_des.shape",h_des.shape)
		#
		# h_s = self.pre_out_ent_embeddings(batch_h)
		# t_s = self.pre_out_ent_embeddings(batch_t)
		# r_s = self.pre_out_rel_embeddings(batch_r)
		#
		# # print(h_s)
		# # print("h_s.shape",h_s.shape)
		#
		# # connection
		# h = torch.cat([h_s,h_des], dim=-1)
		# t = torch.cat([t_s,t_des], dim=-1)
		# r = torch.cat([r_s,r_des], dim=-1)

		r_norm = self.norm_vector(batch_r)

		h = self._transfer(h, r_norm)
		t = self._transfer(t, r_norm)
		score = self._calc(h ,t, r, mode)
		if self.margin_flag:
			return self.margin - score
		else:
			return score

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']

		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)

		# h_des = self.ent_embeddings(batch_h)
		# t_des = self.ent_embeddings(batch_t)
		# r_des = self.rel_embeddings(batch_r)
		#
		# h_s = self.pre_out_ent_embeddings(batch_h)
		# t_s = self.pre_out_ent_embeddings(batch_t)
		# r_s = self.pre_out_rel_embeddings(batch_r)
		#
		# # connection
		# h = torch.cat([h_s,h_des], dim=-1)
		# t = torch.cat([t_s,t_des], dim=-1)
		# r = torch.cat([r_s,r_des], dim=-1)

		r_norm = self.norm_vector(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) +
				 torch.mean(r_norm ** 2)) / 4
		return regul
	
	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()

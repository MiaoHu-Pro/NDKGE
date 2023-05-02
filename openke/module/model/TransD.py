import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
import logging

logger = logging.getLogger(__name__)

class TransD(Model):

	def __init__(self, ent_tot, rel_tot, init_en_embed, init_rel_embed, per_out_ent_embed, per_out_rel_embed, id_dim, dim_e = 100, dim_r = 100, p_norm = 1, norm_flag = True, margin = None, epsilon = None):
		super(TransD, self).__init__(ent_tot, rel_tot)
		
		self.dim_e = dim_e
		self.dim_r = dim_r
		self.id_dim = id_dim
		self.margin = margin
		self.epsilon = epsilon
		self.norm_flag = norm_flag
		self.p_norm = p_norm

		# # pre-trained embedding
		# self.pre_out_ent_embeddings = nn.Embedding(self.ent_tot,self.id_dim)
		# self.pre_out_rel_embeddings = nn.Embedding(self.rel_tot,self.id_dim)

		# pre id + word embedding
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e + self.id_dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_e + self.id_dim)

		# transfer self.dim_e + self.id_dim
		self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e + self.id_dim)
		self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r + self.id_dim)

		# # transfer self.dim_e + self.id_dim
		# self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
		# self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)


		if margin == None or epsilon == None:

			logger.info("  Init entity and relation embedding ")
			entity_embedding = torch.cat([per_out_ent_embed,init_en_embed],dim=1)
			relation_embedding = torch.cat([per_out_rel_embed,init_rel_embed],dim=1)
			self.ent_embeddings.weight.data = entity_embedding
			self.rel_embeddings.weight.data = relation_embedding

			logger.info(" entity.shape ", self.ent_embeddings.weight.data.shape)
			logger.info(" relation.shape ", self.rel_embeddings.weight.data.shape)


			# # nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			# # nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			# print("\nInit fun ....")
			# print("load word embedding data....")
			# print("self.ent_embeddings: ",self.dim_e)
			# self.ent_embeddings.weight.data = init_en_embed   # give value init_ent_embs comes from init_entity_embedding.txt
			# self.rel_embeddings.weight.data = init_rel_embed   # give value
			#
			# # print("load random data....")
			# # nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			# # nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
			#
			# print("load pre-out embeddings....")
			# self.pre_out_ent_embeddings.weight.data = per_out_ent_embed
			# self.pre_out_rel_embeddings.weight.data = per_out_rel_embed
			# print("per_out_rel_embed \n",per_out_rel_embed.shape)

			nn.init.xavier_uniform_(self.ent_transfer.weight.data)
			nn.init.xavier_uniform_(self.rel_transfer.weight.data)

		else:
			self.ent_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
			)
			self.rel_embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.ent_embedding_range.item(), 
				b = self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.ent_transfer.weight.data, 
				a= -self.ent_embedding_range.item(), 
				b= self.ent_embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_transfer.weight.data, 
				a= -self.rel_embedding_range.item(), 
				b= self.rel_embedding_range.item()
			)
		if margin != None:
			self.margin = nn.Parameter(torch.Tensor([margin]))
			self.margin.requires_grad = False
			self.margin_flag = True
		else:
			self.margin_flag = False

	def _resize(self, tensor, axis, size):
		shape = tensor.size()
		osize = shape[axis]
		if osize == size:
			return tensor
		if (osize > size):
			return torch.narrow(tensor, axis, 0, size)
		paddings = []
		for i in range(len(shape)):
			if i == axis:
				paddings = [0, size - osize] + paddings
			else:
				paddings = [0, 0] + paddings
		print (paddings)
		return F.pad(tensor, paddings = paddings, mode = "constant", value = 0)

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

	def _transfer(self, e, e_transfer, r_transfer):
		if e.shape[0] != r_transfer.shape[0]:
			e = e.view(-1, r_transfer.shape[0], e.shape[-1])
			e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
			r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
			e = F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)			
			return e.view(-1, e.shape[-1])
		else:
			return F.normalize(
				self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
				p = 2, 
				dim = -1
			)

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
		# h_s = self.pre_out_ent_embeddings(batch_h)
		# t_s = self.pre_out_ent_embeddings(batch_t)
		# r_s = self.pre_out_rel_embeddings(batch_r)
		#
		# # connection
		# h = torch.cat([h_s,h_des], dim=-1)
		# t = torch.cat([t_s,t_des], dim=-1)
		# r = torch.cat([r_s,r_des], dim=-1)

		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		h = self._transfer(h, h_transfer, r_transfer)
		t = self._transfer(t, t_transfer, r_transfer)
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

		h_transfer = self.ent_transfer(batch_h)
		t_transfer = self.ent_transfer(batch_t)
		r_transfer = self.rel_transfer(batch_r)
		regul = (torch.mean(h ** 2) + 
				 torch.mean(t ** 2) + 
				 torch.mean(r ** 2) + 
				 torch.mean(h_transfer ** 2) + 
				 torch.mean(t_transfer ** 2) + 
				 torch.mean(r_transfer ** 2)) / 6
		return regul

	def predict(self, data):
		score = self.forward(data)
		if self.margin_flag:
			score = self.margin - score
			return score.cpu().data.numpy()
		else:
			return score.cpu().data.numpy()

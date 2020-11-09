import torch
import torch.nn as nn
import torch.nn.init as init
import math

from pytorch_pretrained_bert.modeling import BertModel,BertEmbeddings,BertPooler,BertEncoder

class VocabGraphConvolution(nn.Module):
    def __init__(self,voc_dim, num_adj, hid_dim, out_dim, dropout_rate=0.2):
        super(VocabGraphConvolution, self).__init__()
        self.voc_dim=voc_dim
        self.num_adj=num_adj
        self.hid_dim=hid_dim
        self.out_dim=out_dim

        for i in range(self.num_adj):
            setattr(self, 'W%d_vh'%i, nn.Parameter(torch.randn(voc_dim, hid_dim)))

        self.fc_hc=nn.Linear(hid_dim,out_dim) 
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.reset_parameters()

    def reset_parameters(self):
        for n,p in self.named_parameters():
            if n.startswith('W') or n.startswith('a') or n in ('W','a','dense'):
                init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, vocab_adj_list, X_dv, add_linear_mapping_term=False):
        for i in range(self.num_adj):
            H_vh=vocab_adj_list[i].mm(getattr(self, 'W%d_vh'%i))
            H_vh=self.dropout(H_vh)
            H_dh=X_dv.matmul(H_vh)

            if add_linear_mapping_term:
                H_linear=X_dv.matmul(getattr(self, 'W%d_vh'%i))
                H_linear=self.dropout(H_linear)
                H_dh+=H_linear

            if i == 0:
                fused_H = H_dh
            else:
                fused_H += H_dh

        out=self.fc_hc(fused_H)
        return out

class Pretrain_VGCN(nn.Module):
    def __init__(self, word_emb, word_emb_dim, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim, num_labels,dropout_rate=0.2):
        super(Pretrain_VGCN, self).__init__()
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.word_emb=word_emb
        self.vocab_gcn=VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim) #192/256
        self.classifier = nn.Linear(gcn_embedding_dim*word_emb_dim, num_labels)
    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
        words_embeddings = self.word_emb(input_ids)
        vocab_input=gcn_swop_eye.matmul(words_embeddings).transpose(1,2)
        gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input).transpose(1,2)
        gcn_vocab_out=self.dropout(self.act_func(gcn_vocab_out))
        out=self.classifier(gcn_vocab_out.flatten(start_dim=1))
        return out


class VGCNBertEmbeddings(BertEmbeddings):
    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim):
        super(VGCNBertEmbeddings, self).__init__(config)
        assert gcn_embedding_dim>=0
        self.gcn_embedding_dim=gcn_embedding_dim
        self.vocab_gcn=VocabGraphConvolution(gcn_adj_dim, gcn_adj_num, 128, gcn_embedding_dim) #192/256

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None):
        words_embeddings = self.word_embeddings(input_ids)
        vocab_input=gcn_swop_eye.matmul(words_embeddings).transpose(1,2)
        
        if self.gcn_embedding_dim>0:
            gcn_vocab_out = self.vocab_gcn(vocab_adj_list, vocab_input)
         
            gcn_words_embeddings=words_embeddings.clone()
            for i in range(self.gcn_embedding_dim):
                tmp_pos=(attention_mask.sum(-1)-2-self.gcn_embedding_dim+1+i)+torch.arange(0,input_ids.shape[0]).to(input_ids.device)*input_ids.shape[1]
                gcn_words_embeddings.flatten(start_dim=0, end_dim=1)[tmp_pos,:]=gcn_vocab_out[:,:,i]

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.gcn_embedding_dim>0:
            embeddings = gcn_words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VGCN_Bert(BertModel):

    def __init__(self, config, gcn_adj_dim, gcn_adj_num, gcn_embedding_dim, num_labels, output_attentions=False, keep_multihead_output=False):
        super(VGCN_Bert, self).__init__(config,output_attentions,keep_multihead_output)
        self.embeddings = VGCNBertEmbeddings(config,gcn_adj_dim,gcn_adj_num, gcn_embedding_dim)
        self.encoder = BertEncoder(config, output_attentions=output_attentions,
                                           keep_multihead_output=keep_multihead_output)
        self.pooler = BertPooler(config)
        self.num_labels=num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.will_collect_cls_states=False
        self.all_cls_states=[]
        self.output_attentions=output_attentions

        self.apply(self.init_bert_weights)

    def forward(self, vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False, head_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        embedding_output = self.embeddings(vocab_adj_list, gcn_swop_eye, input_ids, token_type_ids,attention_mask)


        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) 
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand_as(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1) 
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) 
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if self.output_attentions:
            output_all_encoded_layers=True
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers = encoded_layers

        pooled_output = self.pooler(encoded_layers[-1])
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if self.output_attentions:
            return all_attentions, logits

        return logits

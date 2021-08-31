import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import pickle
import os
import numpy as np
import os

class Time2Vec(nn.Module):
    def __init__(self,vocab_size,n_embd ):
        super().__init__()
        self.time =  nn.Embedding(vocab_size, n_embd)
    def forward(self, x,word =None):
        return self.time(x)

    def save_embeddings(self):
        return self.time.weight.detach().cpu().numpy()
    def load_embeddings(self, paras):
        self.time.weight = nn.Parameter(torch.from_numpy(paras).float())
    def save_in_text_format(self, id2word, path, embeddings=None):
        pass


class Time2Sin(nn.Module):
    def __init__(self, hidden_size, fun_type = "mix", add_phase_shift = False):
        super().__init__()
    
        self.frequency_emb = nn.Parameter(torch.Tensor(hidden_size))
        if add_phase_shift:
            self.phase_emb = nn.Parameter(torch.Tensor(hidden_size))

        self.fun_type = fun_type
        self.add_phase_shift = add_phase_shift
        self.hidden_size = hidden_size

    def forward(self, x,word =None):
        phase = x.unsqueeze(-1).float() @ self.frequency_emb.unsqueeze(0)
        if self.fun_type == "mixed":
            if not self.add_phase_shift:
                encoded = torch.cat([torch.cos(phase[:,:self.hidden_size//2]), torch.sin(phase[:,self.hidden_size//2:])], -1)
            else:
                encoded = torch.cat([torch.cos(phase[:,:self.hidden_size//2]+self.phase_emb[:self.hidden_size//2]), torch.sin(phase[:,self.hidden_size//2:]+self.phase_emb[self.hidden_size//2:])], -1)
        else:
            if self.fun_type == "cos":
                encoded = torch.cos(phase)
            elif self.fun_type == "sin":
                encoded = torch.sin(phase)
            if self.add_phase_shift:
                encoded += self.phase_emb
        # print("sin used")
        return encoded
    def save_embeddings(self):
        if self.add_phase_shift:
            return self.frequency_emb.cpu().data.numpy(),self.phase_emb.cpu().data.numpy()
        else:
            return self.frequency_emb.cpu().data.numpy()
    def load_embeddings(self, paras):
        if len(paras) ==2:
            self.frequency_emb.weight =  nn.Parameter(torch.from_numpy(paras[0]).float())
            self.phase_emb.weight = nn.Parameter(torch.from_numpy(paras[1]).float())
        else:
            self.frequency_emb.weight =  nn.Parameter(torch.from_numpy(paras).float())
    def save_in_text_format(self, id2word,path, embeddings=None):
        pass

class Time2Linear(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.slope = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, x,word =None):
        encoded = x.unsqueeze(-1).float() @ self.slope.unsqueeze(0) + self.bias
        return encoded

    def save_embeddings(self):
        return self.slope.cpu().data.numpy(), self.bias.cpu().data.numpy()
    def load_embeddings(self, paras):
        self.slope.data,self.bias.data = nn.Parameter(torch.from_numpy(paras[0]).float()),nn.Parameter(torch.from_numpy(paras[1]).float())
    def save_in_text_format(self, id2word,path, embeddings=None):
        pass


class WordawareEncoder(nn.Module):
    def __init__(self, hidden_size,vocab_size,operator = "linear", add_phase_shift = True,  add_amplitude = False, frequencies = None, dropout=0, fre_pattern ="1-10000" ):
        super().__init__()
        # dropout not used
        self.add_phase_shift = add_phase_shift
        self.para_embedding = nn.Embedding(vocab_size, hidden_size)

        if add_phase_shift:
            self.phase_shift_embedding = nn.Embedding(vocab_size, hidden_size)
        self.operator = operator
        self.hidden_size = hidden_size
        self.add_amplitude = add_amplitude

        if self.add_amplitude:
            self.amplitude_embedding = nn.Embedding(vocab_size, hidden_size)
        
        if  operator == "mixed_fixed":
            if frequencies is None:
                base, divided = [int(x) for x in fre_pattern.split("-")]
                print("use frequencey parttern with the base {} and the divided {}".format(base,divided))
                frequencies = [base/np.power(divided,2 * (hid_ixd)/hidden_size ) for hid_ixd in range(hidden_size//2)]
            assert len(frequencies) == self.hidden_size//2 , "fixed frequencies size not match"
            self.frequencies = torch.nn.parameter.Parameter(torch.Tensor(frequencies),requires_grad=False)


    def forward(self, _time,word ):
        time = _time.unsqueeze(-1).repeat([1,self.hidden_size])
        if self.operator == "linear":
            return self.para_embedding(word)*time

        if self.operator == "mixed_fixed":
            # frequencies = self.frequencies.unsqueeze(0).repeat([word.size(0),1])  # d/2 -> b * d/2
            omega = self.frequencies*(_time.unsqueeze(-1).repeat([1,self.hidden_size//2])) # (b * d/2) * (b * d/2)
            if self.add_phase_shift:
                init_phase = self.phase_shift_embedding(word)
                phase = torch.cat([torch.cos(omega+ init_phase[:,:self.hidden_size//2]) , torch.sin(omega + init_phase[:,self.hidden_size//2:])], -1)
            else:
                phase = torch.cat([torch.cos(omega) , torch.sin(omega)], -1)
            amplitute =  self.para_embedding(word)
            return amplitute * phase

        phase = self.para_embedding(word)*(time.float())

        if self.add_phase_shift:
            phase += self.phase_shift_embedding(word)


        if self.operator == "cos":
            output= torch.cos(phase)
        elif self.operator == "sin":
            output= torch.sin(phase)
        elif self.operator == "mixed":
            # print("mixed with cos and sine")
            output=  torch.cat([torch.cos(phase[:,:self.hidden_size//2]), torch.sin(phase[:,self.hidden_size//2:])], -1)
        else:
            exit("not implemented")
        try:
            ampli = self.amplitude_embedding(word)
        except:
            return output
        return output * ampli


    def save_embeddings(self):
        if not self.add_phase_shift:
            return self.para_embedding.weight.detach().cpu().numpy()
        else:
            return self.para_embedding.weight.detach().cpu().numpy(),self.phase_shift_embedding.cpu().weight.detach().numpy()
        
    def load_embeddings(self, paras):
        if len(paras) ==2 :
            self.para_embedding.weight = nn.Parameter(torch.from_numpy(paras[0]).float())
            self.phase_shift_embedding.weight = nn.Parameter(torch.from_numpy(paras[1]).float())
        else:
            self.para_embedding.weight = nn.Parameter(torch.from_numpy(paras).float())





def load_time_machine(embedding_type, hidden_size = None, vocab_size = None,add_phase_shift = False ,dropout =0,fre_pattern ="1-10000"):
    assert hidden_size != None, "hidden_size should not be not none "
    print("used {}".format(embedding_type))
    if embedding_type == "linear":
        return Time2Linear(hidden_size)
    elif embedding_type == "sin":
        return Time2Sin(hidden_size,fun_type="sin",add_phase_shift=add_phase_shift)
    elif embedding_type == "cos":
        return Time2Sin(hidden_size,fun_type="cos",add_phase_shift=add_phase_shift)
    elif embedding_type == "mixed":
        return Time2Sin(hidden_size,fun_type="mixed",add_phase_shift=add_phase_shift)
    elif embedding_type == "word_linear":
        return WordawareEncoder(hidden_size,vocab_size, "linear",add_phase_shift=add_phase_shift, dropout = dropout)
    elif embedding_type == "word_sin":
        return WordawareEncoder(hidden_size,vocab_size, "sin",add_phase_shift=add_phase_shift, dropout = dropout)
    elif embedding_type == "word_cos":
        return WordawareEncoder(hidden_size,vocab_size, "cos",add_phase_shift=add_phase_shift, dropout = dropout)
    elif embedding_type == "word_cos_amplitude":
        return WordawareEncoder(hidden_size,vocab_size, "cos",add_phase_shift=add_phase_shift,add_amplitude = True, dropout = dropout)
    elif embedding_type == "word_sin_amplitude":
        return WordawareEncoder(hidden_size,vocab_size, "cos",add_phase_shift=add_phase_shift,add_amplitude = True, dropout = dropout)
    elif embedding_type == "word_mixed":
        return WordawareEncoder(hidden_size,vocab_size, "mixed",add_phase_shift=add_phase_shift, dropout = dropout)

    elif embedding_type == "word_mixed_fixed":
        return WordawareEncoder(hidden_size,vocab_size, "mixed_fixed",add_phase_shift=add_phase_shift, dropout = dropout,fre_pattern = fre_pattern)
    elif embedding_type == "word_mixed_amplitude":
        return WordawareEncoder(hidden_size, vocab_size, "mixed", add_phase_shift=add_phase_shift,add_amplitude = True, dropout = dropout)
    elif embedding_type == "word_mixed_fixed_no_amplitude":
        return WordawareEncoder(hidden_size, vocab_size, "word_mixed", add_phase_shift=add_phase_shift,add_amplitude = False, dropout = dropout, fre_pattern = fre_pattern)
    else:
        assert vocab_size != None, "vocab_size should not be none "
        return Time2Vec(vocab_size,hidden_size)





class TimestampedSkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, time_type, add_phase_shift = False,dropout= 0,fre_pattern="1-10000", in_batch_negative = False):
        super(TimestampedSkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.in_batch_negative = in_batch_negative
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension) # sparse=True
        self.add_phase_shift = add_phase_shift
        self.time_encoder = load_time_machine(time_type, vocab_size = emb_size, hidden_size = emb_dimension, add_phase_shift = add_phase_shift, dropout = dropout,fre_pattern=fre_pattern )

        # dropout not used
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward_embedding(self,pos_u,time=None):
        emb_u = self.u_embeddings(pos_u)
        if time is not None:
            # emb_u += self.time_encoder(time)
            emb_u += self.time_encoder(time,pos_u)
        return emb_u

    def get_temporal_embedding(self,word,time):
        return self.forward_embedding(word,time).cpu().data.numpy()


    def forward(self, pos_u, pos_v, neg_v,time=None):

        emb_u = self.forward_embedding(pos_u,time)
        emb_v = self.v_embeddings(pos_v)

        if not self.in_batch_negative:

            emb_neg_v = self.v_embeddings(neg_v)
            score = torch.sum(torch.mul(emb_u, emb_v), dim=1)

            score = torch.clamp(score, max=10, min=-10)
            score = -F.logsigmoid(score)


            neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
            neg_score = torch.clamp(neg_score, max=10, min=-10)
            neg_score = -torch.mean(F.logsigmoid(-neg_score), dim=1) #-torch.sum(F.logsigmoid(-neg_score), dim=1)
        else:
            in_batch_score = emb_u @ emb_v.transpose(0,1)
            in_batch_score = torch.clamp(in_batch_score, max=10, min=-10)

            score = -F.logsigmoid(torch.diagonal(in_batch_score))
            n1,n2 = in_batch_score.size()
            # off_diagonal =   in_batch_score.masked_select(~torch.eye(n1, dtype=bool)).view(n1, n1 - 1)
            off_diagonal = in_batch_score.flatten()[1:].view(n1 - 1, n1 + 1)[:, :-1].reshape(n1, n1 - 1)

            neg_score = -torch.mean(F.logsigmoid(-off_diagonal), dim=1)

        return torch.mean(score + neg_score), torch.mean(score), torch.mean(neg_score)

    def save_in_text_format(self, id2word, path):
        embeddings = self.u_embeddings.weight.cpu().data.numpy()
        return self.time_encoder.save_in_text_format(id2word,path,embeddings)

    def save_embedding(self, id2word, path):
        # if self.add_phase_shift:
        #     path = path + "_shift"
        if not os.path.exists(path):
            os.mkdir(path)

        embedding = self.u_embeddings.weight.cpu().data.numpy()
        file_name = os.path.join(path,"vectors.txt")
        with open(file_name, 'w',encoding = "utf-8") as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
        pickle.dump( id2word,open("{}/dict.pkl".format(path),"wb"))
        pickle.dump( self.time_encoder.save_embeddings(),open("{}/para.pkl".format(path),"wb"))


    def save_dict(self,id2word,path):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path,"vocab.txt"), 'w',encoding = "utf-8") as f:
            for wid, w in id2word.items():
                f.write('{}\t{}\n' .format(wid, w))

    def read_embeddings_from_file(self,file_name):
        embedding_dict = dict()
        with open(file_name) as f:
            for i,line in enumerate(f):
                if i==0:
                    vocab_size,emb_dimension = [int(item) for item in line.split()]
                    # embeddings= np.zeros([vocab_size,emb_dimension])
                else:
                    tokens = line.split()
                    word, vector = tokens[0], [float(num_str) for num_str in tokens[1:]]
                    embedding_dict[word] = vector
        return embedding_dict
                     

    def load_embeddings(self, id2word, path):
        file_name = os.path.join(path,"vectors.txt")
        print("load embeddings from " + file_name)
        word2id = {value:key for key,value in id2word.items()}


        with open(file_name,encoding="utf-8") as f:
            for i,line in enumerate(f):
                if i==0:
                    vocab_size,emb_dimension = [int(item) for item in line.split()]
                    embeddings= np.zeros([vocab_size,emb_dimension])
                else:
                    tokens = line.split()
                    word, vector = tokens[0], [float(num_str) for num_str in tokens[1:]]
                    embeddings[word2id[word]]=vector
        self.u_embeddings.weight = torch.nn.Parameter(torch.from_numpy(embeddings).float())
        other_embeddings_pkl = "{}/para.pkl".format(path)
        a = pickle.load(open(other_embeddings_pkl,"rb"))
        # print(a)

        self.time_encoder.load_embeddings(a)
    def get_embedding(self,id2word,word, year =None):
        word2id = {value:key for key,value in id2word.items()}
        id_of_word = word2id(word)
        # embeddings = self.u_embeddings.weight.cpu().data.numpy()
        # embed = embeddings[id_of_word]
        word,time = torch.FloatTensor([word]),torch.FloatTensor([year])
        emb_u = self.u_embeddings(word)
        if time is not None:
            # emb_u += self.time_encoder(time)
            emb_u += self.time_encoder(time,emb_u)
        return emb_u.cpu().data.numpy()


class DE(nn.Module):
    def __init__(self, emb_size, emb_dimension, time_type,dropout= 0):
        super(DE, self).__init__()
        self.emb_dimension = emb_dimension
        # time encoder
        self.dense1 = nn.Linear(emb_dimension,emb_dimension)
        self.dense2 = nn.Linear(emb_dimension, emb_dimension)
        self.dense4 = nn.Linear(emb_dimension, emb_dimension)
        # word encoder
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension*3)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension*3)
        self.T = nn.Parameter(torch.randn(emb_dimension,emb_dimension,emb_dimension*3))
        self.B = nn.Parameter(torch.randn(emb_dimension,emb_dimension*3))

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def word_embedding(self, pos_u,timevec =None):
        emb_u = self.u_embeddings(pos_u)
        trans_w = torch.einsum('ijk,bk->bij', self.T, emb_u)
        h3 = torch.einsum('bij,bi->bj', trans_w, timevec)
        use_w = self.dense4(h3)
        return use_w

    def time_encoding(self,time):
        h1 = torch.tanh(self.dense1(time.unsqueeze(-1).repeat(1, self.emb_dimension).float()))
        timevec = torch.tanh(self.dense2(h1))
        return  timevec
    def forward(self, pos_u, pos_v, neg_v,time=None):

        timevec = self.time_encoding(time)
        use_w = self.word_embedding(pos_u,timevec)

        #encoding target for postive
        emb_v = self.v_embeddings(pos_v)
        trans_w_v = torch.einsum('ijk,bk->bij', self.T, emb_v)
        h3_v = torch.einsum('bij,bi->bj', trans_w_v, timevec)
        use_c_v = self.dense4(h3_v)

        # encoding targets for negative
        emb_v_neg = self.v_embeddings(neg_v)
        trans_w_v_neg = torch.einsum('ijk,blk->blij', self.T, emb_v_neg) # l is the numbers of nagetive samples
        h3_v_neg = torch.einsum('blij,bli->blj', trans_w_v_neg, timevec.unsqueeze(-2).repeat(1,neg_v.size(1) ,1))
        use_c_v_neg = self.dense4(h3_v_neg)

        score = torch.sum(torch.mul(use_w, use_c_v), dim=1)

        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(use_c_v_neg, use_w.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.mean(F.logsigmoid(-neg_score), dim=1) #-torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score), torch.mean(score), torch.mean(neg_score)

    def get_temporal_embedding(self,word,time):
        timevec = self.time_encoding(time)
        use_w = self.word_embedding(word, timevec)
        return use_w.cpu().data.numpy()

    def get_embedding(self,id2word,word, year =None):
        word2id = {value:key for key,value in id2word.items()}
        id_of_word = word2id(word)
        # embeddings = self.u_embeddings.weight.cpu().data.numpy()
        # embed = embeddings[id_of_word]
        word,time = torch.FloatTensor([word]),torch.FloatTensor([year])
        timevec = self.time_encoding(time)
        use_w = self.word_embedding(word, timevec)
        return use_w.cpu().data.numpy()


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension) #, sparse=True
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension) # , sparse=True

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward_embedding(self,pos_u,time=None):
        emb_u = self.u_embeddings(pos_u)
        return emb_u
    def get_temporal_embedding(self,word,time):
        return self.forward_embedding(word,time).cpu().data.numpy()

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.mean(F.logsigmoid(-neg_score), dim=1)


        return torch.mean(score + neg_score),torch.mean(score), torch.mean(neg_score)

    def save_embedding(self, id2word, path):
        if not os.path.exists(path):
            os.mkdir(path)
        file_name = os.path.join(path,"vectors.txt")
        print("save in " + file_name)
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
    def load_embeddings(self, id2word, path):
        
        print("not implemented")

        


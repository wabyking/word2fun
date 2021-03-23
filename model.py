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
        # print("Time2Vec used")
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
        # print(x.unsqueeze(-1).shape)
        # print(self.frequency_emb.unsqueeze(0).shape)
        # print(x.unsqueeze(-1) @ self.frequency_emb.unsqueeze(0))
        phase = x.unsqueeze(-1).float() @ self.frequency_emb.unsqueeze(0)
        if self.fun_type == "mixed":
            # if not self.add_phase_shift:
            #     encoded = torch.cat([torch.cos(phase), torch.sin(phase)], -1)
            # else:
            #     encoded = torch.cat([torch.cos(phase)+self.phase_emb_cos, torch.sin(phase)+self.phase_emb_sin], -1)
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
        # print(x.unsqueeze(-1).shape)
        # print(self.frequency_emb.unsqueeze(0).shape)
        # print(x.unsqueeze(-1) @ self.frequency_emb.unsqueeze(0))
        encoded = x.unsqueeze(-1).float() @ self.slope.unsqueeze(0) + self.bias
        # print("linear used")
        return encoded

    def save_embeddings(self):
        return self.slope.cpu().data.numpy(), self.bias.cpu().data.numpy()
    def load_embeddings(self, paras):
        self.slope.data,self.bias.data = nn.Parameter(torch.from_numpy(paras[0]).float()),nn.Parameter(torch.from_numpy(paras[1]).float())
    def save_in_text_format(self, id2word,path, embeddings=None):
        pass


class WordawareEncoder(nn.Module):
    def __init__(self, hidden_size,vocab_size,operator = "linear", add_phase_shift = True, fixed_frequencies = False, frequencies = None ):
        super().__init__()
        self.add_phase_shift = add_phase_shift
        self.para_embedding = nn.Embedding(vocab_size, hidden_size)
        if add_phase_shift:
            self.phase_shift_embedding = nn.Embedding(vocab_size, hidden_size)
        self.operator = operator
        self.hidden_size = hidden_size
        
        if  operator == "mixed_fixed":
            if frequencies is None: 
                frequencies = [1/np.power(10000,2 * (hid_ixd)/hidden_size ) for hid_ixd in range(hidden_size//2)]
            assert len(frequencies) == self.hidden_size//2 , "fixed frequencies size not match"
            self.frequencies = torch.nn.parameter.Parameter(torch.Tensor(frequencies),requires_grad=False)



    def forward(self, _time,word ):
        time = _time.unsqueeze(-1).repeat([1,self.hidden_size])
        if self.operator == "linear":
            return self.para_embedding(word)*time

        if self.operator == "mixed_fixed":
            frequencies = self.frequencies.unsqueeze(0).repeat([word.size(0),1])  # d/2 -> b * d/2
            omega = self.frequencies*(_time.unsqueeze(-1).repeat([1,self.hidden_size//2])) # (b * d/2) * (b * d/2)
            if self.add_phase_shift:
                init_phase = self.phase_shift_embedding(word)
                phase = torch.cat([torch.cos(omega+ init_phase[:,:self.hidden_size//2]) , torch.sin(omega + init_phase[:,self.hidden_size//2:])], -1)
            else:
                phase = torch.cat([torch.cos(omega) , torch.sin(omega)], -1)
            amplitute =  self.para_embedding(word)
            return amplitute * phase


        phase = self.para_embedding(word)*time
        # print(self.para_embedding)
        if self.add_phase_shift:
            try:
            # print(self.phase_shift_embedding.device)
                phase+= self.phase_shift_embedding(word)
            except:
                self.cuda()
                phase+= self.phase_shift_embedding(word)

        if self.operator == "cos":
            return torch.cos(phase)
        elif self.operator == "sin":
            return torch.sin(phase)
        elif self.operator == "mixed":
            # print("mixed with cos and sine")
            return  torch.cat([torch.cos(phase[:,:self.hidden_size//2]), torch.sin(phase[:,self.hidden_size//2:])], -1)
        else:
            exit("not implemented")

    def save_embeddings(self):
        if not self.add_phase_shift:
            # print(":used shift")
            return self.para_embedding.weight.detach().cpu().numpy()
        else:
            # print("do not used shift")
            return self.para_embedding.weight.detach().cpu().numpy(),self.phase_shift_embedding.cpu().weight.detach().numpy()
        
    def load_embeddings(self, paras):
        if len(paras) ==2 :
            self.para_embedding.weight = nn.Parameter(torch.from_numpy(paras[0]).float())
            self.phase_shift_embedding.weight = nn.Parameter(torch.from_numpy(paras[1]).float())
        else:
            self.para_embedding.weight = nn.Parameter(torch.from_numpy(paras).float())

    def save_in_text_format(self, id2word, path, embeddings = None):
        if not os.path.exists(path):
            os.mkdir(path)

        # write in new format!!!
        para_embedding = self.para_embedding.weight.cpu().data.numpy()
        if self.add_phase_shift:
            phase_shift_embedding = self.phase_shift_embedding.weight.cpu().data.numpy()
        if self.operator == "mixed_fixed":
            frequencies = self.frequencies.cpu().data.numpy()

        file_name = os.path.join(path,"functions.txt")
        print("write in :" + file_name)
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.hidden_size))
            for wid, w in id2word.items():
                word, para,  = embeddings[wid], para_embedding[wid]
                if self.add_phase_shift:
                    phase_shift = phase_shift_embedding[wid] # linear case would not use this

                if self.operator == "linear":
                    items = [ "{}+{}t".format(word[i],para[i]) for i in range(self.hidden_size)]
                
                else:
                    if self.operator == "mixed_fixed":
                        items = [ "{}+{}cos{}t".format(word[i], para[i],frequencies[i]) for i in range(self.hidden_size//2)]
                        items +=[ "{}+{}sin{}t".format(word[i], para[i],frequencies[i-self.hidden_size//2]) for i in range(self.hidden_size//2,self.hidden_size)]
                    elif self.operator == "cos" or self.operator == "sin":
                        items = [ "{}+{}{}t".format(word[i],self.operator, para[i]) for i in range(self.hidden_size)]
                    elif self.operator == "mixed":
                        items = [ "{}+cos{}t".format(word[i], para[i]) for i in range(self.hidden_size//2)]
                        items +=[ "{}+sin{}t".format(word[i], para[i]) for i in range(self.hidden_size//2,self.hidden_size)]
                    else:
                        exit("not implemented operator: %s".format(self.operator))
                    # print(items)
                    if self.add_phase_shift:
                            items = ["{}+{}".format(items[i],phase_shift[i]) for i in range(self.hidden_size)]
                f.write('%s %s\n' % (w,  ";\t".join(items) ))
        




def load_time_machine(embedding_type, hidden_size = None, vocab_size = None,add_phase_shift = False ):
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
    elif embedding_type == "word_sin":
        return WordawareEncoder(hidden_size,vocab_size, "sin",add_phase_shift=add_phase_shift)
    elif embedding_type == "word_cos":
        return WordawareEncoder(hidden_size,vocab_size, "cos",add_phase_shift=add_phase_shift)
    elif embedding_type == "word_mixed":
        return WordawareEncoder(hidden_size,vocab_size, "mixed",add_phase_shift=add_phase_shift)
    elif embedding_type == "word_linear":
        return WordawareEncoder(hidden_size,vocab_size, "linear",add_phase_shift=add_phase_shift)
    elif embedding_type == "word_mixed_fixed":
        return WordawareEncoder(hidden_size,vocab_size, "mixed_fixed",add_phase_shift=add_phase_shift)
    else:
        assert vocab_size != None, "vocab_size should not be none "
        return Time2Vec(vocab_size,hidden_size)





class TimestampedSkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, time_type, add_phase_shift = False):
        super(TimestampedSkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension) # sparse=True
        self.add_phase_shift = add_phase_shift
        self.time_encoder = load_time_machine(time_type, vocab_size = emb_size, hidden_size = emb_dimension, add_phase_shift = add_phase_shift )

        # self.word_time_encoder = WordawareEncoder(emb_dimension,emb_size)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward_embedding(self,pos_u,time=None):
        emb_u = self.u_embeddings(pos_u)
        if time is not None:
            # emb_u += self.time_encoder(time)
            emb_u += self.time_encoder(time,pos_u)
        return emb_u


    def forward(self, pos_u, pos_v, neg_v,time=None):
        # print(pos_u.shape)
        # print(pos_v.shape)
        # print(neg_v.shape)
        # print(time.shape)
        emb_u = self.forward_embedding(pos_u,time)
        emb_v = self.v_embeddings(pos_v)

        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.mean(F.logsigmoid(-neg_score), dim=1) #-torch.sum(F.logsigmoid(-neg_score), dim=1)



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
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))
        pickle.dump( id2word,open("{}/dict.pkl".format(path),"wb"))
        pickle.dump( self.time_encoder.save_embeddings(),open("{}/para.pkl".format(path),"wb"))


    def save_dict(self,id2word,path):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path,"vocab.txt"), 'w') as f:
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

        


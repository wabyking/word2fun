import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW,get_linear_schedule_with_warmup


from model import SkipGramModel,TimestampedSkipGramModel,DE
from data_reader import DataReader, Word2vecDataset,TimestampledWord2vecDataset
import json

import os
import argparse
import pickle
import numpy as np
# from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE


from functools import wraps
import time
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "Running the function {} takes {:.2f} seconds".format(func.__name__,delta))
        return ret
    return _deco


#word_sin word_cos word_mixed word_linear word_mixed_fixed
parser = argparse.ArgumentParser(description='parameter information')
parser.add_argument('--time_type', dest='time_type', type=str,default= "word_mixed_amplitude", help='sin cos  mixed others  linear, sin,  word_sin,word_cos,word_linear')
parser.add_argument('--text', dest='text', type=str,default= "nyt_yao_tiny.txt.norm.dev", help='text dataset')
parser.add_argument('--use_time', dest='use_time', default= 1, type=int, help='use_time or not')
parser.add_argument('--output', dest='output', default= "word2vec" , type=str, help='output dir to save embeddings')
parser.add_argument('--log_step', dest='log_step', default= 100 , type=int, help='log_step')
parser.add_argument('--from_scatch', dest='from_scatch', default= 1 , type=int, help='from_scatch or not')
parser.add_argument('--batch_size', dest='batch_size', default= 2, type=int, help='batch_size')
parser.add_argument('--emb_dimension', dest='emb_dimension', default= 100 , type=int, help='emb_dimension')
parser.add_argument('--add_phase_shift', dest='add_phase_shift', default= 0, type=int, help='add_phase_shift')
parser.add_argument('--verbose', dest='verbose', default= 0, type=int, help='verbose')
parser.add_argument('--lr', dest='lr', default= 0.0025, type=float, help='learning rate')
parser.add_argument('--do_eval', dest='do_eval', default= 0, type=int, help='verbose')
parser.add_argument('--iterations', dest='iterations', default= 20, type=int, help='iterations')
parser.add_argument('--years', dest='years', default= 30, type=int, help='years')
parser.add_argument('--weight_decay', dest='weight_decay', default= 0.00000000001, type=float, help='weight_decay')
parser.add_argument('--weight_decay_fre', dest='weight_decay_fre', default= 0.00000000001, type=float, help='weight_decay_fre')
parser.add_argument('--time_scale', dest='time_scale', default= 1, type=int, help='time_scale')
parser.add_argument('--min_count', dest='min_count', default= 200, type=int, help='min_count')
parser.add_argument('--window_size', dest='window_size', default= 5, type=int, help='window_size')
parser.add_argument('--dropout', dest='dropout', default= 0, type=float, help='dropout rate')
parser.add_argument('--fre_pattern', dest='fre_pattern', default= "1-10000", type=str, help='fre_pattern with base and the divided')
parser.add_argument('--save_step', dest='save_step', default= 100000000 , type=int, help='log_step')
parser.add_argument('--seed', dest='seed', default= 42 , type=int, help='seed')
parser.add_argument('--in_batch_negative', dest='in_batch_negative', default= 0 , type=int, help='in_batch_negative')

args = parser.parse_args()

if not  torch.cuda.is_available():
    args.verbose = 1


torch.manual_seed(args.seed)
import random
random.seed(args.seed)
np.random.seed(args.seed)


def save_dict( id2word, path):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, "vocab.txt"), 'w', encoding="utf-8") as f:
        for wid, w in id2word.items():
            f.write('{}\t{}\n'.format(wid, w))

class Word2VecTrainer:
    def __init__(self, args):# input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,initial_lr=0.01, min_count=25,weight_decay = 0, time_scale =1

        # self.data = DataReader(args.text, args.min_count)
        # if not args.use_time:
        #      dataset = Word2vecDataset(self.data, args.window_size)
        # else:
        #     dataset = TimestampledWord2vecDataset(self.data, args.window_size,args.time_scale)
        #
        # self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
        #                              shuffle=True, num_workers=0, collate_fn=dataset.collate)
        self.data,self.dataloader = self.load_train(args) # self.data

        if "train" in args.text:
            test_filename = args.text.replace("train","test")
            if  os.path.exists(test_filename):
                print("load test  dataset: ".format(test_filename))
                self.test = self.load_train(args, data = self.data, filename=test_filename, is_train=False )
            else:
                self.test = None

            dev_filename = args.text.replace("train", "dev")
            if  os.path.exists(dev_filename):
                print("load dev dataset: ".format(dev_filename))
                self.dev = self.load_train(args, data = self.data, filename=dev_filename, is_train=False)
            else:
                self.dev = None
        else:
            self.dev, self.test = None, None

        
        if args.use_time:
            self.output_file_name = "{}/{}".format(args.output, args.time_type)
            if args.add_phase_shift:
                self.output_file_name  += "_shift"
        else:
            self.output_file_name = "{}/{}".format(args.output, "word2vec")
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        if not os.path.exists(self.output_file_name):
            os.mkdir(self.output_file_name)
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.emb_dimension
        self.batch_size = args.batch_size
        self.iterations = args.iterations
        self.lr = args.lr
        self.time_type = args.time_type
        self.weight_decay = args.weight_decay

        print(args)


        if args.use_time:
            # self.skip_gram_model = TimestampedSkipGramModel(self.emb_size, self.emb_dimension,time_type = args.time_type,add_phase_shift=args.add_phase_shift,dropout =args.dropout,fre_pattern=args.fre_pattern,in_batch_negative = args.in_batch_negative)
            self.skip_gram_model = DE(self.emb_size, self.emb_dimension, time_type=args.time_type,dropout=args.dropout)
        else:
            self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            print("using cuda and GPU ....")
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = torch.nn.DataParallel(self.skip_gram_model)
            else:
                self.model = self.skip_gram_model
            self.model.cuda()
        else:
            self.model = self.skip_gram_model
        print(self.model)

        # load_path = "{}/{}".format(self.output_file_name)
        # torch.save(self.skip_gram_model,"pytorch.bin")
        # self.skip_gram_model =  torch.load("pytorch.bin")
        # self.skip_gram_model = load_model(self.skip_gram_model,"pytorch.bin")
        # exit()
        # if not args.from_scatch and os.path.exists(self.output_file_name):
        #
        #     print("loading parameters  ....")
        #     self.skip_gram_model.load_embeddings(self.data.id2word,self.output_file_name)

    def load_train(self,args,data= None, filename = None, is_train = True):
        if data is None:
            assert is_train==True, "wrong to load data 1"
            data = DataReader(args.text, args.min_count)
            filename = args.text
        else:
            assert is_train == False, "wrong to load test data 2"
            assert filename is not None, "wrong to load test data 3"
            assert data is not None, "wrong to load test data 4"
            print("load filenames as dev/test {}".format(filename))
        if not args.use_time:
            dataset = Word2vecDataset(data, input_text = filename, window_size= args.window_size)
        else:
            dataset = TimestampledWord2vecDataset(data,input_text = filename, window_size= args.window_size, time_scale=args.time_scale,in_batch_negative = args.in_batch_negative)
        print("load data length: {}".format(len(dataset)))
        # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        #     dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # else:
        #     dataset_sampler = torch.utils.data.RandomSampler(dataset)

        process_method = dataset.collate_in_batch_negative if args.in_batch_negative else  dataset.collate
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=is_train, num_workers=0, collate_fn=process_method) # shuffle if it is train

        if is_train:
            return data,dataloader
        else:
            return dataloader

    @log_time_delta
    def evaluation_loss(self,logger =None):
        results = []
        self.skip_gram_model.eval()
        print("evaluating ...")
        for index,dataloader in enumerate([self.dev,self.test]):
            if dataloader is None:
                continue
            losses = []

            for i, sample_batched in enumerate(tqdm(dataloader)):
                if len(sample_batched[0]) > 1:

                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    if args.use_time:
                        time = sample_batched[3].to(self.device)
                        # print(time)
                        loss, pos, neg = self.skip_gram_model.forward(pos_u, pos_v, neg_v, time)
                    else:

                        loss, pos, neg = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    # print(loss)
                    losses.append(loss.item())
            mean_result = np.array(losses).mean()
            results.append(mean_result)
            print("test{} loss is {}".format(index, mean_result))
            logger.write("Loss in  test{}: {} \n".format( index, str(mean_result)))
            logger.flush()

        self.skip_gram_model.train()
        return results

    def train(self):
        print(os.path.join(self.output_file_name,"log.txt"))
        if not os.path.exists(self.output_file_name):
            os.mkdir(self.output_file_name)
        print(self.model)

        if args.time_type =="word_mixed_amplitude":

            no_decay = ['para_embedding']
            # print([n for  n, p in param_optimizer])
            print("using small weight decay")
            print([n for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)])
            print("using big weight decay")
            print([n for n, p in self.model.named_parameters() if  any(nd in n for nd in no_decay)])

            #weight_decay_fre
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_fre}
            ]
            print(optimizer_grouped_parameters)
        else:
            optimizer_grouped_parameters = self.model.parameters()

        optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.lr,weight_decay=self.weight_decay) #,
        # optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr,weight_decay=self.weight_decay)

        # optimizer = optim.Adam(, lr=self.lr, weight_decay=self.weight_decay)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader)*self.iterations)
        # scheduler = get_linear_schedule_with_warmup(optimizer,0, len(self.dataloader)*self.iterations)

        save_dict(self.data.id2word,self.output_file_name)
        print(self.skip_gram_model)
        with open("{}/log.txt".format(self.output_file_name,"log.txt"),"w") as f:
            for iteration in range(self.iterations):

                print("\nIteration: " + str(iteration + 1))
                f.write(str(args) +"\n")
                # optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)

                # self.evaluation_loss(logger=f)

                running_loss = 0.0
                # if torch.cuda.is_available() and torch.cuda.device_count() >1:
                #     dataloader_sampler = torch.utils.data.distributed.DistributedSampler(self.dataloader)
                # else:
                dataloader_sampler = self.dataloader

                for i, sample_batched in enumerate(tqdm(dataloader_sampler)):

                    if len(sample_batched[0]) > 1:

                        pos_u = sample_batched[0].to(self.device)
                        pos_v = sample_batched[1].to(self.device)

                        neg_v = sample_batched[2].to(self.device) if not args.in_batch_negative else None

                        optimizer.zero_grad()
                        if args.use_time:
                            time = sample_batched[3].to(self.device)
                            # print(time)
                            loss,pos,neg = self.model.forward(pos_u, pos_v, neg_v,time)
                        else:

                            loss,pos,neg = self.model.forward(pos_u, pos_v, neg_v)
                        # print(loss)
                        if torch.cuda.device_count()>1:
                            loss,pos,neg = loss.mean(),pos.mean(),neg.mean()
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()



                        loss,pos,neg = loss.item(),pos.item(),neg.item()

                        if  i % args.log_step == 0: # i > 0 and
                            f.write("Loss in {} steps: {} {}, {}\n".format(i,str(loss),str(pos),str(neg)))

                        if  not torch.cuda.is_available() or i % (args.log_step*10) == 0 :
                            print("Loss in {} steps: {} {}, {}\n".format(i,str(loss),str(pos),str(neg)))
                        if (i+1) % args.save_step == 0 :
                            torch.save(self.model, os.path.join(self.output_file_name, "pytorch_{}_{}.bin".format(iteration,i)))

                self.evaluation_loss(logger=f)
                epoch_path = os.path.join(self.output_file_name,str(iteration))
                if not os.path.exists(epoch_path):
                    os.mkdir(epoch_path)

                torch.save(self.model, os.path.join( epoch_path,"pytorch.bin") )

                # self.skip_gram_model.save_embedding(self.data.id2word, os.path.join(self.output_file_name,str(iteration)))
                # self.skip_gram_model.save_in_text_format(self.data.id2word, os.path.join(self.output_file_name, str(iteration)))
            # self.skip_gram_model.save_in_text_format(self.data.id2word,self.output_file_name)


            torch.save(self.model, os.path.join(self.output_file_name,"pytorch.bin") )
            with open(os.path.join(self.output_file_name,"config.json"), "wt") as f:
                json.dump(vars(args), f, indent=4)
            save_dict(self.data.id2word,self.output_file_name)



if __name__ == '__main__':
    

    w2v = Word2VecTrainer(args)
    #input_file = args.text, output_file = args.output, batch_size = args.batch_size, initial_lr = args.lr, weight_decay = args.weight_decay, iterations = args.iterations, time_scale = args.time_scale
    w2v.train()


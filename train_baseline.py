import shutil
import argparse
import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, IterableDataset
import logging
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from evaluator_bioes import get_score_basleline as get_score
import random
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    pl.seed_everything(random_seed)
parser = argparse.ArgumentParser(description='Zero-Shot Cross-Lingual ABSA via Generation Task')


parser.add_argument('--lang',
                    type=str,
                    default='es',
                    help='target language')
    
    
parser.add_argument('--domain',
                    type=str,
                    default='res',
                    help='review domain')



parser.add_argument('--freeze',
                    type=bool,
                    default=True,
                    help='whether to freeze low layers of encoder')

parser.add_argument('--lr',
                    type=float,
                    default=2e-5,
                    help='learning rate')
    
parser.add_argument('--epochs',
                    type=int,
                    default=5,
                    help='training epcohs')
    
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.04,
                    help='weight decay')

parser.add_argument('--warm_up',
                    type=float,
                    default=0.1,
                    help='warm up')
    
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help='batch size')

args = parser.parse_args()
    
lang_code = {'es': 'es_XX' , 'en': 'en_XX', 'fr':'fr_XX' , 'nl':'nl_XX', 'ru':'ru_RU' ,'tu':'tr_TR'}

prompt_code = {'es':['<inicio>','<fin>'
                    ],
               'en':['<start>', '<end>'
                    ],
               'fr':['<début>' , '<fin>'
                    ],
               'nl':['<begin>', '<ein>'
                    ],
               'ru':[ '<начало>','<конец>'
                    ]}

verbal_code = {'es':['positivo', 'negativo' , 'neutral'],
               'en':['positive','negative','neutral'],
               'fr':['positif', 'négatif', 'neutre'],
               'nl':['positief', 'negatief', 'neutraal'],
               'ru':[ 'положительный', 'отрицательный', 'нейтральный']}
               
sep_code = {'es':'y',
               'en':'and',
               'fr':'et',
               'nl':'en',
               'ru':'и'}

               

class ReviewDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, pad_index = 1, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t')
        self.docs.dropna(inplace=True)
        self.docs.reset_index(inplace=True,drop=True)
        self.len = len(self.docs)
        self.pad_index = pad_index
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    def get_pair(self, instance, pol_list, sep_code):
        pair=[]
        for aspect, polarity in zip(instance['target'].split('<&>'), instance['polarity'].split('<&>')):
            if polarity=='positive':
                pair.append(aspect + ' ' + pol_list[0])
            elif polarity=='negative':
                pair.append(aspect + ' ' + pol_list[1])
            else:
                pair.append(aspect + ' ' + pol_list[2])
                
        pair = f' {sep_code} '.join(pair)
        return pair.strip()
    def get_pair_trans(self, instance, pol_list, sep_code):
        pair=[]
        for aspect, polarity in zip(instance['target_trans'].split('<&>'), instance['polarity'].split('<&>')):
            if polarity=='positive':
                pair.append(aspect + ' ' + pol_list[0])
            elif polarity=='negative':
                pair.append(aspect + ' ' + pol_list[1])
            else:
                pair.append(aspect + ' ' + pol_list[2])
                
        pair = f' {sep_code} '.join(pair)
        return pair.strip()
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['review']+' </s> ')
        input_ids = self.add_padding_data(input_ids)
        
        pair = self.get_pair(instance, verbal_code['en'], sep_code['en'])
        
#        pair = instance['target'].replace('<&>', sep_code['en'])
        target = prompt_code['en'][0] + ' ' + pair + ' ' + prompt_code['en'][1]

        label_ids = self.tokenizer.encode(target)
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.pad_index]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)
        
        with tokenizer.as_target_tokenizer():
            input_ids_trg = self.tokenizer.encode(instance['review']+' </s> ')
            input_ids_trg = self.add_padding_data(input_ids_trg)
            
#            pair_trg = instance['target_trans'].replace('<&>', sep_code[args.lang])
            # without template, so verbalizer is in english
            pair_trg = self.get_pair_trans(instance, verbal_code['en'], sep_code['en'])
            target_trg = prompt_code[args.lang][0] + ' ' + pair_trg + ' ' + prompt_code[args.lang][1]
            
            label_ids_trg = self.tokenizer.encode(target_trg)
            label_ids_trg.append(self.tokenizer.eos_token_id)
            dec_input_ids_trg = [self.pad_index]
            dec_input_ids_trg += label_ids_trg[:-1]
            dec_input_ids_trg = self.add_padding_data(dec_input_ids_trg)
            label_ids_trg = self.add_ignored_data(label_ids_trg)
        
        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_),
               
                'input_ids_trg': np.array(input_ids_trg, dtype=np.int_),
                'decoder_input_ids_trg': np.array(dec_input_ids_trg, dtype=np.int_),
                'labels_trg': np.array(label_ids_trg, dtype=np.int_)}

    def __len__(self):
        return self.len

class ReviewDataModule(pl.LightningDataModule):
    def __init__(self, train_file,
                 test_file, tokenizer,
                 max_len=128,
                 batch_size=args.batch_size,
                 num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.train = ReviewDataset(self.train_file_path,
                                 self.tokenizer,
                                 self.max_len)
        self.test = ReviewDataset(self.test_file_path,
                                self.tokenizer,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.train,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.test,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val


class Base(pl.LightningModule):
    def __init__(self):
        super(Base, self).__init__()

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay}, # 0.04
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, correct_bias=False) #1e-5
        # warm up lr
        num_workers = 4
        data_len = len(data_module.train)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (args.batch_size * num_workers) * args.epochs) # batch size, max epochs
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * args.warm_up) # warm up ratio
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class PairGenerator(Base):
    def __init__(self):
#         super(self).__init__()
        super().__init__()
        self.tokenizer = tokenizer
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
        self.model.resize_token_embeddings(len(tokenizer))
        if args.freeze == True:
            self.model = self.freezing(self.model)
        self.model.train()
        self.pad_token_id = 1
        
    def freezing(self, model):
        for name, param in model.named_parameters():
            if name.startswith("model.shared"):
                param.requires_grad = False
            if name.startswith("model.encoder.layers.0"):
                param.requires_grad = False
            if name.startswith("model.encoder.layers.1"):
                param.requires_grad = False
            if name.startswith("model.encoder.layers.2"):
                param.requires_grad = False
        print('lower freezed')
        return model
    def forward(self, inputs):
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        attention_mask_trg = inputs['input_ids_trg'].ne(self.pad_token_id).float()
        decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask_trg = inputs['decoder_input_ids_trg'].ne(self.pad_token_id).float()
        
        output=self.model(input_ids=inputs['input_ids'],
                          attention_mask=attention_mask,
                          decoder_input_ids=inputs['decoder_input_ids'],
                          decoder_attention_mask=decoder_attention_mask,
                          labels=inputs['labels'], return_dict=True)
        
        output_trg=self.model(input_ids=inputs['input_ids_trg'],
                          attention_mask=attention_mask_trg,
                          decoder_input_ids=inputs['decoder_input_ids_trg'],
                          decoder_attention_mask=decoder_attention_mask_trg,
                          labels=inputs['labels_trg'], return_dict=True)
        
        loss = output.loss + output_trg.loss
        
        
        return loss/2


    def training_step(self, batch, batch_idx):
        outs = self(batch)
        self.log('train_loss', outs, prog_bar=True)
        return outs

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        return (outs)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)


        
if __name__ == '__main__':
    
    
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang=lang_code[args.lang],additional_special_tokens=[prompt_code['en'][0],prompt_code['en'][1],prompt_code[args.lang][0], prompt_code[args.lang][1]])
    data_module=ReviewDataModule(f'datasets/{args.domain}_en_train_in_{args.lang}.tsv',                                                                    f'datasets/{args.domain}_en_validation_in_{args.lang}.tsv',
                                 tokenizer)
    seed_everything(42)
    model = PairGenerator()
    save_model_path = f'model_{args.lang}/{args.domain}_{args.lang}_lr={args.lr}_bs={args.batch_size}_frz={args.freeze}_baseline'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=save_model_path,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
#                                                        save_last=True,
                                                       mode='min',
                                                       save_top_k=1)
    lr_logger = pl.callbacks.LearningRateMonitor()
    tb_logger = pl_loggers.TensorBoardLogger(save_model_path, 'tb_logs')
    trainer = pl.Trainer(logger=tb_logger, callbacks=[checkpoint_callback, lr_logger],
                    max_epochs=args.epochs,gpus=1, progress_bar_refresh_rate=30)
    trainer.fit(model, data_module)
    
    best_checkpoint = glob.glob(save_model_path+'/model_chp/*')
    model=model.load_from_checkpoint(best_checkpoint[-1])
    model.model.save_pretrained(save_model_path+'/model')
    tokenizer.save_pretrained(save_model_path+'/model')
    ## zero-shot
    test_df = pd.read_csv(f'datasets/{args.domain}_{args.lang}_test.tsv',sep='\t')
    lang = args.lang
    
    get_score(test_df, save_model_path+'/model', lang)

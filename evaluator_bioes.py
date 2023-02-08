import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
from glob import glob
from sklearn.metrics import precision_recall_fscore_support , f1_score
from seqeval import metrics as seqeval_metrics
from sklearn.metrics import accuracy_score, recall_score, precision_recall_fscore_support
from seqeval.scheme import IOBES

import argparse
import json

lang_code = {'es': 'es_XX' , 'en': 'en_XX', 'fr':'fr_XX' , 'nl':'nl_XX', 'ru':'ru_RU' }


verbal_code_direct = {'es':['bueno', 'malo' , 'bien'],
               'en':['good','bad','okay'],
               'fr':['bon' , 'mauvais', 'bien'],
               'nl':['goed',  'slecht', 'oké'],
               'ru':[ 'хороша','плохой', 'порядке']}
               
verbal_code_indirect = {'es':['positivo', 'negativo' , 'neutral'],
               'en':['positive','negative','neutral'],
               'fr':['positif', 'négatif', 'neutre'],
               'nl':['positief', 'negatief', 'neutraal'],
               'ru':[ 'положительный', 'отрицательный', 'нейтральный']}

sep_code = {'es':'y',
               'en':'and',
               'fr':'et',
               'nl':'en',
               'ru':'и'}


special_code = ['<aspect>','</aspect>','<polarity>','</polarity>']
def check_special_code(gen):
    for s in special_code:
        if s not in gen:
            return False
    return True
    
def get_score_direct(df, model_name, lang, time, num):
    
    
    if time =='past' and num=='single':
        prompt_code = {'es':['<inicio>','<fin>',
                            'Sentí que <aspect> <mask> </aspect> era <polarity> <mask> </polarity>.'
                            ],
                       'en':['<start>', '<end>',
                            'I felt <aspect> <mask> </aspect> was <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            'Je sentais que <aspect> <mask> </aspect> était <polarity> <mask> </polarity>.'],
                       'nl':['<begin>', '<ein>',
                            'Ik voelde dat <aspect> <mask> </aspect> <polarity> <mask> </polarity> was.'],
                       'ru':[ '<начало>','<конец>',
                            'Я чувствовал, что <aspect> <mask> </aspect> была <polarity> <mask> </polarity>.']}

    elif time=='present' and num=='single':
         prompt_code = {'es':['<inicio>','<fin>',
                            'Siento que <aspect> <mask> </aspect> es <polarity> <mask> </polarity>.'
                            ],
                       'en':['<start>', '<end>',
                            'I feel <aspect> <mask> </aspect> is <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            "J'ai l'impression que <aspect> <mask> </aspect> est <polarity> <mask> </polarity>."],
                       'nl':['<begin>', '<ein>',
                            'Ik voel dat <aspect> <mask> </aspect> <polarity> <mask> </polarity> is.'],
                       'ru':[ '<начало>','<конец>',
                            'Я чувствую, что <aspect> <mask> </aspect> это <polarity> <mask> </polarity>.']}

    elif time=='past' and num=='multiple':
         prompt_code = {'es':['<inicio>','<fin>',
                            'Sentí que <aspect> <mask> </aspect> eran <polarity> <mask> </polarity>.'
                            ],
                       'en':['<start>', '<end>',
                            'I felt <aspect> <mask> </aspect> were <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            "Je sentais que <aspect> <mask> </aspect> étaient <polarity> <mask> </polarity>."],
                       'nl':['<begin>', '<ein>',
                            'Ik voelde dat <aspect> <mask> </aspect> <polarity> <mask> </polarity> waren.'],
                       'ru':[ '<начало>','<конец>',
                            'Я чувствовал, что <aspect> <mask> </aspect> был <polarity> <mask> </polarity>.']}

    elif time=='present' and num=='multiple':
         prompt_code = {'es':['<inicio>','<fin>',
                            'Siento que <aspect> <mask> </aspect> son <polarity> <mask> </polarity>.'
                            ],
                       'en':['<start>', '<end>',
                            'I feel <aspect> <mask> </aspect> are <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            "J'ai l'impression que <aspect> <mask> </aspect> sont <polarity> <mask> </polarity>."],
                       'nl':['<begin>', '<ein>',
                            'Ik vind dat <aspect> <mask> </aspect> <polarity> <mask> </polarity> zijn.'],
                       'ru':[ '<начало>','<конец>',
                            'Я чувствую, что <aspect> <mask> </aspect> - это <polarity> <mask> </polarity>.']}


    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)
    spltr = sep_code[lang]
    prompt = prompt_code[lang][2]
    good, bad, okay = verbal_code_direct[lang][0], verbal_code_direct[lang][1], verbal_code_direct[lang][2]
    start=250056
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.to('cuda')
    model.eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    print(model_name)

    gen=[]
    for sent in tqdm(df['review']):
        with tokenizer.as_target_tokenizer():
                model_inputs = tokenizer.encode(sent + ' </s> ' + prompt, return_tensors="pt")
                output = tokenizer.decode(model.generate(model_inputs.to('cuda'),
                               forced_bos_token_id=start)[0])
            
            
        gen.append(output.strip().replace('  ',' '))
        
    df['generated']=gen
    tags = []
    for sent , gens in zip(df['review'],df['generated']):
                if lang == 'ru':
                    gens = gens.replace('<аspect>', '<aspect>')
                    gens = gens.replace('</аspect>', '</aspect>')
                    
                bio = ['O' for _ in range(len(sent.split()))]
                if check_special_code(gens) == False: # error control
                    tags.append(' '.join(bio))
                else:
                    boa, eoa = gens.index('<aspect>'), gens.index('</aspect>')
                    bop, eop = gens.index('<polarity>'), gens.index('</polarity>')
                    cands = []
                    a, b = gens[boa+8:eoa].strip() , gens[bop+10:eop].strip()
                    nt=a.strip().split(' {} '.format(spltr))
                    np=b.strip().split(' {} '.format(spltr))
                    for tt , pp in zip(nt, np):
                                p=pp.replace('.','').strip()
                                p=p.replace('good','positive')
                                p=p.replace('bad','negative')
                                p=p.replace('okay','neutral')
                                p=p.replace(good,'positive')
                                p=p.replace(bad,'negative')
                                p=p.replace(okay,'neutral')
                                cands.append((tt.strip(), p.strip()))
    #                 print(cands)
                    bio = ['O' for _ in range(len(sent.split()))]
                    for i, word in enumerate(sent.split()):
                                for c in cands:
                                    t, p = c[0].strip() , c[1].strip()
                                    if len(t.split()) == 1 and t in word:
                                        if p in ('positive', 'negative', 'neutral'):
                                            bio[i] = 'S-{}'.format(p)
                                    else:
                                        for j, nt in enumerate(t.split()):
                                            if nt in word:
                                                if p in ('positive', 'negative', 'neutral'):
                                                    if j == 0:
                                                        bio[i] = 'B-{}'.format(p)
                                                    elif j == len(t.split())-1:
                                                        bio[i] = 'E-{}'.format(p)
                                                    else:
                                                        bio[i] = 'I-{}'.format(p)


                    tags.append(' '.join(bio))


           
                                                      
    df['preds'] = tags
    preds =[t.split() for t in df['preds'].tolist()]
    labels =[t.split() for t in df['tags'].tolist()]

    res=seqeval_metrics.classification_report(labels, preds, output_dict=True, scheme=IOBES )
    res_float={k:{k_2:float(v_2) for k_2, v_2 in v_dict.items()} for k, v_dict in res.items()}
    print(res['micro avg']['f1-score'])
    df[['review','target','tags','generated', 'preds']].to_csv('/'.join(model_name.split('/')[:2]) + '/inference_result_bieos_new.tsv', index=False, sep='\t')
    with open('/'.join(model_name.split('/')[:2]) + '/result_bieos_new.json', 'w') as f:
        json.dump(res_float,f)
    f.close()

def get_score_indirect(df, model_name, lang, time,num):
    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)
    spltr = sep_code[lang]
    
    if time =='past' and num=='single':
        prompt_code = {'es':['<inicio>','<fin>',
                            'El sentimiento de <aspect> <mask> </aspect> era <polarity> <mask> </polarity>.'],
                       'en':['<start>', '<end>',
                            'The feeling of <aspect> <mask> </aspect> was <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            'Le sentiment de <aspect> <mask> </aspect> était <polarity> <mask> </polarity>.'],
                       'nl':['<begin>', '<ein>',
                            'Het gevoel van <aspect> <mask> </aspect> was <polarity> <mask> </polarity>.'],
                       'ru':[ '<начало>','<конец>',
                            'Ощущение <aspect> <mask> </аspect> было <polarity> <mask> </polarity>.']}

    elif time=='present' and num=='single':
         prompt_code = {'es':['<inicio>','<fin>',
                            'El sentimiento de <aspect> <mask> </aspect> es <polarity> <mask> </polarity>.'],
                       'en':['<start>', '<end>',
                            'The feeling of <aspect> <mask> </aspect> is <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            "Le sentiment de <aspect> <mask> </aspect> est <polarity> <mask> </polarity>."],
                       'nl':['<begin>', '<ein>',
                            'Het gevoel van <aspect> <mask> </aspect> is <polarity> <mask> </polarity>.'],
                       'ru':[ '<начало>','<конец>',
                            'Ощущение <aspect> <mask> </aspect> есть <polarity> <mask> </polarity>.']}

    elif time=='past' and num=='multiple':
         prompt_code = {'es':['<inicio>','<fin>',
                            'Los sentimientos de <aspect> <mask> </aspect> eran <polarity> <mask> </polarity>.'
                            ],
                       'en':['<start>', '<end>',
                            'The feelings <aspect> <mask> </aspect> were <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            "Je sentais que <aspect> <mask> </aspect> étaient <polarity> <mask> </polarity>."],
                       'nl':['<begin>', '<ein>',
                            'De gevoelens <aspect> <mask> </aspect> waren <polarity> <mask> </polarity>.'],
                       'ru':[ '<начало>','<конец>',
                            'Чувства <aspect> <mask> </aspect> были <polarity> <mask> </polarity>.']}

    elif time=='present' and num=='multiple':
         prompt_code = {'es':['<inicio>','<fin>',
                            'Los sentimientos <aspect> <mask> </aspect> son <polarity> <mask> </polarity>.'],
                       'en':['<start>', '<end>',
                            'The feelings of <aspect> <mask> </aspect> are <polarity> <mask> </polarity>.'],
                       'fr':['<début>' , '<fin>',
                            "Les sentiments <aspect> <mask> </aspect> sont <polarity> <mask> </polarity>."],
                       'nl':['<begin>', '<ein>',
                            'De gevoelens van <aspect> <mask> </aspect> zijn <polarity> <mask> </polarity>.'],
                       'ru':[ '<начало>','<конец>',
                            'Чувства <aspect> <mask> </aspect> есть <polarity> <mask> </polarity>.']}


    
    
    prompt = prompt_code[lang][2]
    good, bad, okay = verbal_code_indirect[lang][0], verbal_code_indirect[lang][1], verbal_code_indirect[lang][2]
    start=250056
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.to('cuda')
    model.eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    print(model_name)

    gen=[]
    for sent in tqdm(df['review']):
        with tokenizer.as_target_tokenizer():
                model_inputs = tokenizer.encode(sent + ' </s> ' + prompt, return_tensors="pt")
                output = tokenizer.decode(model.generate(model_inputs.to('cuda'),
                               forced_bos_token_id=start)[0])
            
            
        gen.append(output.strip().replace('  ',' '))
        
    df['generated']=gen
    tags = []
    for sent , gens in zip(df['review'],df['generated']):
                bio = ['O' for _ in range(len(sent.split()))]
                if check_special_code(gens) == False: # error control
                    tags.append(' '.join(bio))
                else:
                    boa, eoa = gens.index('<aspect>'), gens.index('</aspect>')
                    bop, eop = gens.index('<polarity>'), gens.index('</polarity>')
                    cands = []
                    a, b = gens[boa+8:eoa].strip() , gens[bop+10:eop].strip()
                    nt=a.strip().split(' {} '.format(spltr))
                    np=b.strip().split(' {} '.format(spltr))
                    for tt , pp in zip(nt, np):
                                p=pp.replace('.','').strip()
                                p=p.replace('good','positive')
                                p=p.replace('bad','negative')
                                p=p.replace('okay','neutral')
                                p=p.replace(good,'positive')
                                p=p.replace(bad,'negative')
                                p=p.replace(okay,'neutral')
                                cands.append((tt.strip(), p.strip()))
    #                 print(cands)
                    bio = ['O' for _ in range(len(sent.split()))]
                    for i, word in enumerate(sent.split()):
                                for c in cands:
                                    t, p = c[0].strip() , c[1].strip()
                                    if len(t.split()) == 1 and t in word:
                                        if p in ('positive', 'negative', 'neutral'):
                                            bio[i] = 'S-{}'.format(p)
                                    else:
                                        for j, nt in enumerate(t.split()):
                                            if nt in word:
                                                if p in ('positive', 'negative', 'neutral'):
                                                    if j == 0:
                                                        bio[i] = 'B-{}'.format(p)
                                                    elif j == len(t.split())-1:
                                                        bio[i] = 'E-{}'.format(p)
                                                    else:
                                                        bio[i] = 'I-{}'.format(p)


                    tags.append(' '.join(bio))



    df['preds'] = tags
    preds =[t.split() for t in df['preds'].tolist()]
    labels =[t.split() for t in df['tags'].tolist()]

    res=seqeval_metrics.classification_report(labels, preds, output_dict=True, scheme=IOBES )
    res_float={k:{k_2:float(v_2) for k_2, v_2 in v_dict.items()} for k, v_dict in res.items()}
    print(res['micro avg']['f1-score'])
    df[['review','target','tags','generated', 'preds']].to_csv('/'.join(model_name.split('/')[:2]) + '/inference_result_bieos_new.tsv', index=False, sep='\t')
    with open('/'.join(model_name.split('/')[:2]) + '/result_bieos.json', 'w') as f:
        json.dump(res_float,f)
    f.close()
    

    
def get_score_basleline(df, model_name, lang):
    

    prompt_code = {'es':['<inicio>','<fin>'],
               'en':['<start>', '<end>'],
               'fr':['<début>' , '<fin>'],
               'nl':['<begin>', '<ein>'],
               'ru':[ '<начало>','<конец>']}
    def check_gen(out):
        for code in prompt_code[lang]:
            if code not in out:
                return False
        return True
    
    
    df.dropna(inplace=True)
    df.reset_index(inplace=True,drop=True)
    spltr = sep_code[lang]
    start=250056
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    model.to('cuda')
    model.eval()
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    print(model_name)

    gen=[]
    for sent in tqdm(df['review']):
        with tokenizer.as_target_tokenizer():
                model_inputs = tokenizer.encode(sent + ' </s> ', return_tensors="pt")
                output = tokenizer.decode(model.generate(model_inputs.to('cuda'),
                               forced_bos_token_id=start)[0])
            
            
        gen.append(output.strip().replace('  ',' '))
        
    df['generated']=gen
    tags = []
    for sent , gens in zip(df['review'],df['generated']):
                    bio = ['O' for _ in range(len(sent.split()))]
                    if check_gen == False:
                        tags.append(' '.join(bio))
                    else:
                        gens = gens.replace('</s>','')
                        gens = gens.replace(prompt_code[lang][0],'')
                        gens = gens.replace(prompt_code[lang][1],'')
                        gens_splt = gens.strip().split(' {} '.format(spltr))
                        gens_splt = [g for g in gens_splt if g!=''] 
                        if gens_splt == []:
                            tags.append(' '.join(bio))
                        else:
                            cands = []

                            nt=[' '.join(x.strip().split()[:-1]) for x in gens_splt]
                            np=[x.strip().split()[-1] for x in gens_splt]
                            for tt , pp in zip(nt, np):
                                        cands.append((tt.strip(), pp.strip()))
            #                 print(cands)
                            for i, word in enumerate(sent.split()):
                                        for c in cands:
                                            t, p = c[0].strip() , c[1].strip()
                                            if len(t.split()) == 1 and t in word:
                                                if p in ('positive', 'negative', 'neutral'):
                                                    bio[i] = 'S-{}'.format(p)
                                            else:
                                                for j, nt in enumerate(t.split()):
                                                    if nt in word:
                                                        if p in ('positive', 'negative', 'neutral'):
                                                            if j == 0:
                                                                bio[i] = 'B-{}'.format(p)
                                                            elif j == len(t.split())-1:
                                                                bio[i] = 'E-{}'.format(p)
                                                            else:
                                                                bio[i] = 'I-{}'.format(p)


                            tags.append(' '.join(bio))



    df['preds'] = tags
    preds =[t.split() for t in df['preds'].tolist()]
    labels =[t.split() for t in df['tags'].tolist()]

    res=seqeval_metrics.classification_report(labels, preds, output_dict=True, scheme=IOBES )
    res_float={k:{k_2:float(v_2) for k_2, v_2 in v_dict.items()} for k, v_dict in res.items()}
    print(res['micro avg']['f1-score'])
    df[['review','target','tags','generated', 'preds']].to_csv('/'.join(model_name.split('/')[:2]) + '/inference_result_bieos_new.tsv', index=False, sep='\t')
    with open('/'.join(model_name.split('/')[:2]) + '/result_bieos.json', 'w') as f:
        json.dump(res_float,f)
    f.close()
    

    

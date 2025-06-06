from torch.utils.data import Dataset
import pandas as pd
import re
from typing import List
from random import shuffle
from sklearn.model_selection import train_test_split

class CompletionDataset:
    def __init__(self,
                data: pd.DataFrame,
                prompt_layout: str,
                prompt_tags: str,
                tokenizer,
                n_icl_samples: int =  3, 
                clean: bool = True):
        self.data = data
        self.prompt_layout = prompt_layout
        self.prompt_tags = prompt_tags
        self.tokenizer = tokenizer
        self.n_icl_samples = n_icl_samples
        if clean:
            self.clean()
        self.train, self.dev = train_test_split(self.data, test_size=0.2)
        self.train = self.train.reset_index()
        self.dev, self.test = train_test_split(self.dev, test_size=0.5)
        self.dev = self.dev.reset_index()
        self.test = self.test.reset_index()
        self.train = self.make_samples('train')
        self.dev = self.make_samples('dev')
        self.test = self.make_samples('test')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def make_samples(self, split = 'train'):
        split_data = getattr(self, split)
        text_og_list = split_data['text_og']
        text_an_list = split_data['text_an']
        sample_list = []
        for i in range(len(text_og_list)):
            sentence = text_og_list.iloc[i]
            examples_an = text_an_list.drop(index=i)
            mask_pos = examples_an.apply(lambda x: '</' in x)
            mask_neg = examples_an.apply(lambda x: '</' not in x)
            examples_an_pos = examples_an[mask_pos].sample(n=self.n_icl_samples)
            examples_an_neg = examples_an[mask_neg].sample(n=self.n_icl_samples)
            examples_og_pos = text_og_list[examples_an_pos.index]
            examples_og_neg = text_og_list[examples_an_neg.index]
            examples_pos = [f'###{og} ==> {an}' for og, an in zip(examples_an_pos, examples_og_pos)]
            examples_neg = [f'###{og} ==> {an}' for og, an in zip(examples_an_neg, examples_og_neg)]
            # examples_pos = [f'###{og} ==> {self.tokenizer.bos_token}{an}{self.tokenizer.eos_token}' for og, an in zip(examples_an_pos, examples_og_pos)]
            # examples_neg = [f'###{og} ==> {self.tokenizer.bos_token}{an}{self.tokenizer.eos_token}' for og, an in zip(examples_an_neg, examples_og_neg)]
            examples = examples_pos + examples_neg
            shuffle(examples)
            examples = '\n'.join(examples)
            prompt = self.prompt_layout.format(prompt_tags=self.prompt_tags,
                                               examples=examples,
                                               sentence=sentence,
                                               eos_token=self.tokenizer.eos_token)
            sample = {
                'prompt': prompt,
                'completion': text_an_list.iloc[i],
            }
            sample_list.append(sample)
        return sample_list

    def clean(self):
        self.data['text_og'] = self.data['text_og'].apply(lambda x: x.replace(r'\0', ''))

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def make_chat(self, sample):
        chat = [{"role": "user", "content": f'{sample}'},]
        return self.tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt = True)

    def collate(self, batch):
        text_an_tok = self.tokenizer([self.make_chat(el['text_an']) for el in batch],
                return_tensors='pt',
                padding='longest',
                truncation=True)
        text_og_tok = self.tokenizer([self.make_chat(el['text_og']) for el in batch],
                return_tensors='pt',
                padding='longest',
                truncation=True)
        return {
            'input_ids': text_og_tok['input_ids'],
            'attention_mask': text_og_tok['attention_mask'],
        }

def make_tags_prompt(tags_csv_path: str):
    df = pd.read_csv(tags_csv_path, sep='\t')
    tags = df['Tag'].tolist()
    description = df['Description'].tolist()
    out_prompt = ''
    for t, d in zip(tags, description):
        out_prompt += f'{t}: {d}\n'
    return out_prompt

class CausalLMDataset(Dataset):
    def __init__(self,
                data: pd.DataFrame,
                prompt_layout: str,
                prompt_tags: str,
                tokenizer,
                n_icl_samples: int =  3, 
                clean: bool = True):
        self.data = data
        self.prompt_layout = prompt_layout
        self.prompt_tags = prompt_tags
        self.tokenizer = tokenizer
        self.n_icl_samples = n_icl_samples
        if clean:
            self.clean()
        self.make_samples()
        self.data = self.data.to_dict(orient='records')
        self.train, self.dev = train_test_split(self.data, test_size=0.2)
        self.dev, self.test = train_test_split(self.dev, test_size=0.5)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def make_samples(self):
        text_og_list = self.data['text_og']
        text_an_list = self.data['text_an']
        sample_list = []
        for i in range(len(text_og_list)):
            sentence = text_og_list.iloc[i]
            examples_an = text_an_list.drop(index=i)
            mask_pos = examples_an.apply(lambda x: '</' in x)
            mask_neg = examples_an.apply(lambda x: '</' not in x)
            examples_an_pos = examples_an[mask_pos].sample(n=self.n_icl_samples)
            examples_an_neg = examples_an[mask_neg].sample(n=self.n_icl_samples)
            examples_og_pos = text_og_list[examples_an_pos.index]
            examples_og_neg = text_og_list[examples_an_neg.index]
            examples_pos = [f'###{og} ==> {self.tokenizer.bos_token}{an}{self.tokenizer.eos_token}' for og, an in zip(examples_an_pos, examples_og_pos)]
            examples_neg = [f'###{og} ==> {self.tokenizer.bos_token}{an}{self.tokenizer.eos_token}' for og, an in zip(examples_an_neg, examples_og_neg)]
            examples = examples_pos + examples_neg
            shuffle(examples)
            examples = '\n'.join(examples)
            prompt = self.prompt_layout.format(prompt_tags=self.prompt_tags,
                                               examples=examples,
                                               sentence=sentence,
                                               eos_token=self.tokenizer.eos_token)
            sample = {
                'prompt': prompt,
                'completion': text_an_list.iloc[i],
            }
            sample_list.append(sample)
        self.data['samples'] = sample_list

    def clean(self):
        self.data['text_og'] = self.data['text_og'].apply(lambda x: x.replace(r'\0', ''))
    
    
    
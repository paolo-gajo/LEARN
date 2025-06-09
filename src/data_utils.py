import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split

class CausalLMDataset:
    def __init__(self,
                data: pd.DataFrame,
                prompt_layout: str,
                prompt_tags: str,
                tokenizer,
                n_icl_samples: int=3, 
                clean: bool=True,
                ):
        self.data = data
        self.prompt_layout = prompt_layout
        self.prompt_tags = prompt_tags
        self.tokenizer = tokenizer
        self.n_icl_samples = n_icl_samples
        if clean:
            self.clean()
        self.sys_prompt = 'You are an AI specialized in the task of annotating grammatical errors.'
        df_train, df_dev = train_test_split(self.data, test_size=0.2, random_state=42)
        self.train = df_train.reset_index(drop=True)
        df_dev, df_test = train_test_split(df_dev, test_size=0.5, random_state=42)
        self.dev = df_dev.reset_index(drop=True)
        self.test = df_test.reset_index(drop=True)
        self.train_samples = self.make_samples('train')
        self.dev_samples = self.make_samples('dev')
        self.test_samples = self.make_samples('test')

    def clean(self):
        self.data['text_og'] = self.data['text_og'].apply(lambda x: x.replace(r'\0', ''))
    
    def make_samples(self, split):
        split_data = getattr(self, split)
        text_og_list = split_data['text_og']
        text_an_list = split_data['text_an']
        text_og_list_examples = self.train['text_og']
        text_an_list_examples = self.train['text_an']
        chat_list = []
        for i in range(len(text_og_list)):
            sentence = text_og_list.iloc[i]
            expected_output = text_an_list.iloc[i]
            
            if split == 'train':
                examples_an = text_an_list_examples.drop(labels=split_data.index[i])
            else:
                examples_an = text_an_list_examples
            mask_pos = examples_an.apply(lambda x: '</' in x)
            mask_neg = examples_an.apply(lambda x: '</' not in x)
            examples_an_pos = examples_an[mask_pos].sample(n=self.n_icl_samples)
            examples_an_neg = examples_an[mask_neg].sample(n=self.n_icl_samples)
            examples_og_pos = text_og_list_examples[examples_an_pos.index]
            examples_og_neg = text_og_list_examples[examples_an_neg.index]
            examples_pos = [f'{og}###{an}' for og, an in zip(examples_og_pos, examples_an_pos)]
            examples_neg = [f'{og}###{an}' for og, an in zip(examples_og_neg, examples_an_neg)]
            examples = examples_pos + examples_neg
            shuffle(examples)
            examples = '\n'.join(examples)
            prompt = self.prompt_layout.format(prompt_tags=self.prompt_tags,
                                               examples=examples,
                                               sentence=sentence,
                                               eos_token=self.tokenizer.eos_token)
            
            # Create the prompt part (same for train and eval)
            chat_prompt = [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": prompt},
            ]
            
            if split == 'train':
                # For training: include the assistant response
                chat = chat_prompt + [{"role": "assistant", "content": expected_output}]
                chat_formatted = self.tokenizer.apply_chat_template(chat,
                                                        tokenize=False,
                                                        add_generation_prompt=False)
                chat_list.append({'text': chat_formatted})
            else:
                # For evaluation: separate prompt and expected output
                chat_prompt_formatted = self.tokenizer.apply_chat_template(chat_prompt,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
                chat_list.append({
                    'prompt': chat_prompt_formatted,
                    'completion': expected_output,
                    'prompt_length': len(self.tokenizer.encode(chat_prompt_formatted))
                })
        return pd.DataFrame(chat_list)  
    
def make_tags_prompt(tags_csv_path: str):
    df = pd.read_csv(tags_csv_path, sep='\t')
    tags = df['Tag'].tolist()
    description = df['Description'].tolist()
    out_prompt = ''
    for t, d in zip(tags, description):
        out_prompt += f'{t}: {d}\n'
    return out_prompt
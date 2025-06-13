import argparse
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
import re
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from typing import List

debug_mode = 1

class CausalLMDataset:
    def __init__(self,
                data: pd.DataFrame,
                prompt_layout: str,
                prompt_tags: str,
                tokenizer,
                config: dict,
                clean: bool=True,
                ):
        self.data = data
        self.prompt_layout = prompt_layout
        self.tokenizer = tokenizer
        self.prompt_tags = prompt_tags
        self.sys_prompt = 'You are an AI specialized in the task of annotating grammatical errors.'
        self.prompt_tags_preamble = 'The following are the tags you should use for annotation:'
        self.examples_preamble = 'Below are reference examples:'
        self.use_prompt_tags = config['use_prompt_tags']
        self.n_icl_samples = config['n_icl_samples']
        self.tag_dict = config['tag_dict']
        self.coarse = config['coarse']
        if clean:
            self.clean()
        df_train, df_dev = train_test_split(self.data, test_size=0.2, random_state=config['seed'])
        self.train = df_train.reset_index(drop=True)
        df_dev, df_test = train_test_split(df_dev, test_size=0.5, random_state=config['seed'])
        self.dev = df_dev.reset_index(drop=True)
        self.test = df_test.reset_index(drop=True)
        if self.coarse:
            self.train['text_an'] = self.train['text_an'].apply(lambda x: self.coarsen_tags(x))
            self.dev['text_an'] = self.dev['text_an'].apply(lambda x: self.coarsen_tags(x))
            self.test['text_an'] = self.test['text_an'].apply(lambda x: self.coarsen_tags(x))
        self.train_samples = self.make_samples('train')
        self.dev_samples = self.make_samples('dev')
        self.test_samples = self.make_samples('test')
        train_steps = config['train_steps']
        eval_steps = config['eval_steps']
        if train_steps:
            self.train_samples = self.train_samples[:train_steps]
        if eval_steps:
            self.dev_samples = self.dev_samples[:eval_steps]
            self.test_samples = self.test_samples[:eval_steps]

    def clean(self):
        self.data['text_og'] = self.data['text_og'].apply(lambda x: x.replace(r'\0', ''))

    def coarsen_tags(self, input: str):
        tags = set([el[0] for el in extract_tags(input)])
        for tag in tags:
            input = input.replace(tag, self.tag_dict[tag])
        return input

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
            if examples:
                examples = f'\n{self.examples_preamble}\n\n{examples}\n'
            if self.use_prompt_tags:
                tags_prompt = f'\n{self.prompt_tags_preamble}\n\n{self.prompt_tags}'
            else:
                tags_prompt = ''
            prompt = self.prompt_layout.format(tags_prompt=tags_prompt,
                                               examples_prompt=examples,
                                               sentence=sentence,
                                               eos_token=self.tokenizer.eos_token)
            if i==0:
                with open(f'./misc/prompt_sample_{split}.txt', 'w') as f: f.write(prompt)
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

def get_original_text(element):
    """Extract the raw text of a turn exactly as it appears in the XML file, including all tags"""
    # Get the element's inner text content including children tags
    content = ''.join([el for el in element.itertext()]).strip()
    return content

def subel_to_string(subel, just_value = False):
    if subel.tag == 'turn':
        content = subel.text
        return content if content is not None else ''
    k, v = list(subel.attrib.items())[0]
    if not just_value:
        content = f'<{subel.tag} {k}=\"{v}\">{subel.text}</{subel.tag}>'
        return content
    else:
        return v

def get_text(element, just_value = False):
    content_list = []
    for subel in element.iter():
        subel_string = subel_to_string(subel, just_value=just_value)
        if subel.tail:
            tail = subel.tail
            subel_string += f'{tail}'
            subel_string = subel_string.strip()
        if subel_string:
            content_list.append(subel_string)
    content = ' '.join(content_list)
    return content

def get_inner_xml(element):
    """Preserve tags and their attributes inside an element."""
    print(element.attrib)
    out = []
    for e in element:
        out.append(ET.tostring(e, encoding="unicode"))
    out_string = ''.join(out)
    return out_string

def convert_files(walk_path = './data'):
    df = pd.DataFrame()
    for root, dirs, files in os.walk(walk_path):
        for F in files:
            if F.endswith('.era'):
                filename = os.path.join(root, F)
                tree = ET.parse(filename)
                xml_root = tree.getroot()

                elem = xml_root.find(".//text")
                meta = elem.attrib
                meta_df = pd.DataFrame([meta])

                turns = []
                for turn in elem.findall(".//turn"):
                    speaker = turn.attrib.get("who", "student")
                    turn_type = turn.attrib.get("type", "")
                    
                    text_og = get_original_text(turn)
                    text_an = get_text(turn, just_value = False)
                    text_ok = get_text(turn, just_value = True)
                    
                    turns.append({
                        "speaker": speaker,
                        "turn_type": turn_type,
                        "text_an": text_an,
                        "text_og": text_og,
                        "text_ok": text_ok,
                    })

                turns_df = pd.DataFrame(turns)
                turns_df['text_an'] = turns_df['text_an'].apply(lambda x: x if x else np.nan)
                turns_df = turns_df[turns_df['speaker'] == 'student']
                df = pd.concat([df, turns_df])
    return df.reset_index()

def make_dataset(data_path: str, layout_path: str, tags_path: str, tokenizer_name: str, n_icl_samples: int):
    df = convert_files(data_path)
    print(df)
    prompt_layout = open(layout_path, 'r').read()
    prompt_tags = open(tags_path, 'r').read()
    dataset = CausalLMDataset(df,
                                prompt_layout,
                                prompt_tags,
                                tokenizer_name,
                                n_icl_samples=n_icl_samples,
                                )
    return dataset

def collect_results(dir_path: str):
    df_list = []
    for root, dirs, files in os.walk(dir_path):
        if 'config.json' in files:
            # print(files)
            config = json.load(open(os.path.join(root, 'config.json')))
            try:
                results_dev = json.load(open(os.path.join(root, 'results_dev.json')))
                best_epoch_dev = results_dev[np.argmax([el['micro_f1'] for el in results_dev])]['epoch']
                micro_f1_dev = results_dev[best_epoch_dev]['micro_f1']
                fine_tuned = 1
            except:
                best_epoch_dev = 0
                micro_f1_dev = 0
                fine_tuned = 0
            results_test = json.load(open(os.path.join(root, 'results_test.json')))
            model_name = config['model_name'].split('/')[-1]
            entry = {
                'model_name': model_name,
                'seed': config['seed'],
                'tags': 'y' if config['use_prompt_tags'] else 'n',
                'n_icl': config['n_icl_samples'],
                'epoch': best_epoch_dev,
                'ft': fine_tuned,
                'micro_f1_dev': micro_f1_dev,
                'micro_f1': results_test['micro_f1'],
                'micro_p': results_test['micro_precision'],
                'micro_r': results_test['micro_recall'],
            }
            metric_dict = {k: v['f1'] for k, v in results_test['per_tag_metrics'].items()}
            entry.update(metric_dict)
            # print(entry)
            df_list.append(entry)
    df = pd.DataFrame(df_list)
    return df

def extract_tags(input: str) -> List[List[str]]:
    """
    Extract tags with 'corr' attribute.
    Returns list of tuples: (tag, content)
    """
    # Primary pattern: corr with double quotes (with proper closing tag)
    pattern1 = r'<([A-Z]+[A-Z0-9]*)\s+[^>]*corr="[^"]*"[^>]*>(.*?)</\1>'
    matches1 = re.findall(pattern1, input)
    
    # Secondary pattern: corr with single quotes (with proper closing tag)  
    pattern2 = r"<([A-Z]+[A-Z0-9]*)\s+[^>]*corr='[^']*'[^>]*>(.*?)</\1>"
    matches2 = re.findall(pattern2, input)
    
    # Combine all matches (preserving duplicates)
    all_matches = matches1 + matches2
    
    return [(tag, content.strip()) for tag, content in all_matches]

def main(args):
    dataset = make_dataset(args.data_path, args.layout_path, args.tags_path, args.tokenizer_name, args.n_icl_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A sample argparse program")
    parser.add_argument("--data_path", help="Path of the directory containing raw data files to convert and gather into a dataset")
    parser.add_argument("--layout_path", help="Path of the layout text file", default='./misc/prompt_layout_tags.txt')
    parser.add_argument("--tags_path", help="Path of the tags text file", default='./misc/prompt_tags.txt')
    parser.add_argument("--tokenizer_name",
                        help="Path of the directory containing raw data files to convert and gather into a dataset",
                        default = "meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--n_icl_samples", type=int, help="Number of training epochs", default=10)
    args = parser.parse_args()
    main(args)

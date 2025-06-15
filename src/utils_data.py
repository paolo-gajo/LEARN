import argparse
import xml.etree.ElementTree as ET
import numpy as np
import os
import json
import re
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from bs4 import BeautifulSoup

class CausalLMDataset:
    def __init__(self,
                data: pd.DataFrame,
                tokenizer,
                config: dict,
                clean: bool=True,
                debug: bool=False,
                ):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_len = 0
        self.debug = debug
        if clean:
            self.clean()
        if self.config['samples_type'] == 'random':
            df_train, df_dev = train_test_split(self.data, test_size=0.2, random_state=config['seed'])
            self.train = df_train.reset_index(drop=True)
            df_dev, df_test = train_test_split(df_dev, test_size=0.5, random_state=config['seed'])
            self.dev = df_dev.reset_index(drop=True)
            self.test = df_test.reset_index(drop=True)
        elif self.config['samples_type'] in ['context_raw', 'context_ann']:
            grouped_data = self.group_conv_id(self.data)
            df_train, df_dev = train_test_split(grouped_data, test_size=0.2, random_state=config['seed'])
            self.train = pd.concat(df_train).reset_index(drop=True)
            df_dev, df_test = train_test_split(df_dev, test_size=0.5, random_state=config['seed'])
            self.dev = pd.concat(df_dev).reset_index(drop=True)
            self.test = pd.concat(df_test).reset_index(drop=True)
            ...

        if self.config['coarse']:
            self.train['text_annotated'] = self.train['text_annotated'].apply(lambda x: self.coarsen_tags(x))
            self.dev['text_annotated'] = self.dev['text_annotated'].apply(lambda x: self.coarsen_tags(x))
            self.test['text_annotated'] = self.test['text_annotated'].apply(lambda x: self.coarsen_tags(x))
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

    def group_conv_id(self, df):
        return [df[df['conv_id'] == id] for id in df['conv_id'].unique()]

    def clean(self):
        self.data['text_raw'] = self.data['text_raw'].apply(lambda x: x.replace(r'\0', ''))

    def coarsen_tags(self, input: str):
        tags = set([el[0] for el in extract_tags(input)])
        for tag in tags:
            input = input.replace(tag, self.config['tag_dict'][tag])
        return input

    def rng_icl_examples(self, split, idx):
        if not self.config['n_icl_samples']:
            return ''
        text_og_list_examples = self.train['text_raw']
        text_an_list_examples = self.train['text_annotated']
        if split == 'train':
            examples_an = text_an_list_examples.drop(labels=getattr(self, split).index[idx])
        else:
            examples_an = text_an_list_examples
        mask_pos = examples_an.apply(lambda x: '</' in x)
        mask_neg = examples_an.apply(lambda x: '</' not in x)
        examples_an_pos = examples_an[mask_pos].sample(n=self.config['n_icl_samples'])
        examples_an_neg = examples_an[mask_neg].sample(n=self.config['n_icl_samples'])
        examples_og_pos = text_og_list_examples[examples_an_pos.index]
        examples_og_neg = text_og_list_examples[examples_an_neg.index]
        examples_pos = [f'{og}###{an}' for og, an in zip(examples_og_pos, examples_an_pos)]
        examples_neg = [f'{og}###{an}' for og, an in zip(examples_og_neg, examples_an_neg)]
        examples = examples_pos + examples_neg
        shuffle(examples)
        examples = '\n'.join(examples)
        examples = f"\n\n{self.config['examples_preamble']}\n\n{examples}\n\n"
        return examples
    
    def context_examples(self, split, idx):
        split_data = getattr(self, split)
        sentence_row = split_data.iloc[idx]
        if sentence_row['turn_type'] != 'student':
            return None
        conv_id = sentence_row['conv_id']
        sentence_row_id = sentence_row['local_id']
        df = split_data[split_data['conv_id'] == conv_id]
        df = df.iloc[max(0, sentence_row_id - self.config['k_window']):sentence_row_id]
        if self.config['samples_type'] == 'context_raw':
            field = 'text_raw'
        elif self.config['samples_type'] == 'context_ann':
            field = 'text_annotated'
        convo = df.apply(lambda x: f"{x['speaker']}: {x[field]}", axis = 1).tolist()
        out = '\n'.join(convo)
        out = f"\n\n{self.config['convo_preamble']}\n\n{out}\n\n"
        return out

    def make_samples(self, split):
        split_data = getattr(self, split)
        text_list_raw = split_data['text_raw']
        text_list_ann = split_data['text_annotated']
        chat_list = []
        len_list = []
        for i in range(len(text_list_raw)):
            sentence = text_list_raw.iloc[i]
            expected_output = text_list_ann.iloc[i]
            if self.config['samples_type'] == 'random':
                if len(self.config['speakers']) > 1:
                    print(f'`samples_type` is `random`, but `speakers`>1, make sure this is correct!!!')
                examples = self.rng_icl_examples(split, i)
            elif self.config['samples_type'] in ['context_raw', 'context_ann']:
                assert len(self.config['speakers']) > 1, f"Speakers need to be > 1 with `samples_type` == {self.config['samples_type']}"
                examples = self.context_examples(split, i)
                if examples is None:
                    continue
                sentence = f'student: {sentence}'
                expected_output = f'student: {expected_output}'
                
            if self.config['use_prompt_tags']:
                tags_prompt = f"\n\n{self.config['prompt_tags_preamble']}\n\n{self.config['prompt_tags']}\n\n"
            else:
                tags_prompt = ''
            prompt = self.config['prompt_layout'].format(tags_prompt=tags_prompt,
                                               examples_prompt=examples,
                                               sentence=sentence,
                                               eos_token=self.tokenizer.eos_token)
            prompt = re.sub(r'\n{3,}', '\n\n', prompt)
            # Create the prompt part (same for train and eval)
            chat_prompt = [
                {"role": "system", "content": self.config['sys_prompt']},
                {"role": "user", "content": prompt},
            ]
            prompt_length_tok = self.get_prompt_len(chat_prompt)
            len_list.append(prompt_length_tok)
            assert prompt_length_tok < self.config['max_length'], f"prompt_length_tok {prompt_length_tok} >= {self.config['max_length']}"
            if split == 'train':
                # For training: include the assistant response
                chat = chat_prompt + [{"role": "assistant", "content": expected_output}]
                chat_prompt_formatted = self.tokenizer.apply_chat_template(chat,
                                                        tokenize=False,
                                                        add_generation_prompt=False)
                last_sample = {'text': chat_prompt_formatted}
                chat_list.append(last_sample)
            else:
                # For evaluation: separate prompt and expected output
                chat_prompt_formatted = self.tokenizer.apply_chat_template(chat_prompt,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
                last_sample = {
                    'prompt': chat_prompt_formatted,
                    'completion': expected_output,
                    'prompt_length': len(self.tokenizer.encode(chat_prompt_formatted))
                }
                chat_list.append(last_sample)
            if self.debug:
                with open(f'./scratch/prompt_sample_{split}.txt', 'w') as f: f.write(chat_prompt_formatted)
        max_len_list = max(len_list)
        if max_len_list > self.max_len:
            self.max_len = max_len_list
        return pd.DataFrame(chat_list)  
    
    def get_prompt_len(self, input):
        return self.tokenizer(self.tokenizer.apply_chat_template(input,
                            tokenize=False,
                            add_generation_prompt=False),
                            return_tensors = 'pt')['input_ids'].shape[-1]

class SampleMaker:
    def __init__(self, split, ):
        self.split = split
    


        

def make_tags_prompt(tags_csv_path: str):
    df = pd.read_csv(tags_csv_path, sep='\t')
    tags = df['Tag'].tolist()
    description = df['Description'].tolist()
    out_prompt = ''
    for t, d in zip(tags, description):
        out_prompt += f'{t}: {d}\n'
    return out_prompt

def get_raw_input(element):
    """Extract the raw text of a turn exactly as it appears in the XML file, including all tags"""
    # Get the element's inner text content including children tags
    content = ''.join([el for el in element.itertext()]).strip()
    return content

def subel_to_string(subel, just_value=False) -> str:
    # 1) For <turn> elements, just return their .text
    if subel.tag == 'turn':
        return subel.text or ''
    
    # 2) Fetch the one-and-only corr attribute
    attrib_items = list(subel.attrib.items())
    if not attrib_items:
        return ''   # no corr, nothing to do
    key, corr_val = attrib_items[0]
    if just_value:
        return corr_val
    
    # 3) Serialize this element to XML and re‑run through extract_tags
    #    to get precisely the same tag/content logic you already tested.
    fragment = ET.tostring(subel, encoding='unicode')
    #    extract_tags returns [(TAG_NAME, content_str)]
    tag_name, content = extract_tags(fragment)[0]
    
    # 4) Re‑substitute any real null bytes
    if '\x00' in content:
        content = re.sub('\x00', r'\\0', content)
    
    # 5) Re‑build the snippet exactly as you'd like
    return f'<{subel.tag} {key}="{corr_val}">{content}</{subel.tag}>'

def get_text(element, just_value=False):
    content_list = []
    for subel in element.iter():
        subel_string = subel_to_string(subel, just_value=just_value)
        if subel.tail:
            subel_string += subel.tail
        if subel_string.strip():
            content_list.append(subel_string.strip())
    out = " ".join(content_list)
    return out

def get_inner_xml(element):
    """
    Extract the inner XML content from an XML element.
    
    Args:
        element: An xml.etree.ElementTree.Element object
        
    Returns:
        str: The inner XML content as a string
    """
    # Get the text content directly after the opening tag
    inner_content = element.text or ""
    
    # Add all child elements and their tail text
    for child in element:
        # Convert child element back to XML string
        child_xml = ET.tostring(child, encoding='unicode')
        inner_content += child_xml
    return inner_content

def convert_files(walk_path = './data', speakers = ['student', 'chatbot']):
    df = pd.DataFrame()
    i = 0
    for root, dirs, files in os.walk(walk_path):
        for F in files:
            if F.endswith('.era'):
                filename = os.path.join(root, F)
                tree = ET.parse(filename)
                xml_root = tree.getroot()

                task_list = xml_root.findall(".//task")
                for task in task_list:
                    task_type = task.attrib.get("type", "")
                    meta = task.attrib
                    meta_df = pd.DataFrame([meta])
                    turns = []
                    for turn in task.findall(".//turn"):
                        speaker = turn.attrib.get("who", "student")
                        turn_type = turn.attrib.get("type", "")
                        text_raw = get_raw_input(turn)
                        # turn looks like this in the dataset:
                        # <turn type="student">Thank you so much. But I also have to say that I <LP corr="have been mocked a lot"><GVT corr="have received">received</GVT> a lot of mocking</LP> because of this passion of mine</turn>
                        # here it goes as an Element
                        # `text_annotated` simply needs to be what is shown inside the turn, e.g.
                        # Thank you so much. But I also have to say that I <LP corr="have been mocked a lot"><GVT corr="have received">received</GVT> a lot of mocking</LP> because of this passion of mine
                        text_annotated = get_inner_xml(turn).strip()
                        text_correct = get_text(turn, just_value = True)
                        turns.append({
                            "speaker": speaker,
                            "turn_type": turn_type,
                            "text_annotated": text_annotated,
                            "text_raw": text_raw,
                            "text_correct": text_correct,
                        })

                    turns_df = pd.DataFrame(turns)
                    turns_df['text_annotated'] = turns_df['text_annotated'].apply(lambda x: x if x else np.nan)
                    turns_df['conv_id'] = i
                    turns_df['task_type'] = task_type
                    speaker_mask = turns_df['turn_type'].apply(lambda x: x in speakers)
                    turns_df = turns_df[speaker_mask]
                    turns_df['local_id'] = range(len(turns_df))
                    df = pd.concat([df, turns_df])
                i += 1
    df = df.reset_index()
    return df

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
            df_list.append(entry)
    df = pd.DataFrame(df_list)
    return df

def extract_tags(input_str: str) -> List[Tuple[str, str]]:
    """
    Extract all tags with a 'corr' attribute in document order.
    Returns list of tuples: (tag_name, full_text_content)
    """
    wrapper = f"<root>{input_str}</root>"
    soup = BeautifulSoup(wrapper, "html.parser")
    results = []
    
    for tag in soup.find_all(lambda t: t.has_attr("corr")):
        # 1) get only direct text parts
        direct_parts = [c for c in tag.contents if isinstance(c, str)]
        direct_text = "".join(direct_parts).strip()
        
        # 2) decide which text to use
        if tag.name.upper() == "GNC":
            # for GNC, always take the full_text (nested + direct)
            content = tag.get_text().strip()
        elif direct_text:
            # for everything else, if there's any direct text, use *only* that
            content = direct_text
        else:
            # otherwise fall back to the full text
            content = tag.get_text().strip()
        
        # 3) replace **any** actual null bytes with the two-character string "\0"
        if "\x00" in content:
            content = re.sub("\x00", r"\\0", content)
        
        results.append((tag.name.upper(), content))
    
    return results

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

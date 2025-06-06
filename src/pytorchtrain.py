import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from convert import convert_files
from utils import CausalLMDataset, Collator
from torch.utils.data import DataLoader

df = convert_files('./data')
print(df)

model_name = "meta-llama/Llama-3.2-1B-Instruct"
# model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "meta-llama/Llama-3.3-70B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side = 'left')
tokenizer.pad_token = tokenizer.eos_token
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LlamaForCausalLM.from_pretrained(model_name).to(device)
prompt_layout = open('./misc/prompt_layout_no_tags.txt', 'r').read()
prompt_tags = open('./misc/prompt_tags.txt', 'r').read()
dataset = CausalLMDataset(df, prompt_layout, prompt_tags, tokenizer)
collator = Collator(tokenizer)
loader = DataLoader(dataset, batch_size=1, collate_fn=collator.collate)

for batch in loader:
    batch = {k: v.to(model.device) for k, v in batch.items()}
    out = model(**batch)
    for i in range(out.shape[0]):
        input_ids_sample = batch['input_ids'][i]
        labels_sample = batch['labels'][i]
        len_padding_input = torch.where(input_ids_sample == tokenizer.pad_token_id)[0].shape[0] + 1
        len_prompt_output = len(input_ids_sample)
        len_padding_labels = torch.where(labels_sample == tokenizer.pad_token_id)[0].shape[0] + 1
        input_readable = tokenizer.decode(input_ids_sample[len_padding_input:], skip_special_tokens=False)
        output_readable = tokenizer.decode(out[i][len_prompt_output:], skip_special_tokens=False)
        labels_readable = tokenizer.decode(labels_sample[len_padding_labels:], skip_special_tokens=False)
        print('input', input_readable)
        print('output', output_readable)
        print('labels', labels_readable)
        print('#' * 100)

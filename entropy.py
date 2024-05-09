from transformers import LlamaForCausalLM, AutoTokenizer
from data_converter import convert_wiki_dataset, convert_cnn_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
device = "cuda:7"
eps = 1e-9
llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map=device, torch_dtype=torch.float16, _attn_implementation="eager")
llm = llm.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

dataset = convert_cnn_dataset(tokenizer=tokenizer, seq_len=64)
GEN_LEN = 1024
avg_entropy = torch.zeros(GEN_LEN).to(device)
avg_infomation = torch.zeros(GEN_LEN).to(device)
num_samples = torch.zeros(GEN_LEN).to(device)
with torch.inference_mode():
    for batch in tqdm(dataset, total=len(dataset)):
        input_ids = batch["input_ids"].to(device).unsqueeze(0)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(0)
        past_key_values = None
        outputs = llm(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values)
        for i in range(GEN_LEN):
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            prob = F.softmax(logits, dim=-1)[:, -1,:]
            
            
            new_token = torch.multinomial(prob, num_samples=1)
            
            
            if new_token[0][0] == 0 or new_token[0][0] == 2:
                break
            avg_entropy[i] += torch.sum(-prob[0] * torch.log(prob[0]+eps))
            
            avg_infomation[i] += (-torch.log(prob[0][new_token[0][0]]+eps))
            num_samples[i] += 1
            input_ids = new_token
            outputs = llm(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
        
    
entropy = [0 for _ in range(GEN_LEN)]
infomation = [0 for _ in range(GEN_LEN)]

for i in range(GEN_LEN):
    if num_samples[i].item() > 0.9:
        entropy[i] = avg_entropy[i].item() / num_samples[i].item()
        infomation[i] = avg_infomation[i].item() / num_samples[i].item()


plt.scatter(list(range(GEN_LEN)), entropy)
plt.savefig("7b_cnn_entropy_dropout.jpg")
plt.clf()

plt.scatter(list(range(GEN_LEN)), infomation)
plt.savefig("7b_cnn_information_dropout.jpg")







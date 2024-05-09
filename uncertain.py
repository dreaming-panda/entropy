from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, GemmaForCausalLM, MistralForCausalLM
from data_converter import convert_wiki_dataset, convert_cnn_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
        "--model",  type=str, default="google/gemma-7b", help="Name of model e.g. `hf`"
    )
parser.add_argument(
        "--output",  type=str, default="gemma2b.pdf", help="Name of output file e.g. `hf`"
    )
args = parser.parse_args()
device = "cuda:0"
eps = 1e-9
path = args.model.split("/")[1]

llm = AutoModelForCausalLM.from_pretrained(args.model, device_map=device, torch_dtype="auto", _attn_implementation="eager")
llm = llm.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token == None:
    tokenizer.pad_token = tokenizer.eos_token
sentences = []
dataset = convert_cnn_dataset(tokenizer=tokenizer, seq_len=64)
GEN_LEN = 1024
avg_kl = torch.zeros(GEN_LEN).to(device)
avg_entropy = torch.zeros(GEN_LEN).to(device)
num_samples = torch.zeros(GEN_LEN).to(device)
with torch.inference_mode():
    for batch in tqdm(dataset, total=len(dataset)):
        sentence :torch.LongTensor =  batch["input_ids"].to(device).unsqueeze(0)
        input_ids = batch["input_ids"].to(device).unsqueeze(0).repeat(1,1)
        attention_mask = batch["attention_mask"].to(device).unsqueeze(0).repeat(1,1)
        past_key_values = None
        outputs = llm(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values)
        for i in range(GEN_LEN):
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            prob = F.softmax(logits.to(torch.float32) / 0.6, dim=-1)[:, -1,:]
            
            
            new_token = torch.multinomial(prob[0:1], num_samples=1)
            sentence = torch.cat([sentence, new_token], dim=-1)
            new_token = new_token.repeat(1,1)
            
            if new_token[0][0] == tokenizer.pad_token_id or new_token[0][0] == tokenizer.eos_token_id:
                break
            
            #avg_kl[i] += torch.nn.functional.kl_div(torch.log(prob[0:4]+eps), prob[4:8], reduction='batchmean')
            
            avg_entropy[i] += torch.sum(-prob[0] * torch.log(prob[0]+eps))
            num_samples[i] += 1
            input_ids = new_token
            outputs = llm(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
        
        sentence = tokenizer.decode(
                    sentence[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                    spaces_between_special_tokens=False)
        sentences.append("\n")
        sentences.append(sentence)
        sentences.append("*"*50 + "\n")
        
        
    
kl = [0 for _ in range(GEN_LEN)]
entropy = [0 for _ in range(GEN_LEN)]

for i in range(GEN_LEN):
    if num_samples[i].item() > 0.9:
        kl[i] = avg_kl[i].item() / num_samples[i].item()
        entropy[i] = avg_entropy[i].item() / num_samples[i].item()

plt.scatter(list(range(GEN_LEN)), entropy)
plt.savefig("pdfs/" + path + "_entropy.pdf")
plt.clf()
# plt.scatter(list(range(1, GEN_LEN)), kl[1:])
# plt.savefig("pdfs/" + path + "_uncertainty.pdf")
# plt.clf()
plt.scatter(list(range(1, GEN_LEN)), num_samples[1:].tolist())
plt.savefig("pdfs/" + path+ "_num_samples.pdf")

text_path = "texts/"+path+".txt"

with open(text_path, "w") as f:
    f.writelines(sentences)


torch.save(avg_entropy, "raw/" + path + "_entropy.pt")
torch.save(num_samples, "raw/" + path + "_num_samples.pt")
    





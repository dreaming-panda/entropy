from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from data_converter import convert_wiki_dataset, convert_cnn_dataset
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from transformers.generation.configuration_utils import GenerationConfig
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
        "--model",  type=str, default="google/gemma-2b", help="Name of model e.g. `hf`"
    )
parser.add_argument(
        "--output",  type=str, default="gemma2b_cnn_entropy.jpg", help="Name of output file e.g. `hf`"
    )
parser.add_argument(
        "--T",  type=float, default=1.0, help="temperature of sampling"
    )
parser.add_argument
args = parser.parse_args()
path = args.model.split("/")[1]
device = "cuda:8"
eps = 1e-9
llm = AutoModelForCausalLM.from_pretrained(args.model, device_map=device, torch_dtype=torch.float16)
llm = llm.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token == None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = convert_cnn_dataset(tokenizer=tokenizer, seq_len=64)
GEN_LEN = 1024
avg_entropy = torch.zeros(GEN_LEN).to(device)
avg_infomation = torch.zeros(GEN_LEN).to(device)
num_samples = torch.zeros(GEN_LEN).to(device)

config = GenerationConfig.from_pretrained(args.model)
config.do_sample = False
config.num_beams = 8
config.max_new_tokens = GEN_LEN
with torch.inference_mode():
    for batch in tqdm(dataset, total=len(dataset)):
        input_ids = batch["input_ids"].to(device).unsqueeze(0)
        initial_len = input_ids.shape[1]
        outputs = llm.generate(inputs=input_ids, generation_config=config)

        logits :torch.Tensor = llm(outputs).logits[:,initial_len:,]
        logits = F.softmax(logits/args.T, dim=-1).float()
        e = -(logits * torch.log(logits + eps)).sum(dim=-1).squeeze()

        gen_len = outputs.shape[1] - initial_len
        avg_entropy[:gen_len] += e
        num_samples[:gen_len] += 1
        
        
    
entropy = [0 for _ in range(GEN_LEN)]


for i in range(GEN_LEN):
    if num_samples[i].item() > 0.9:
        entropy[i] = avg_entropy[i].item() / num_samples[i].item()
        


plt.scatter(list(range(GEN_LEN)), entropy)
plt.savefig(path + "_entropy.pdf")
plt.clf()
plt.scatter(list(range(1, GEN_LEN)), num_samples[1:].tolist())
plt.savefig(path+ "_num_samples.pdf")








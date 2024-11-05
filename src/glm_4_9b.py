import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#from huggingface_hub import login

MODEL_ID = "THUDM/glm-4-9b-chat"

class GLM_4_9B():
    def __init__(self, 
                 device : torch.device, 
                 dtype : torch.dtype):
        self.device = device
        self.dtype = dtype

    def setup(self):
        print("--- initialize tokenizer ---")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True)
        
        print("--- start load model ---")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True).to(self.device)
    
    @torch.inference_mode()
    def infer(self, query : str) -> str:
        print("--- start inference ---")
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_dict=True, 
                                            return_tensors="pt",
                                            ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                #max_length = 2500, 
                max_new_tokens= 50, 
                top_k = 1, 
                do_sample = True, 
                return_dict_in_generate = True, 
                output_scores = True)
            
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def cleanup(self):
        del self.model
        del self.tokenizer

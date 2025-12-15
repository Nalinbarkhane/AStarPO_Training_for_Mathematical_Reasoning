import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class Sampler:
    def __init__(self, model_name, device="cuda", load_in_8bit=False):
        """
        Initialize the sampler with a language model.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cuda' or 'cpu')
            load_in_8bit: Whether to use 8-bit quantization
        """
        print(f"Loading model: {model_name}")
        self.device = device
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=load_in_8bit
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        self.model.eval()
        
        # Load system prompt
        prompt_path = "prompts/pag_system.txt"
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
        else:
            # Default system prompt if file doesn't exist
            self.system_prompt = "You are a helpful math assistant. Solve the following problem step by step and provide your final answer in \\boxed{}."
            print(f"Warning: {prompt_path} not found. Using default system prompt.")
    
    def format_prompt(self, question):
        """
        Format the question with system prompt using chat template.
        
        Args:
            question: The math question
            
        Returns:
            Formatted prompt string
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def generate_samples(self, question, num_samples=4, max_new_tokens=512, 
                        temperature=0.7, top_p=0.9):
        """
        Generate multiple solution attempts for a question.
        
        Args:
            question: The math problem
            num_samples: How many solutions to generate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            List of generated solutions (strings)
        """
        # Format the prompt
        prompt = self.format_prompt(question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        samples = []
        prompt_length = inputs['input_ids'].shape[1]
        
        # Generate multiple samples
        for i in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,  # Enable sampling
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode only the generated part (remove prompt)
            generated_ids = outputs[0][prompt_length:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            samples.append(generated_text.strip())
        
        return samples
    
    def generate_single(self, question, max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
        Generate a single solution (useful for evaluation).
        """
        samples = self.generate_samples(
            question, 
            num_samples=1, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return samples[0]
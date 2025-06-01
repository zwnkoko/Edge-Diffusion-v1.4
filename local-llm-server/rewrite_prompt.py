import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import time
import re

class PromptEnricher:
    def __init__(self):
        # Load the TinyLlama model
        print("Loading TinyLlama model...")
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="auto"
        )
        print("Model loaded!")
        
    
    def enrich_prompt(self, prompt):
        """Enrich the user's prompt with more descriptive details for better image generation."""
        
        # Format for TinyLlama chat model
        input_text = f"<|system|>\nYou are a helpful assistant that improves image generation prompts by adding only concise, objective physical appearance details to the objects mentioned in the prompt. Describe only factual attributes such as size, shape, color, texture, and arrangement. Do not introduce any new objects, background elements, or narrative context. \n<|user|>\nEnrich this prompt with visual details: {prompt}\n<|assistant|>"

        
        # Generate the enriched prompt
        start_time = time.time()
        inputs = self.tokenizer(input_text, return_tensors="pt")
        
        # Generate with appropriate parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Increased to avoid cutoff
                do_sample=False,      
                # temperature=0.7,
                # top_p=0.9,
                num_return_sequences=1,
                # repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id  # Ensure proper ending
            )
        
        # Decode the generated text
        enriched_prompt = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the enriched prompt part (after the assistant token)
        assistant_pattern = "<|assistant|>"
        if assistant_pattern in enriched_prompt:
            enriched_prompt = enriched_prompt.split(assistant_pattern)[-1].strip()
        
        # Log the enrichment
        print(f"Original prompt: {prompt}")
        print(f"Enriched prompt: {self.remove_incomplete_sentence(enriched_prompt)}")
        print(f"Enrichment time: {time.time() - start_time:.2f} seconds")
        
        return self.remove_incomplete_sentence(enriched_prompt)
    
    def remove_incomplete_sentence(self,text):
        sentences = re.findall(r'[^.]*\.', text)
        return ''.join(sentences) if sentences else ''

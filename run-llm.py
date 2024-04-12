import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate responses using a language model.")
    parser.add_argument('--prompts_file', type=str, required=True, help='File to read prompts from.')
    parser.add_argument('--output_file', type=str, required=True, help='File to append the responses to.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of model.')
    parser.add_argument('--tokenizer_name', type=str, help='Name of model.')
    parser.add_argument('--config_path', type=str, help='Path to config.')
    parser.add_argument('--prompt_template_file', type=str, help='File with template for prompt.')
    parser.add_argument('--flip_tokens', action='store_true', help='Flip tokens before and after model.')
    args = parser.parse_args()

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(args.config_path or args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)
    
    # Ensure the model is in evaluation mode
    model = model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Read prompts from the input file
    with open(args.prompts_file, 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    # Optionally format prompts using a template
    if args.prompt_template_file:
        with open(args.prompt_template_file, 'r') as file:
            prompt_template = file.read().strip()
        prompts = [prompt_template.format(prompt=prompt) for prompt in prompts]
    
    # Generate responses in batches
    batch_size = 8
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, return_token_type_ids=False)
        inputs = inputs.to(device)
        if args.flip_tokens:
            inputs['input_ids'] = inputs['input_ids'].flip((-1,))
            inputs['attention_mask'] = inputs['attention_mask'].flip((-1,))
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)
        if args.flip_tokens:
            outputs = outputs.flip((-1,))
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses.extend(batch_responses)

    # Write responses to the output file
    with open(args.output_file, "w", encoding="utf-8") as file:
        for i, response in enumerate(responses):
            print(f"RESPONSE {i+1}:\n", response)
            file.write(response + "\n")

if __name__ == "__main__":
    main()

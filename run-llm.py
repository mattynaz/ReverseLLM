import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate responses using a language model.")
    parser.add_argument('--prompts_file', type=str, required=True, help='File to read prompts from.')
    parser.add_argument('--output_file', type=str, required=True, help='File to append the responses to.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of model.')
    parser.add_argument('--prompt_template_file', type=str, help='File with template for prompt.')
    args = parser.parse_args()

    # Load model and tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure the model is in evaluation mode
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read prompts from the input file
    with open(args.input_file, 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]
    
    # Optionally format prompts using a template
    if args.prompt_template_file:
        with open(args.prompt_template_file, 'r') as file:
            prompt_template = file.read().strip()
        prompts = [prompt_template.format(prompt) for prompt in prompts]
    
    # Generate responses in batches
    batch_size = 8
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        batch_inputs = batch_inputs.to(device)
        with torch.no_grad():
            batch_outputs = model.generate(**batch_inputs, max_new_tokens=150, num_return_sequences=1)
        batch_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in batch_outputs]
        responses.extend(batch_responses)

    # Write responses to the output file
    with open(args.output_file, "w") as file:
        for response in responses:
            file.write(response + "\n")


if __name__ == "__main__":
    main()

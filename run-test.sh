# python run-llm.py \
#     --prompts_file "mistral-7b-reverse-instruct-prompts.txt" \
#     --output_file "mistral-7b-reverse-instruct-responses.txt" \
#     --model_name "Philipp-Sc/mistral-7b-reverse-instruct" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --config_path "./config.json" \
#     --prompt_template_file "mistral-7b-reverse-instruct-prompt-template.txt"

# python run-llm.py \
#     --prompts_file "prompts.txt" \
#     --output_file "vikp_reverse_instruct-responses.txt" \
#     --model_name "vikp/reverse_instruct" \
#     --prompt_template_file "vikp_reverse_instruct-prompt-template.txt"

python run-llm.py \
    --prompts_file "prompts.txt" \
    --output_file "reverse-pythia-160m-responses.txt" \
    --model_name "afterless/reverse-pythia-160m" \
    --flip_tokens

# python run-llm.py \
#     --prompts_file "reverse-pythia-160m-responses.txt" \
#     --output_file "llama-2-7b-chat-hf-responses.txt" \
#     --model_name "meta-llama/Llama-2-7b-chat-hf" \

# python run-llm.py \
#     --prompts_file "mistral-7b-reverse-instruct-prompts.txt" \
#     --output_file "mistral-7b-reverse-instruct-responses.txt" \
#     --model_name "Philipp-Sc/mistral-7b-reverse-instruct" \
#     --tokenizer_name "meta-llama/Llama-2-7b-chat-hf" \
#     --config_path "./config.json" \
#     --prompt_template_file "mistral-7b-reverse-instruct-prompt-template.txt"

python run-llm.py \
    --prompts_file "prompts.txt" \
    --output_file "vikp_reverse_instruct-responses.txt" \
    --model_name "Philipp-Sc/mistral-7b-reverse-instruct" \
    --prompt_template_file "vikp_reverse_instruct-prompt-template.txt"


python run-llm.py \
    --prompts_file "vikp_reverse_instruct-responses.txt" \
    --output_file "mistral-7b-instruct-responses.txt" \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \

python run-llm.py \
    --prompts_file "mistral-7b-reverse-instruct-prompts.txt" \
    --output_file "mistral-7b-reverse-instruct-responses.txt" \
    --model_name "Philipp-Sc/mistral-7b-reverse-instruct" \
    --prompt_template_file "mistral-7b-reverse-instruct-prompt-template.txt"

python run-llm.py \
    --prompts_file "mistral-7b-reverse-instruct-responses.txt" \
    --output_file "mistral-7b-instruct-responses.txt" \
    --model_name "mistralai/Mistral-7B-Instruct-v0.2" \

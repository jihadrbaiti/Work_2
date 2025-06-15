import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smoothing_function = SmoothingFunction().method1
test_data = pd.read_csv('/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/test.csv')

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    '/localssd/chouaib/geo_ai/Model2/SFT/merged_model_final',
    use_cache=True,
    torch_dtype=torch.float16
).to(device)
#MBZUAI-Paris/Atlas-Chat-2B
tokenizer = AutoTokenizer.from_pretrained(
    '/localssd/chouaib/geo_ai/Model2/SFT/merged_model_final',
    padding_side='left'
)

if tokenizer.chat_template is None:
    tokenizer.chat_template = """{% for message in messages %}
{% if message['role'] == 'system' %}<|system|>
{{ message['content'] }}
<|end|>
{% elif message['role'] == 'user' %}<|user|>
{{ message['content'] }}
<|end|>
{% elif message['role'] == 'assistant' %}<|assistant|>
{{ message['content'] }}
<|end|>
{% endif %}
{% endfor %}"""

special_tokens_to_add = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})

print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")

model.eval()

source_language = "Moroccan dialect"
target_language = "English"

generated_response1 = []
bleu_scores = []
ref = []
src = []

for i in range(len(test_data)):
    prompt = f"Translate this from [{source_language}] to [{target_language}]:\n{source_language}: {test_data['darija'].iloc[i]}\n{target_language}: "
    system = f'You are a native speaker of both [{source_language}] and [{target_language}]. You are an expert post editor of translations from [{source_language}] into [{target_language}] and a helpful assistant dedicated to improving translation quality. You will be provided with a source sentence in [{source_language}] and its translation in [{target_language}]. Your task is to carefully analyze the provided source sentence and translation, and suggest improvements to the translation. Note that you only need to generate a refined translation in [{target_language}] and do not generate anything else.'

    chat_style_prompt = [{"role": "user", "content": prompt}]
    messages = [{"role": "user", "content": prompt}]
    
    
    prompt1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        [prompt1], return_tensors="pt", padding=True, max_length=40, truncation=True
    ).to(device)
    
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            num_beams=5,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated output
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
    ]
    generated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Append source, reference, and prediction
    src.append(test_data['darija'].iloc[i])
    ref.append(test_data['english'].iloc[i])
    generated_response1.append(generated_response)
    
    # Calculate BLEU score
    bleu_score = sentence_bleu(
        [test_data['english'].iloc[i].split()],  # Reference as tokenized words
        generated_response.split(),             # Prediction as tokenized words
        smoothing_function=smoothing_function
    )
    bleu_scores.append(bleu_score)
    
    print(f"**** User Input: {test_data['darija'].iloc[i]}")
    print(f"**** Reference: {test_data['english'].iloc[i]}")
    print(f"**** Generated: {generated_response}")
    print()

# Save results to CSV
results = pd.DataFrame({'Input': src, 'Output': ref, 'Prediction': generated_response1})
results.to_csv('/localssd/chouaib/geo_ai/Work_2/Predictions/predictions_sft_2.csv', index=False)

# Calculate and print average BLEU score
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")

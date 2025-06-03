import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import re 

torch.cuda.set_device(0) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smoothing_function = SmoothingFunction().method1

print('Test script for prediction of CPO _atlas training model')
test_data = pd.read_csv('/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/test.csv')

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    '/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat_2/XALMA/checkpoint-10000',
    torch_dtype=torch.float16
).to(device)
#model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    "/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat_2/XALMA/checkpoint-10000",
    padding_side='left'
)

source_language = "Moroccan dialect"
target_language = "English"

generated_response1 = []
bleu_scores = []
ref = []
src = []

for i in range(len(test_data)):
    prompt = f"Translate this from [{source_language}] to [{target_language}]:\n{source_language}: {test_data['darija'].iloc[i]} \n{target_language}: "
    
    chat_style_prompt = [{"role": "user", "content": prompt}]
    messages = [{"role": "user", "content": prompt}]
    
    
    prompt1 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt1], return_tensors="pt", padding=True, max_length=3000, truncation=True).to(device)
    # Set pad_token to eos_token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    """if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_SFT_CHAT_TEMPLATE"""
### Start not correct 
    #print(inputs.input_ids)
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids.to('cuda:0'),
            num_beams=5,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    # Decode the generated output
    #generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    ## added
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
    ]
    generated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    '''
    # Clean up prompt artifacts
    for removal in [
        f"Translate this from [{source_language}] to [{target_language}]:",
        f"{source_language}:", f"{target_language}:",
        "user", "user user", "\n"
    ]:
        generated_response = generated_response.replace(removal, "")

    generated_response = generated_response.strip()

    def clean_chat_output(text):
        # Remove prompt echo and repeated tags
        text = re.sub(r"<\|user\|>.*?<\|model\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"(?im)^\s*(moroccan dialect|english|user|model)\s*[:\-–]*\s*", "", text)
        text = text.strip()
    
        # Remove Arabic script (Unicode range: \u0600 to \u06FF + Arabic Extended)
        text = re.sub(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+", "", text)

        # Collapse multiple spaces and strip again
        text = re.sub(r"\s+", " ", text).strip()

        return text
    generated_response = clean_chat_output(generated_response)
    '''
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
results.to_csv('/home/jihad.rbaiti/Work_2/CPO/CPO_atlas/Predictions/xalma_atlas_2.csv', index=False)

# Calculate and print average BLEU score
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smoothing_function = SmoothingFunction().method1
test_data = pd.read_csv('/localssd/chouaib/geo_ai/Work_2/data/parallel_dataset/final_data_splitting/test.csv')
print('test of NLLB/xalma_attention')
# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(
    '/localssd/chouaib/geo_ai/Model_NLLB/XALMA_attention/checkpoint-10000',
    torch_dtype=torch.float16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/localssd/chouaib/geo_ai/Model_NLLB/XALMA_attention/checkpoint-10000",
    padding_side='left'
)

'''source_language = "Moroccan dialect"
target_language = "English"'''

generated_response1 = []
bleu_scores = []
ref = []
src = []
output_file = '/localssd/chouaib/geo_ai/Work_2/Predictions/NLLB/nllb_xalma_attention.csv'

for i in range(len(test_data)):
    inputs = tokenizer([test_data['darija'].iloc[i]], return_tensors="pt", padding=True, max_length=200, truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(inputs.input_ids, num_beams=5, max_new_tokens=200, do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.pad_token_id, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
            
            # Decode the generated output
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    source = test_data['darija'].iloc[i]
    reference = test_data['english'].iloc[i]
            #generated_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Append source, reference, and prediction
    src.append(source)
    ref.append(reference)
    generated_response1.append(generated_response)
            
            # Calculate BLEU score
    bleu_score = sentence_bleu( [test_data['english'].iloc[i].split()],  # Reference as tokenized words
                generated_response.split(),             # Prediction as tokenized words
                smoothing_function=smoothing_function)
    bleu_scores.append(bleu_score)
            
    print(f"**** User Input: {test_data['darija'].iloc[i]}")
    print(f"**** Reference: {test_data['english'].iloc[i]}")
    print(f"**** Generated: {generated_response}")
    print()

    # Save results to CSV
data = pd.DataFrame()
data['Input'] = src
data['Output'] = ref
data['Predictions'] = generated_response1
data['BLEU'] = bleu_scores
data.to_csv(output_file, index=False)
# Calculate and print average BLEU score
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
print('generation test_cpo_atlas')
# Load the trained model and tokenizer
model_path = '/home/jihad.rbaiti/lustre/aim_neural-7he0p8agska/users/jihad.rbaiti/Work2_vf/Atlas_chat/CPO/checkpoint-10000'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("MBZUAI-Paris/Atlas-Chat-2B")

# Smoothing function for BLEU score
smoothing_function = SmoothingFunction().method1

# Load the test dataset from CSV file
dataset_path = '/home/jihad.rbaiti/Work_2/CPO/data/parallel_dataset/final_data_splitting/test.csv'
test_dataset = pd.read_csv(dataset_path)

# Extract inputs and references
inputs = test_dataset['darija'].tolist()
references = test_dataset['english'].tolist()

# Generate responses and calculate BLEU scores
bleu_scores = []
max_input_length = 128  # Get the model's max token length
src= []
ref = []
gen = []
for user_input, reference in zip(inputs, references):
    # Tokenize and ensure input length is within model limits
    inputs_encoded = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding="max_length"
    )
    
    # Add attention mask explicitly
    attention_mask = inputs_encoded['attention_mask']
    
    # Generate response
    output = model.generate(
        inputs_encoded["input_ids"],
        attention_mask=attention_mask,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Tokenize reference and generated text for BLEU
    reference_tokens = reference.split()
    generated_tokens = generated_response.split()
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing_function)
    bleu_scores.append(bleu_score)
    
    # Print for inspection
    print(f"User Input: {user_input}", '\n\n')
    print(f"Reference: {reference}", '\n\n')
    print(f"Generated: {generated_response}", '\n\n')
    print(f"BLEU Score: {bleu_score:.4f}\n", '\n\n')
    src.append(user_input)
    ref.append(reference)
    gen.append(generated_response)
data = pd.DataFrame()
data['Input'] = src
data['Output'] = ref
data['Predictions'] = gen
data['BLEU'] = bleu_scores
data.to_csv('/home/jihad.rbaiti/Work_2/CPO/data/test_cpo_data/predictions_cpo_atlas_ar.csv', index=False)

# Average BLEU score
average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU Score: {average_bleu:.4f}")
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import Dataset

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # GPU or CPU 

# Load the dataset
with open('chat_data.json', 'r') as f:
    data = json.load(f)

# Prepare the data for training
train_data = [f"Customer: {item['Customer']}\nAgent: {item['Agent']}" for item in data]

# Initialize tokenizer and model for GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos token 

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)  

# Tokenize the dataset with dynamic padding and truncation
train_encodings = tokenizer(train_data, padding=True, truncation=True, return_tensors="pt")

# Create a custom dataset class
class ChatbotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].to(device) for key, val in self.encodings.items()}  # Move batch to GPU

    def __len__(self):
        return len(self.encodings.input_ids)

dataset = ChatbotDataset(train_encodings)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,  # Larger batch size due to smaller model
    gradient_accumulation_steps=2,  # Lower accumulation to compensate for larger batch size
    learning_rate=0.005,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_steps=1000,
    save_total_limit=3,
    evaluation_strategy="steps",
    eval_steps=500,
    dataloader_pin_memory=True,
    report_to="tensorboard",
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False,  # Use mixed precision if GPU is available
)

# Initialize data collator for dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  
)

# Initialize the Trainer with GPT-2 and early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Stop if no improvement after 2 evaluations
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./trained_chatbot_model")
tokenizer.save_pretrained("./trained_chatbot_model")

# Multi-turn memory mechanism
conversation_history = []

def chatbot_response(user_input):
    global conversation_history
    
    # Append the user input to the conversation history
    conversation_history.append(f"Customer: {user_input}")
    
    # Prepare the input for the model
    input_text = "\n".join(conversation_history[-5:])  # Keep the last 5 turns for context
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)  # Move input data to GPU
    
    # Generate a response using advanced generative techniques
    output = model.generate(
        input_ids,
        max_length=200,
        num_beams=5,       # Beam search for better quality responses
        top_k=50,          # Top-k sampling for diversity
        top_p=0.95,        # Nucleus sampling to control randomness
        temperature=0.5,   # Control creativity (lower for more deterministic, higher for more creative)
        repetition_penalty=1.2,  # Penalty to reduce repetition
        early_stopping=True
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the agent's part of the response
    response_text = response.split("Agent:")[-1].strip()
    
    # Add the response to the conversation history
    conversation_history.append(f"Agent: {response_text}")
    
    return response_text

# Example usage
# user_input = "Why is my electricity bill so high?"
# print(chatbot_response(user_input))

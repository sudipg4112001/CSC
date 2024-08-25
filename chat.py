import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
model_name = "./trained_chatbot_model"  # Path to the saved model directory
model = GPT2LMHeadModel.from_pretrained(model_name, ignore_mismatched_sizes=False)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to generate a response
def generate_response(customer_input):
    # Prepare the input for the model
    input_text = f"Customer: {customer_input}\nAgent:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response from the model
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        num_beams=5,  # Increase the number of beams
        no_repeat_ngram_size=2,
        early_stopping=True,  # Now relevant with beam search
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the agent's response from the generated text
    agent_response = response.split("Agent:")[-1].strip()
    return agent_response

# Interactive chat session
def chat():
    print("Start chatting with your Electricity and Gas Bill Chatbot! (Type 'exit' to stop)")
    while True:
        customer_input = input("You: ")
        if customer_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = generate_response(customer_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()

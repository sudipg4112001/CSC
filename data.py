import json

def process_chat_data(file_path):
    # Read the input file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content_lines = f.readlines()

    parsed_data = []
    current_query = ""
    current_response = ""
    capturing_response = False

    for line in content_lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("Customer:"):
            if current_query and current_response:
                parsed_data.append({
                    "Customer": current_query,
                    "Agent": current_response
                })
            current_query = line[len("Customer:"):].strip()
            current_response = ""
            capturing_response = False
        elif line.startswith("Agent:"):
            current_response = line[len("Agent:"):].strip()
            capturing_response = True
        else:
            if capturing_response:
                current_response += " " + line

    # Append the last dialogue entry
    if current_query and current_response:
        parsed_data.append({
            "Customer": current_query,
            "Agent": current_response
        })

    return parsed_data

def write_json_output(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4)

# Paths for input and output
input_chat_file = 'customer_service_data.txt'
output_json_file = 'chat_data.json'

# Process and save the data
processed_data = process_chat_data(input_chat_file)
write_json_output(processed_data, output_json_file)

print("Data processing complete.")

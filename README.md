# GenAI Chatbot for Electricity and Gas Customer Queries

## Overview

Welcome to the GenAI Chatbot project! This repository contains a chatbot designed to handle customer queries related to electricity and gas services. Utilizing advanced GenAI technology, the chatbot provides accurate and timely responses to user inquiries, improving customer support efficiency.

## Dataset

The dataset used for training the chatbot is stored in `customer_service_data.txt`. This file contains pairs of customer inputs and agent responses. The customer queries serve as inputs, and the agent responses provide sample outputs for the chatbot.

**Structure:**
- **Customer:** [Input query]
- **Agent:** [Sample response]

## To set up the chatbot, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sudipg4112001/CSC.git

2. **Navigate into the project directory:** `cd CSC`

3. **Install dependencies:** `pip install -r requirements.txt`
4. The dataset used for training the chatbot is initially stored in `customer_service_data.txt`. This file is converted from text to JSON format using the data.py script. To execute this conversion, run:
   ```bash
   python data.py

5. The resulting JSON file(chat_data.json) is then used to train the GPT-2 model. To train the model, use the train.py script by running:
   ```bash
   python train.py
   
6. **Interact the chatbot script:** `python chat.py` (Interact using terminal)
7. **Interact the chatbot on Webpage:** `python app.py`

## To interact with the chatbot using the already trained model:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sudipg4112001/CSC.git

2. **Navigate into the project directory:** `cd CSC`

3. **Install dependencies:** `pip install -r requirements.txt`
4. **To interact with chatbot, using the already trained model:** `python chat.py`
5. **To interact with chatbot, using the already trained model through the web API:** `python app.py`

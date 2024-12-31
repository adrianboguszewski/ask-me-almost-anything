# Use an official Python image as a base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the server code into the container
COPY main.py /app/server.py

# Copy requirements file (if you have one) to install dependencies
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to log in to Hugging Face and start the server
ENTRYPOINT ["sh", "-c", "huggingface-cli login --token $HF_TOKEN && python server.py --model_name meta-llama/Llama-3.2-1B-Instruct --offensive_speech_model_name patrickquick/BERTicelli"]

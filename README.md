# Ask me almost anything

A simple project that works in client-server mode and allow a user (using client) to ask a chatbot (running on server).

## Setup

First, clone the repo and change the working directory.

```bash
git clone https://github.com/adrianboguszewski/ask-me-almost-anything.git
cd ask-me-almost-anything
```

## Server

The server part is responsible for providing chatbot interface through HTTP API. You can run it locally or with Docker.

### Run locally

Create a virtual environment, activate it and install all dependencies.

```bash
cd server

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Using llama models require you to accept the license. Please, visit [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and get the access. Then, use your [access token](https://huggingface.co/settings/tokens) to log in to Hugging Face CLI.

```bash
huggingface-cli login --token <your_token>
```

Run the following and wait until you see `Uvicorn running on http://0.0.0.0:8000`. Then, the server is ready.

```bash
python main.py
```

By default, the server uses Llama 3.2 1B model as a chatbot and BERTicelli to verify offensive language. You can change the models with the input arguments.

```bash
python main.py --model_name meta-llama/Llama-3.2-1B-Instruct --offensive_speech_model_name
```

> NOTE: For now, the only supported model for offensive speech classification is BERTicelli, as other models use different label names. Using others would require code changes.

## Client

The utilize the chatbot running on the server, you can use any HTTP client e.g. Postman. Nevertheless, the project provides a simple client also.

To use that client, once again, create a virtual environment, activate it and install client dependencies.

```bash
cd client

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Then, run the client. You should also specify the server URL if the server is not running on localhost.

```bash
python main.py --url http://localhost:8000
```

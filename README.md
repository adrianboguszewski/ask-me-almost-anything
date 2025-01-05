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

### Run in Docker

To use Docker, you need to have it installed and configured. Please follow these [instructions](https://docs.docker.com/engine/install/) to do so. Then, [enable](https://docs.docker.com/engine/install/linux-postinstall/) docker for non-root users. If you want to use Nvidia GPU inside the container, you also need to install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

The project provides a Dockerfile, so just run the following to build the image.

```bash
cd server
docker build -t chat-server .
```

Then, start the new container with:

```bash
docker run -e HF_TOKEN="<hf_token>" -p 8000:8000 --gpus all chat-server
```

The serevr is ready when you see `Uvicorn running on http://0.0.0.0:8000`.

> NOTE: Using llama models require you to accept the license. Please, visit [model page](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and get the access. Then, use your [access token](https://huggingface.co/settings/tokens) above.

### Test

There are unit tests provided for the server part. To run them and validate the server do:

```bash
cd tests/unit
PYTHONPATH=../.. python -m unittest test_server.py
```

### Benchmark

You can also benchmark the server and measure average response time for many concurrent users and different request lengths by using the provided tool. First install dependencies.

```bash
cd tests/benchmark
pip install -r ../requirements.txt
```

Then, use the tool and provided arguments.

```bash
python tool.py --url <server_url> --iterations <iterations> --num_users <users> --request_length <request>
```

where:
- `<server_url>` means the URL with port number when server is running
- `<iterations` means how many times the test will be performed (to average results)
- `<users>` means how many concurrent users should send the request
- `<request>` means the length of the request (user's questions) in characters (bytes)

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

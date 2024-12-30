from typing import Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Pipeline

device = "cpu"
speech_classifier: Pipeline | None = None
chat_tokenizer: AutoTokenizer | None = None
chat_model: AutoModelForCausalLM | None = None


class Request(BaseModel):
    question: str


class Response(BaseModel):
    answer: str


app = FastAPI()


@app.post("/answer", response_model=Response)
async def chat(request: Request):
    question = request.question

    if is_offensive_speech(question):
        raise HTTPException(status_code=400, detail="Offensive language detected in input")

    inputs = chat_tokenizer(question, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]

    tokens = chat_model.generate(**inputs, pad_token_id=chat_tokenizer.eos_token_id, max_length=512,
                        temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.25)

    response = chat_tokenizer.decode(tokens[0, input_length:], skip_special_tokens=True)

    if is_offensive_speech(response):
        raise HTTPException(status_code=400, detail="Offensive language detected in output")

    return Response(answer=response)


def is_offensive_speech(text: str) -> bool:
    result = speech_classifier(text)[0]
    return result["label"] == "OFF"


def load_models(model_name: str, offensive_speech_model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Pipeline, str]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = pipeline("text-classification", model=offensive_speech_model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto").to(dev)

    return model, tokenizer, classifier, dev


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Llama model type")
    parser.add_argument("--offensive_speech_model_name", type=str, default="patrickquick/BERTicelli", help="Model to detect offensive language")
    args = parser.parse_args()

    chat_model, chat_tokenizer, speech_classifier, device = load_models(args.model_name, args.offensive_speech_model_name)

    uvicorn.run(app, host="0.0.0.0", port=8000)

from typing import Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, Pipeline

speech_classifier: Pipeline | None = None
chat_model: Pipeline | None = None


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

    output = chat_model(question, max_length=256, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.25)[0]
    response = output["generated_text"][len(question) + 1:]

    if is_offensive_speech(response):
        raise HTTPException(status_code=400, detail="Offensive language detected in output")

    return Response(answer=response)


def is_offensive_speech(text: str) -> bool:
    result = speech_classifier(text)[0]
    return result["label"] == "OFF"


def load_models(model_name: str, offensive_speech_model_name: str) -> Tuple[Pipeline, Pipeline]:
    classifier = pipeline("text-classification", model=offensive_speech_model_name)
    chatbot = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
    return chatbot, classifier


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Llama model type")
    parser.add_argument("--offensive_speech_model_name", type=str, default="patrickquick/BERTicelli",
                        choices=["patrickquick/BERTicelli"], help="Model to detect offensive language")
    args = parser.parse_args()

    chat_model, speech_classifier = load_models(args.model_name, args.offensive_speech_model_name)

    uvicorn.run(app, host="0.0.0.0", port=8000)

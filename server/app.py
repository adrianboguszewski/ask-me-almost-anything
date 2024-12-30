import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, pipeline, Pipeline

speech_classifier: Pipeline | None = None
chat_tokenizer: AutoTokenizer | None = None
chat_model: AutoModelForCausalLM | None = None


def load_models(model_name: str, offensive_speech_model_name: str, device: str) -> None:
    global speech_classifier, chat_tokenizer, chat_model
    speech_classifier = pipeline("text-classification", model=offensive_speech_model_name)

    chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to(device)


def is_offensive_speech(text: str) -> bool:
    result = speech_classifier(text)[0]
    return result["label"] == "OFF"


def run(model_name: str, offensive_speech_model_name: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    load_models(model_name, offensive_speech_model_name, device)

    streamer = TextStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
    while True:
        print("=== Ready! ===")
        question = input()
        if is_offensive_speech(question):
            print("Please don't use offensive language")
            continue

        inputs = chat_tokenizer(question, return_tensors="pt").to(device)

        chat_model.generate(**inputs, streamer=streamer, pad_token_id=chat_tokenizer.eos_token_id, max_length=512,
                            temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Llama model type")
    parser.add_argument("--offensive_speech_model_name", type=str, default="patrickquick/BERTicelli", help="Model to detect offensive language")

    args = parser.parse_args()
    run(args.model_name, args.offensive_speech_model_name)

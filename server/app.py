import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def run(model_name: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to(device)

    while True:
        print("=== Ready! ===")
        question = input()

        inputs = tokenizer(question, return_tensors="pt").to(device)

        model.generate(**inputs, streamer=streamer, pad_token_id=tokenizer.eos_token_id, max_length=512,
                       temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Llama model type")

    args = parser.parse_args()
    run(args.model_name)

import argparse

import requests


def run(url: str):
    while True:
        question = input("Enter your question: ")
        data = {"question": question}

        try:
            response = requests.post(url + "/answer", json=data)

            if response.status_code == 200:
                print("Response from server:", response.json()["answer"])
            else:
                print(f"Error: {response.status_code}, {response.json()["detail"]}")
        except requests.exceptions.RequestException as e:
            print("An error occurred:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()

    run(args.url)

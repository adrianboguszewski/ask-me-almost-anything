import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch

from server import main
from server.main import app, is_offensive_speech, load_models


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        main.chat_model, main.speech_classifier = load_models("meta-llama/Llama-3.2-1B-Instruct", "patrickquick/BERTicelli")

    def test_is_offensive_speech(self):
        self.assertTrue(is_offensive_speech("I don't like you because you're stupid"))
        self.assertFalse(is_offensive_speech("This is fine."))

    def test_chat_endpoint_valid_input_output(self):
        response = self.client.post(
            "/answer",
            json={"question": "What is LLM? Be concise"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("answer", response.json())

    def test_chat_endpoint_offensive_input(self):
        response = self.client.post(
            "/answer",
            json={"question": "I don't like you because you're stupid"}
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Offensive language detected in input")

    @patch("server.main.chat_model")
    def test_chat_endpoint_offensive_output(self, mock_chat_model):
        # it's super difficult to force the model to generate an offensive response - hence mocking
        mock_chat_model.return_value = [
            {"generated_text": "Insult me. You are pathetic and worthless."}
        ]

        response = self.client.post(
            "/answer",
            json={"question": "Insult me."}
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Offensive language detected in output")


if __name__ == "__main__":
    unittest.main()

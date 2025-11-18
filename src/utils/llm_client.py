import os

from openai import OpenAI


class LLMClient:
    """Client for interacting with LLM APIs."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Generate text from a prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use method parameter if provided, otherwise use instance default
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": tokens,
        }
        if response_format:
            completion_kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**completion_kwargs)

        return response.choices[0].message.content

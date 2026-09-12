"""Transformers (Hugging Face) inference backend for Alpha-R1."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import SYSTEM_PROMPT


class HFBackend:
    """Standard ``from_pretrained`` loading; works with hub ids and local paths.

    ``FinStep/Alpha-R1`` is public; if you point ``model_id`` at a gated or
    private repo instead, authenticate with ``HF_TOKEN`` or ``hf auth login``.
    """

    def __init__(self, model_id: str, temperature: float = 0.0, top_p: float = 0.7,
                 max_tokens: int = 12288, device_map: str = "auto",
                 torch_dtype: str = "bfloat16"):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, torch_dtype),
            device_map=device_map,
        )
        self.do_sample = temperature > 0
        self.temperature = temperature or 1.0
        self.top_p = top_p

    def generate(self, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, **self._gen_kwargs())
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _gen_kwargs(self) -> dict:
        kwargs = {"max_new_tokens": self.max_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            kwargs["temperature"] = self.temperature
            kwargs["top_p"] = self.top_p
        return kwargs

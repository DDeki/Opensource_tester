import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal smoke test for google/gemma-3-27b-it (HF download + single prompt)."
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-27b-it",
        help="Hugging Face model id (default: google/gemma-3-27b-it).",
    )
    parser.add_argument(
        "--prompt",
        default="Zdravo!",
        help='User prompt to send (default: "Zdravo!").',
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    try:
        from huggingface_hub import constants as hf_c

        print(f"HF_HOME={hf_c.HF_HOME}")
        print(f"HF_HUB_CACHE={hf_c.HF_HUB_CACHE}")
    except Exception:
        pass

    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        try:
            repo_root = Path(__file__).resolve().parents[2]
            repo_root_str = str(repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)

            from config.secrets import HUGGING_FACE_TOKEN  # type: ignore

            if isinstance(HUGGING_FACE_TOKEN, str) and HUGGING_FACE_TOKEN.strip():
                token = HUGGING_FACE_TOKEN.strip()
        except Exception:
            pass

    if not token:
        print(
            "Missing HF token. Set HUGGINGFACE_HUB_TOKEN (or HF_TOKEN) or define HUGGING_FACE_TOKEN in config/secrets.py "
            "and make sure you've accepted the Gemma license on HF."
        )
        return 2

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else None

    model = (
        Gemma3ForConditionalGeneration.from_pretrained(
            args.model, device_map=device_map, torch_dtype=dtype, token=token
        )
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.model, token=token)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": args.prompt}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device, dtype=dtype)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    generated = generation[0][input_len:]
    decoded = processor.decode(generated, skip_special_tokens=True)
    print(decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


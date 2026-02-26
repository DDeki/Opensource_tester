import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _maybe_load_hf_token() -> str | None:
    token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if token:
        return token

    try:
        # Repo root is 4 levels above this file:
        # llm/mistral-7b/instruct-v0.3/smoke_test.py
        repo_root = Path(__file__).resolve().parents[4]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        from config.secrets import HUGGING_FACE_TOKEN  # type: ignore

        if isinstance(HUGGING_FACE_TOKEN, str) and HUGGING_FACE_TOKEN.strip():
            return HUGGING_FACE_TOKEN.strip()
    except Exception:
        pass

    return None


def _input_device(model: torch.nn.Module) -> torch.device:
    m = getattr(model, "hf_device_map", None)
    if isinstance(m, dict):
        vals = set(m.values())
        for d in vals:
            if isinstance(d, str) and d.startswith("cuda"):
                return torch.device(d)
        if "mps" in vals:
            return torch.device("mps")
        return torch.device("cpu")

    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimal smoke test for mistralai/Mistral-7B-Instruct-v0.3 (HF download + single prompt)."
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="Hugging Face model id (default: mistralai/Mistral-7B-Instruct-v0.3).",
    )
    parser.add_argument(
        "--prompt",
        default="Zdravo!",
        help='User prompt to send (default: "Zdravo!").',
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--offload-folder",
        default=".hf_offload",
        help='Disk offload folder for CPU runs (default: ".hf_offload"). Use "" to disable.',
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import constants as hf_c

        print(f"HF_HOME={hf_c.HF_HOME}")
        print(f"HF_HUB_CACHE={hf_c.HF_HUB_CACHE}")
    except Exception:
        pass

    token = _maybe_load_hf_token()

    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    try:
        model_kwargs: dict = {
            "device_map": "auto",
            "dtype": dtype,
        }
        if token:
            model_kwargs["token"] = token

        if not use_cuda:
            model_kwargs["low_cpu_mem_usage"] = True
            if args.offload_folder:
                model_kwargs["offload_folder"] = args.offload_folder
                model_kwargs["offload_state_dict"] = True

        tok_kwargs: dict = {}
        if token:
            tok_kwargs["token"] = token

        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
            tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
        except Exception as e:
            # If a proxy blocks HF network calls, retry using only local cache.
            msg = repr(e).lower()
            if "proxyerror" in msg or "proxy error" in msg:
                model_kwargs["local_files_only"] = True
                tok_kwargs["local_files_only"] = True
                model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
                tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
            else:
                raise
    except Exception as e:
        chain = []
        cur = e
        seen = set()
        while cur is not None and id(cur) not in seen and len(chain) < 10:
            seen.add(id(cur))
            chain.append(repr(cur))
            cur = cur.__cause__ or cur.__context__

        msg = "\n".join(chain).lower()
        if "proxyerror" in msg or "proxy error" in msg:
            print(
                "HF download failed due to a proxy error.\n"
                "- Rešenje: proveri da li imaš setovane env var: HTTP_PROXY / HTTPS_PROXY / ALL_PROXY.\n"
                "- Pokušaj: `unset HTTP_PROXY HTTPS_PROXY ALL_PROXY` pa pokreni ponovo."
            )
            return 4
        if "gated repo" in msg or "gatedrepoerror" in msg or "403" in msg:
            print(
                "HF access error (gated model).\n"
                f"- Model: {args.model}\n"
                "- Rešenje: prihvati uslove / traži pristup na stranici modela i koristi token sa tog naloga."
            )
            return 3
        raise

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.prompt},
    ]

    if getattr(tokenizer, "chat_template", None):
        chat = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(chat, torch.Tensor):
            input_ids = chat
            attention_mask = None
        else:
            input_ids = chat["input_ids"]
            attention_mask = chat.get("attention_mask")
    else:
        enc = tokenizer(args.prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

    dev = _input_device(model)
    input_ids = input_ids.to(dev)
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated = out[0][input_ids.shape[-1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


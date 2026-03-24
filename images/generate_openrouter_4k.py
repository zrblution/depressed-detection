#!/usr/bin/env python3
"""Generate 4K images with OpenRouter's Gemini image model.

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python /home/tos_lx/Depression-Detection/code/images/generate_openrouter_4k.py \
    --prompt "A cinematic rainy street in Tokyo at night, ultra detailed"

  python /home/tos_lx/Depression-Detection/code/images/generate_openrouter_4k.py \
    --prompt-file /home/tos_lx/Depression-Detection/code/images/diagram_edit_prompt.txt \
    --input-image /home/tos_lx/Depression-Detection/code/images/candidate_4.png \
    --input-image /home/tos_lx/Depression-Detection/code/images/candidate_3.png

Notes:
  - The script requests OpenRouter's 4K image_size bucket.
  - Actual pixel dimensions depend on the chosen aspect ratio.
  - Local input images are embedded as base64 data URLs for image editing.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import struct
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-flash-image-preview"
DEFAULT_ASPECT_RATIO = "21:9"
DEFAULT_IMAGE_SIZE = "4K"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 4K image through OpenRouter and save it locally."
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        help="Text prompt sent to the model.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        default="",
        help="Path to a UTF-8 text file containing the prompt.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default=DEFAULT_ASPECT_RATIO,
        help=f'Image aspect ratio. Defaults to "{DEFAULT_ASPECT_RATIO}".',
    )
    parser.add_argument(
        "--image-size",
        default=DEFAULT_IMAGE_SIZE,
        choices=["0.5K", "1K", "2K", "4K"],
        help=f'Image size bucket. Defaults to "{DEFAULT_IMAGE_SIZE}".',
    )
    parser.add_argument(
        "--input-image",
        action="append",
        default=[],
        help=(
            "Optional local input image for image editing. Repeat this flag to attach "
            "multiple images. The file stem is used as the image label in the prompt."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Optional output file path. If multiple images are returned, "
            "numeric suffixes are added automatically."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds. Defaults to 300.",
    )
    return parser.parse_args()


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt.strip()

    prompt_path = Path(args.prompt_file).expanduser().resolve()
    if not prompt_path.is_file():
        raise RuntimeError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def encode_local_image_as_data_url(image_path: Path) -> tuple[str, str]:
    if not image_path.is_file():
        raise RuntimeError(f"Input image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None or not mime_type.startswith("image/"):
        raise RuntimeError(f"Unsupported image type for input file: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return image_path.stem, f"data:{mime_type};base64,{encoded}"


def build_user_content(prompt_text: str, input_images: list[str]) -> str | list[dict[str, Any]]:
    if not input_images:
        return prompt_text

    content: list[dict[str, Any]] = []
    image_blocks: list[dict[str, Any]] = []
    attachment_lines = [
        "",
        "Attached input images are provided below in this exact order.",
        "Use the following identifiers when interpreting the prompt:",
    ]

    for index, image_spec in enumerate(input_images, start=1):
        image_path = Path(image_spec).expanduser().resolve()
        label, data_url = encode_local_image_as_data_url(image_path)
        attachment_lines.append(f"{index}. {label}: attached image #{index}")
        image_blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url,
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": prompt_text + "\n".join(attachment_lines),
        }
    )
    content.extend(image_blocks)
    return content


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    prompt_text = read_prompt(args)
    return {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": build_user_content(prompt_text, args.input_image),
            }
        ],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": args.aspect_ratio,
            "image_size": args.image_size,
        },
    }


def send_request(payload: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc


def extract_image_urls(result: dict[str, Any]) -> list[str]:
    image_urls: list[str] = []

    for choice in result.get("choices", []):
        message = choice.get("message", {})

        for image in message.get("images", []):
            image_url = ((image or {}).get("image_url") or {}).get("url")
            if image_url:
                image_urls.append(image_url)

        content = message.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "image_url":
                    continue
                image_url = (block.get("image_url") or {}).get("url")
                if image_url:
                    image_urls.append(image_url)

    return image_urls


def decode_image_payload(image_url: str, timeout: int) -> tuple[str, str, bytes]:
    if image_url.startswith("data:"):
        header, encoded = image_url.split(",", 1)
        mime_type = header[5:].split(";", 1)[0]
        extension = mimetypes.guess_extension(mime_type) or ".bin"
        return mime_type, extension, base64.b64decode(encoded)

    if image_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(image_url, timeout=timeout) as response:
            data = response.read()
            mime_type = response.headers.get_content_type() or "application/octet-stream"
            extension = mimetypes.guess_extension(mime_type) or ".bin"
            return mime_type, extension, data

    raise RuntimeError("Unsupported image payload returned by OpenRouter.")


def infer_dimensions(image_bytes: bytes, mime_type: str) -> tuple[int, int] | None:
    if mime_type == "image/png" and len(image_bytes) >= 24 and image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", image_bytes[16:24])
        return width, height

    if mime_type in {"image/jpeg", "image/jpg"}:
        index = 2
        length = len(image_bytes)
        while index + 9 < length:
            if image_bytes[index] != 0xFF:
                index += 1
                continue
            marker = image_bytes[index + 1]
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                height = struct.unpack(">H", image_bytes[index + 5 : index + 7])[0]
                width = struct.unpack(">H", image_bytes[index + 7 : index + 9])[0]
                return width, height
            block_length = struct.unpack(">H", image_bytes[index + 2 : index + 4])[0]
            index += 2 + block_length

    return None


def resolve_output_base(output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (Path(__file__).resolve().parent / f"generated_{timestamp}").resolve()


def save_images(image_urls: list[str], output_arg: str, timeout: int) -> list[Path]:
    output_base = resolve_output_base(output_arg)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    single_image = len(image_urls) == 1

    for index, image_url in enumerate(image_urls, start=1):
        mime_type, extension, image_bytes = decode_image_payload(image_url, timeout)

        if output_base.suffix and single_image:
            output_path = output_base
        elif output_base.suffix:
            output_path = output_base.with_name(
                f"{output_base.stem}_{index}{output_base.suffix}"
            )
        elif single_image:
            output_path = output_base.with_suffix(extension)
        else:
            output_path = output_base.with_name(
                f"{output_base.name}_{index}"
            ).with_suffix(extension)

        output_path.write_bytes(image_bytes)
        saved_paths.append(output_path)

        dimensions = infer_dimensions(image_bytes, mime_type)
        if dimensions is None:
            print(f"Saved {output_path} ({mime_type})")
        else:
            width, height = dimensions
            print(f"Saved {output_path} ({mime_type}, {width}x{height})")

    return saved_paths


def main() -> int:
    args = parse_args()

    if not args.api_key:
        print(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 2

    try:
        payload = build_payload(args)
        result = send_request(payload, args.api_key, args.timeout)
        image_urls = extract_image_urls(result)
        if not image_urls:
            raise RuntimeError(
                "No images were returned. Full response:\n"
                + json.dumps(result, indent=2, ensure_ascii=False)
            )
        save_images(image_urls, args.output, args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

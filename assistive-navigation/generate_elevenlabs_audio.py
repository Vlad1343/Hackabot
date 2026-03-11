"""
HACKABOT ElevenLabs Audio Generation.
Generates WAV files using ElevenLabs API; includes error handling for API failures.
Set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env or pass as arguments.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

try:
    import requests
    from dotenv import load_dotenv
except ImportError as e:
    print(f"[Error] Missing dependency: {e}")
    print("Install: pip install requests python-dotenv")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent / "audio"
INDOORS = BASE_DIR / "indoors"
OUTDOORS = BASE_DIR / "outdoors"
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


def phrase_catalog() -> dict[str, dict[str, str]]:
    from generate_pyttsx3_audio import phrase_catalog as offline_catalog
    return offline_catalog()


def synthesize_wav(api_key: str, voice_id: str, text: str, model_id: str) -> bytes:
    url = API_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.75},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"ElevenLabs API error {response.status_code}: {response.text[:300]}")
    return response.content


def generate_mode(
    *,
    api_key: str,
    voice_id: str,
    model_id: str,
    out_dir: Path,
    items: dict[str, str],
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, text in items.items():
        target = out_dir / f"{key}.wav"
        if target.exists() and not overwrite:
            print(f"[Skip] {target.name}")
            continue

        try:
            print(f"[Generate] {target.name}")
            audio_bytes = synthesize_wav(api_key=api_key, voice_id=voice_id, text=text, model_id=model_id)
            target.write_bytes(audio_bytes)
        except requests.exceptions.RequestException as exc:
            print(f"[Error] API request failed for {target.name}: {exc}")
            traceback.print_exc()
        except RuntimeError as exc:
            print(f"[Error] {exc}")
        except Exception as exc:
            print(f"[Error] Failed to generate {target.name}: {exc}")
            traceback.print_exc()
        else:
            time.sleep(0.3)  # rate limit courtesy


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate WAV files via ElevenLabs API")
    parser.add_argument("--mode", choices=["indoor", "outdoor", "all"], default="all")
    parser.add_argument("--voice-id", default=os.getenv("ELEVENLABS_VOICE_ID", ""))
    parser.add_argument("--model-id", default=os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"))
    parser.add_argument("--api-key", default=os.getenv("ELEVENLABS_API_KEY", ""))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("ELEVENLABS_API_KEY missing. Set it in .env or pass --api-key")
        sys.exit(1)
    if not args.voice_id:
        print("ELEVENLABS_VOICE_ID missing. Set it in .env or pass --voice-id")
        sys.exit(1)

    catalog = phrase_catalog()
    if args.mode in ("indoor", "all"):
        generate_mode(
            api_key=args.api_key,
            voice_id=args.voice_id,
            model_id=args.model_id,
            out_dir=INDOORS,
            items=catalog["indoor"],
            overwrite=args.overwrite,
        )
    if args.mode in ("outdoor", "all"):
        generate_mode(
            api_key=args.api_key,
            voice_id=args.voice_id,
            model_id=args.model_id,
            out_dir=OUTDOORS,
            items=catalog["outdoor"],
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

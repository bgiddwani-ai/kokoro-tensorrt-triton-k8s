"""Check matched eager/served audio, with signed ASR WER/CER deltas."""

import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download

from .settings import SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-dir", required=True)
    parser.add_argument("--triton-dir", required=True)
    parser.add_argument("--corpus", default="benchmarks/texts.json")
    parser.add_argument("--out", default="artifacts/quality.json")
    parser.add_argument("--max-wer-delta", type=float, default=0.0)
    parser.add_argument("--max-cer-delta", type=float, default=0.0)
    parser.add_argument("--max-duration-delta-percent", type=float, default=1.0)
    args = parser.parse_args()
    revision = "d1d751a5f8271d482d14ca55d9e2deeebbae577f"
    model = WhisperModel(
        snapshot_download("Systran/faster-whisper-small.en", revision=revision),
        device="cpu",
        compute_type="int8",
    )
    normalize = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )
    records, references, eager_texts, triton_texts = [], [], [], []
    for case in json.loads(Path(args.corpus).read_text()):
        record = {"case": case["id"], "reference": case["text"]}
        for backend, directory in (
            ("eager", args.eager_dir),
            ("triton", args.triton_dir),
        ):
            files = sorted(
                Path(directory).glob(f"{case['id']:02d}_*.wav"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )
            if not files:
                raise ValueError(f"missing {backend} audio for case {case['id']}")
            chunks = []
            for path in files:
                audio, rate = sf.read(path, dtype="float32")
                if (
                    rate != SAMPLE_RATE
                    or audio.ndim != 1
                    or not audio.size
                    or not np.isfinite(audio).all()
                ):
                    raise ValueError(f"invalid audio: {path.name}")
                chunks.append(audio)
            output = np.concatenate(chunks)
            # Whisper expects 16 kHz arrays; file input handles sample-rate conversion.
            segments = []
            for path in files:
                decoded, _ = model.transcribe(
                    str(path), language="en", beam_size=5, vad_filter=False
                )
                segments.extend(segment.text.strip() for segment in decoded)
            record[backend] = {
                "seconds": output.size / SAMPLE_RATE,
                "peak": float(np.max(np.abs(output))),
                "rms": float(np.sqrt(np.mean(output**2))),
                "transcript": " ".join(segments),
            }
        record["duration_delta_percent"] = (
            record["triton"]["seconds"] / record["eager"]["seconds"] - 1
        ) * 100
        references.append(normalize(case["text"]))
        eager_texts.append(normalize(record["eager"]["transcript"]))
        triton_texts.append(normalize(record["triton"]["transcript"]))
        records.append(record)
    summary = {
        "asr_model_revision": revision,
        "eager_wer": jiwer.wer(references, eager_texts),
        "triton_wer": jiwer.wer(references, triton_texts),
        "eager_cer": jiwer.cer(references, eager_texts),
        "triton_cer": jiwer.cer(references, triton_texts),
    }
    summary["wer_delta"] = summary["triton_wer"] - summary["eager_wer"]
    summary["cer_delta"] = summary["triton_cer"] - summary["eager_cer"]
    summary["max_abs_duration_delta_percent"] = max(
        abs(row["duration_delta_percent"]) for row in records
    )
    summary["passed"] = (
        summary["wer_delta"] <= args.max_wer_delta + 1e-12
        and summary["cer_delta"] <= args.max_cer_delta + 1e-12
        and summary["max_abs_duration_delta_percent"] <= args.max_duration_delta_percent
    )
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"summary": summary, "cases": records}, indent=2, allow_nan=False)
        + "\n"
    )
    print(json.dumps(summary, allow_nan=False))
    if not summary["passed"]:
        raise SystemExit("audio quality gate failed; inspect the report")


if __name__ == "__main__":
    main()

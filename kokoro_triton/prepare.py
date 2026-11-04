"""Download public pinned assets and prepare portable ONNX partitions."""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import graphs
from .settings import ONNX_REPOSITORY, ONNX_REVISION, ONNX_SHA256, Settings


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(settings):
    settings.data.mkdir(parents=True, exist_ok=True)
    with open(settings.data / ".prepare.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        settings.models.mkdir(exist_ok=True)
        stamp = settings.models / "prepared.json"
        expected = {
            "upstream_revision": ONNX_REVISION,
            "upstream_sha256": ONNX_SHA256,
            "transform_sha256": sha256(Path(graphs.__file__)),
        }
        names = [f"hybrid-{stage}.onnx" for stage in ("predictor", "decoder")]
        if stamp.exists():
            previous = json.loads(stamp.read_text())
            if all(previous.get(k) == v for k, v in expected.items()) and all(
                (settings.models / name).exists()
                and sha256(settings.models / name)
                == previous.get("files", {}).get(name)
                for name in names
            ):
                return previous
        source = hf_hub_download(
            ONNX_REPOSITORY, "kokoro-82m-v1.0.onnx", revision=ONNX_REVISION
        )
        if sha256(source) != ONNX_SHA256:
            raise RuntimeError("upstream ONNX SHA256 does not match the pinned model")
        license_path = hf_hub_download(
            ONNX_REPOSITORY, "LICENSE", revision=ONNX_REVISION
        )
        with tempfile.TemporaryDirectory(
            prefix="prepare-", dir=settings.data
        ) as temporary:
            stage = Path(temporary)
            partitions = graphs.split_model(graphs.make_portable(source))
            for name in ("predictor", "decoder"):
                graphs.infer_partition_shapes(
                    partitions.pop(name), stage / f"hybrid-{name}.onnx"
                )
            expected["files"] = {name: sha256(stage / name) for name in names}
            for name in names:
                os.replace(stage / name, settings.models / name)
        shutil.copyfile(license_path, settings.models / "LICENSE-upstream")
        temporary_stamp = stamp.with_suffix(".tmp")
        temporary_stamp.write_text(json.dumps(expected, indent=2) + "\n")
        os.replace(temporary_stamp, stamp)
        return expected


if __name__ == "__main__":
    prepare(Settings.from_env())

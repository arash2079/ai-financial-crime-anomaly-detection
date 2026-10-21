"""Download the pinned source, verifying SHA-256 before the final file appears."""

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from .data import DEFAULT_INPUT, SOURCE_SHA256, SOURCE_URL, sha256_file


def download_data(output=DEFAULT_INPUT):
    output = Path(output)
    if output.exists():
        if sha256_file(output) != SOURCE_SHA256:
            raise ValueError("existing file does not match pinned SHA-256; move it before downloading")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    candidate = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".download", delete=False) as handle:
            candidate = Path(handle.name)
            digest = hashlib.sha256()
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "financial-crime-project/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != SOURCE_SHA256:
            raise ValueError("download SHA-256 does not match pinned dataset version 8")
        os.replace(candidate, output)
        return output
    finally:
        if candidate is not None:
            candidate.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_INPUT)
    args = parser.parse_args()
    try:
        output = download_data(args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Verified source file: {output}")


if __name__ == "__main__":
    main()

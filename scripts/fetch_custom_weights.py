"""Download and validate official NLSY97 custom weights for a prepared ID list.

Run 01_clean_panel.py --prepare-weights first. This helper submits only public
NLSY97 respondent IDs to the official service. It retains the returned package,
validates the positive-weight sample, and installs the weights for stage 1.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://www.nlsinfo.org/nlsinfo-services-api/weights/NLSY97/download"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path,
                        default=ROOT / "data/processed/weight_request/respondent_ids.txt")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/raw/weights_receipt_confirmed")
    parser.add_argument("--install-dir", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    id_bytes = args.ids.read_bytes()
    entries = args.ids.read_text().splitlines()
    if not entries or not all(entry.strip().isdigit() and int(entry) > 0 for entry in entries):
        raise ValueError("The request must contain one positive integer respondent ID per line.")
    ids = [int(entry) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in the weight request.")
    requested = set(ids)
    body = urllib.parse.urlencode({"id-box": "\n".join(map(str, sorted(ids))),
                                   "USE_OR_NOT_AND": "YES", "op": "Download"}).encode()
    request = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "NLSY97-class-research-replication/1.0",
    })
    print(f"Requesting official weights for {len(ids):,} respondent IDs.", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type", "")
        disposition = response.headers.get("Content-Disposition", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not zipfile.is_zipfile(io.BytesIO(payload)):
        diagnostic = args.output_dir / "service_response.bin"
        diagnostic.write_bytes(payload)
        raise ValueError(f"Service did not return a ZIP ({content_type}); response saved to {diagnostic}.")
    (args.output_dir / "official_custom_weights.zip").write_bytes(payload)

    files = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as package:
        for member in package.infolist():
            if member.is_dir():
                continue
            # Store basenames only. Do not execute or extract arbitrary paths.
            basename = Path(member.filename).name
            if basename in files:
                raise ValueError("Duplicate basenames in the service archive.")
            if member.file_size > 5_000_000:
                raise ValueError("Unexpectedly large file in the service archive.")
            files[basename] = package.read(member)
    candidates = [name for name in files if name.endswith(".dat") and "ListOfIdsToWeight" not in name]
    if len(candidates) != 1:
        raise ValueError("Could not identify a unique weight data file in the package.")
    data_name = candidates[0]
    weights = {}
    for line in files[data_name].decode().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError("Weight data must contain two columns.")
        identifier, value = int(parts[0]), float(parts[1])
        if identifier in weights or not (0 <= value < float("inf")):
            raise ValueError("Duplicate IDs or invalid weights in service output.")
        weights[identifier] = value
    positive = {identifier for identifier, value in weights.items() if value > 0}
    if positive != requested:
        raise ValueError("Returned positive-weight IDs differ from the request: "
                         f"{len(requested - positive)} requested IDs lack positive weights; "
                         f"{len(positive - requested)} other IDs have positive weights.")
    id_files = [name for name in files if "ListOfIdsToWeight" in name]
    if id_files:
        returned = {int(value) for value in files[id_files[0]].decode().split()}
        if returned != requested:
            raise ValueError("The service's echoed ID list differs from the submitted list.")
    readmes = [name for name in files if name.endswith(".readme")]
    if len(readmes) != 1:
        raise ValueError("Expected one official README documenting the weight request.")

    # Preserve the exact official response and previous installed weights.
    for name, data in files.items():
        (args.output_dir / name).write_bytes(data)
    (args.output_dir / "submitted_respondent_ids.txt").write_bytes(id_bytes)
    args.install_dir.mkdir(parents=True, exist_ok=True)
    previous = args.install_dir / "custom_weights.dat"
    if previous.exists() and previous.read_bytes() != files[data_name]:
        backup = args.install_dir / "weights_previous" / digest(previous.read_bytes())[:12]
        backup.mkdir(parents=True, exist_ok=True)
        for filename in ["custom_weights.dat", "custom_weights.readme", "custom_weights_metadata.json"]:
            source = args.install_dir / filename
            if source.exists():
                shutil.copy2(source, backup / filename)
    (args.install_dir / "custom_weights.dat").write_bytes(files[data_name])
    (args.install_dir / "custom_weights.readme").write_bytes(files[readmes[0]])
    manifest = {
        "source": ENDPOINT,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_mode": "Weight IDs; no survey-year fields submitted",
        "request_file": str(args.ids.resolve()),
        "request_sha256": digest(id_bytes),
        "requested_respondents": len(requested),
        "positive_weight_respondents": len(positive),
        "zero_weight_rows_outside_sample": len(weights) - len(positive),
        "weight_units": "Official stored values; two implied decimal places",
        "weight_min_raw": min(weights[identifier] for identifier in requested),
        "weight_max_raw": max(weights[identifier] for identifier in requested),
        "download_content_type": content_type,
        "download_content_disposition": disposition,
        "zip_sha256": digest(payload),
        "installed_weight_sha256": digest(files[data_name]),
        "package_directory": str(args.output_dir.resolve()),
    }
    text = json.dumps(manifest, indent=2) + "\n"
    (args.output_dir / "download_manifest.json").write_text(text)
    (args.install_dir / "custom_weights_metadata.json").write_text(text)
    print(f"Validated and installed {len(positive):,} positive custom weights.")
    print(f"Official package retained in {args.output_dir}")


if __name__ == "__main__":
    main()

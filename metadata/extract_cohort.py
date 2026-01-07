#!/usr/bin/env python3.11
"""
extract_cohort.py

Deterministic extraction of a 40-subject EEG cohort from OpenNeuro ds005385.

Usage:
  python3.11 extract_cohort.py /path/to/ds005385

Outputs:
  - cohort_40_manifest.csv
  - subject_summary.csv

Requirements:
  - Python 3.11+
  - ds005385 cloned locally (for example via datalad)
  - EDF files present as real binaries
"""

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def read_edf_header(filepath: Path) -> dict:
    """
    Extract basic metadata from an EDF header.

    Notes:
      - EDF fixed header is 256 bytes.
      - Number of signals is stored in bytes 252:256.
      - Number of data records is stored in bytes 236:244.
      - Duration of each data record is stored in bytes 244:252.
      - Per-signal header fields are 256 bytes each (not 216).
      - The "number of samples per data record" field is 8 bytes per signal.

    This function returns:
      duration_seconds, sampling_rate_hz, channels_count
    """
    metadata = {
        "duration": "",
        "sampling_rate": "",
        "channels_count": "",
    }

    try:
        with filepath.open("rb") as f:
            header = f.read(256)
            if len(header) < 256:
                return metadata

            num_signals_str = header[252:256].decode("ascii", errors="ignore").strip()
            if not num_signals_str:
                return metadata
            num_signals = int(num_signals_str)
            metadata["channels_count"] = str(num_signals)

            num_records_str = header[236:244].decode("ascii", errors="ignore").strip()
            record_duration_str = header[244:252].decode("ascii", errors="ignore").strip()

            num_records = int(num_records_str) if num_records_str else 0
            record_duration = float(record_duration_str) if record_duration_str else 0.0

            if num_records > 0 and record_duration > 0:
                metadata["duration"] = str(num_records * record_duration)

            # Skip fixed header + per-signal headers (256 bytes each)
            # Then read the "number of samples per data record" for the first signal (8 bytes).
            per_signal_header_bytes = 256
            f.seek(256 + num_signals * per_signal_header_bytes)

            samples_str = f.read(8).decode("ascii", errors="ignore").strip()
            if samples_str and record_duration > 0:
                samples_per_record = int(samples_str)
                metadata["sampling_rate"] = str(samples_per_record / record_duration)

    except Exception as e:
        print(f"Warning: Could not read EDF header for {filepath}: {e}")

    return metadata


def parse_bids_parts(filename: str) -> dict:
    """
    Parse BIDS-like parts from a filename such as:
      sub-XXX_ses-X_task-XXXX_acq-pre_eeg.edf

    Returns:
      subject_id, session_id, task, acq
    """
    name = filename
    if name.lower().endswith(".edf"):
        name = name[:-4]

    parts = name.split("_")
    out = {"subject_id": "", "session_id": "", "task": "", "acq": ""}

    for part in parts:
        if part.startswith("sub-"):
            out["subject_id"] = part
        elif part.startswith("ses-"):
            out["session_id"] = part
        elif part.startswith("task-"):
            out["task"] = part.replace("task-", "")
        elif part.startswith("acq-"):
            out["acq"] = part.replace("acq-", "")

    return out


def main(dataset_path: str) -> None:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print("=" * 80)
    print("Extracting 40-subject cohort from ds005385")
    print("=" * 80)

    # Load participants.tsv
    print("\n[1/6] Loading participants.tsv...")
    participants_file = dataset_path / "participants.tsv"
    if not participants_file.exists():
        print(f"Error: participants.tsv not found at {participants_file}")
        sys.exit(1)

    participants: dict[str, dict] = {}
    with participants_file.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row.get("participant_id", "").strip()
            if pid:
                participants[pid] = row

    print(f"Loaded {len(participants)} participants")

    # Find subjects with both sessions
    print("\n[2/6] Identifying subjects with both sessions...")
    subjects_with_both: list[str] = []
    for subj_id, data in participants.items():
        if data.get("session1") == "yes" and data.get("session2") == "yes":
            subjects_with_both.append(subj_id)

    print(f"Found {len(subjects_with_both)} subjects with both sessions")

    # Deterministic cohort selection
    cohort_subjects = sorted(subjects_with_both)[:40]
    print(f"Selected {len(cohort_subjects)} subjects for cohort")

    # Scan for EDF files for cohort subjects
    print("\n[3/6] Scanning for EDF files...")
    edf_files: list[Path] = []

    for subject_dir in dataset_path.glob("sub-*"):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        if subject_id not in cohort_subjects:
            continue

        for edf_file in subject_dir.rglob("*.edf"):
            if edf_file.is_file():
                edf_files.append(edf_file)

    print(f"Found {len(edf_files)} EDF files for cohort")

    # Build manifest
    print("\n[4/6] Building manifest...")
    manifest_records: list[dict] = []

    for edf_path in edf_files:
        filename = edf_path.name
        parsed = parse_bids_parts(filename)

        subject_id = parsed["subject_id"]
        session_id = parsed["session_id"]
        task = parsed["task"]
        acq = parsed["acq"]

        if not subject_id or not session_id:
            print(f"Warning: Could not parse subject/session from {filename}")
            continue

        # Pre/post from acquisition label
        if acq == "pre":
            pre_or_post = "pre"
            treatment_label = "baseline"
        elif acq == "post":
            pre_or_post = "post"
            treatment_label = "post_treatment"
        else:
            pre_or_post = ""
            treatment_label = ""

        edf_sha256 = compute_sha256(edf_path)
        edf_bytes = edf_path.stat().st_size
        edf_metadata = read_edf_header(edf_path)
        edf_relpath = str(edf_path.relative_to(dataset_path))

        part_data = participants.get(subject_id, {})
        age = part_data.get("age", "N/A")
        sex = part_data.get("sex", "N/A")

        record = {
            "subject_id": subject_id,
            "session_id": session_id,
            "condition": pre_or_post,
            "pre_or_post": pre_or_post,
            "treatment_label": treatment_label,
            "recording_modality": "EEG",
            "task": task,
            "run": "1",
            "edf_filename": filename,
            "edf_relpath": edf_relpath,
            "edf_sha256": edf_sha256,
            "edf_bytes": edf_bytes,
            "start_time_if_available": "",
            "duration_if_available": edf_metadata["duration"],
            "sampling_rate_if_available": edf_metadata["sampling_rate"],
            "channels_count_if_available": edf_metadata["channels_count"],
            "dataset_source": "openneuro",
            "dataset_id": "ds005385",
            "notes": f"Age={age}, Sex={sex}",
        }

        manifest_records.append(record)

    print(f"Created {len(manifest_records)} manifest records")

    # Save manifest
    print("\n[5/6] Writing cohort_40_manifest.csv...")
    manifest_fields = [
        "subject_id",
        "session_id",
        "condition",
        "pre_or_post",
        "treatment_label",
        "recording_modality",
        "task",
        "run",
        "edf_filename",
        "edf_relpath",
        "edf_sha256",
        "edf_bytes",
        "start_time_if_available",
        "duration_if_available",
        "sampling_rate_if_available",
        "channels_count_if_available",
        "dataset_source",
        "dataset_id",
        "notes",
    ]

    with Path("cohort_40_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_records)

    print("Saved cohort_40_manifest.csv")

    # Create subject summary
    print("\n[6/6] Writing subject_summary.csv...")
    subject_data = defaultdict(lambda: {"pre": [], "post": []})

    for rec in manifest_records:
        subject = rec["subject_id"]
        condition = rec["pre_or_post"]
        if condition in ("pre", "post"):
            subject_data[subject][condition].append(rec)

    summary_records: list[dict] = []
    for subject in sorted(subject_data.keys()):
        pre_files = subject_data[subject]["pre"]
        post_files = subject_data[subject]["post"]

        has_pre = "yes" if pre_files else "no"
        has_post = "yes" if post_files else "no"

        pre_selected = sorted(pre_files, key=lambda x: (x["task"], x["edf_filename"]))[0] if pre_files else None
        post_selected = sorted(post_files, key=lambda x: (x["task"], x["edf_filename"]))[0] if post_files else None

        flags: list[str] = []
        if not pre_selected:
            flags.append("MISSING_PRE")
        if not post_selected:
            flags.append("MISSING_POST")
        if len(pre_files) > 1:
            flags.append(f"MULTIPLE_PRE({len(pre_files)})")
        if len(post_files) > 1:
            flags.append(f"MULTIPLE_POST({len(post_files)})")

        summary_records.append(
            {
                "subject_id": subject,
                "has_pre": has_pre,
                "has_post": has_post,
                "pre_file": pre_selected["edf_filename"] if pre_selected else "",
                "post_file": post_selected["edf_filename"] if post_selected else "",
                "pre_sha256": pre_selected["edf_sha256"] if pre_selected else "",
                "post_sha256": post_selected["edf_sha256"] if post_selected else "",
                "pre_duration": pre_selected["duration_if_available"] if pre_selected else "",
                "post_duration": post_selected["duration_if_available"] if post_selected else "",
                "pre_sampling_rate": pre_selected["sampling_rate_if_available"] if pre_selected else "",
                "post_sampling_rate": post_selected["sampling_rate_if_available"] if post_selected else "",
                "channel_count_pre": pre_selected["channels_count_if_available"] if pre_selected else "",
                "channel_count_post": post_selected["channels_count_if_available"] if post_selected else "",
                "flags": "; ".join(flags) if flags else "OK",
                "exclusion_reason_if_any": "",
            }
        )

    summary_fields = [
        "subject_id",
        "has_pre",
        "has_post",
        "pre_file",
        "post_file",
        "pre_sha256",
        "post_sha256",
        "pre_duration",
        "post_duration",
        "pre_sampling_rate",
        "post_sampling_rate",
        "channel_count_pre",
        "channel_count_post",
        "flags",
        "exclusion_reason_if_any",
    ]

    with Path("subject_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_records)

    print("Saved subject_summary.csv")

    print("\n" + "=" * 80)
    print("Extraction complete")
    print("=" * 80)
    print(f"Cohort subjects: {len(cohort_subjects)}")
    print(f"Total EDF files: {len(manifest_records)}")
    print("Outputs: cohort_40_manifest.csv, subject_summary.csv")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3.11 extract_cohort.py /path/to/ds005385")
        sys.exit(1)

    main(sys.argv[1])

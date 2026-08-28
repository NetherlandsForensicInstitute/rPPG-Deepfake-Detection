import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.filter import filter_heart_rate_frequency

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


_SUFFIX_BY_BAND = {
    "bandpass": "_bandpass",
    "bandstop": "_bandstop",
    "lowpass":  "_lowonly",
    "highpass": "_highonly",
}


def _default_output_path(input_path: Path, band: str) -> Path:
    return input_path.with_stem(input_path.stem + _SUFFIX_BY_BAND[band])


def _apply_filter_array(bvps: list, fps: float, band: str) -> list:
    """Filter a single BVPS array along its last axis (frames) and return as a nested list."""
    arr = np.array(bvps, dtype=float)
    filtered = filter_heart_rate_frequency(arr, fps=fps, mode=band, axis=-1)
    return filtered.tolist()


def _apply_filter(bvps, fps: float, band: str):
    """Filter BVPS data, which is either a single array (one ROI) or a dict of
    arrays keyed by region name (multiple ROIs, e.g. "per_region" extractions)."""
    if isinstance(bvps, dict):
        return {region: _apply_filter_array(values, fps, band) for region, values in bvps.items()}
    return _apply_filter_array(bvps, fps, band)


_MODE_DESCRIPTIONS = {
    "bandpass": "bandpass (keep 0.5-4 Hz HR band)",
    "bandstop": "bandstop (remove 0.5-4 Hz HR band, keep everything else)",
    "lowpass":  "lowpass (keep only <0.5 Hz — drift/slow motion, no heartbeat)",
    "highpass": "highpass (keep only >4 Hz — tracking jitter/noise, no heartbeat)",
}


def main() -> None:
    """
    Apply a heart-rate frequency-band filter to extracted BVP signals stored in NDJSON.

    Usage:
        python filter_signals.py <input.json> [output.json] [--band {bandpass,bandstop,lowpass,highpass}] [--fps FPS]

    Arguments:
        input        Path to the NDJSON file produced by extraction/extract_signals.py.
        output       (optional) Output path. Defaults to <input>_<suffix>.json.
        --band       Which band to keep/remove (default: bandpass). See --help for details.
        --remove     Deprecated alias for --band bandstop.
        --fps        Override the sampling frequency (Hz) for all records.
                     By default the per-record "FPS" field is used.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Input NDJSON file with extracted signals.")
    parser.add_argument("output", type=Path, nargs="?", help="Output NDJSON path (default: <input>_<suffix>.json).")
    parser.add_argument("--band", choices=list(_SUFFIX_BY_BAND), default="bandpass",
                        help="Frequency band to keep/remove (default: bandpass). "
                             "bandpass=0.5-4Hz kept, bandstop=0.5-4Hz removed, "
                             "lowpass=<0.5Hz kept, highpass=>4Hz kept.")
    parser.add_argument("--remove", action="store_true",
                        help="Deprecated alias for --band bandstop.")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS for all records.")
    args = parser.parse_args()

    input_path: Path = args.input
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    band = "bandstop" if args.remove else args.band
    output_path: Path = args.output or _default_output_path(input_path, band)
    logging.info(f"Filter mode : {_MODE_DESCRIPTIONS[band]}")
    logging.info(f"Input       : {input_path}")
    logging.info(f"Output      : {output_path}")

    processed = skipped = 0

    with input_path.open("r") as fin, output_path.open("w") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            fps = args.fps if args.fps is not None else record.get("FPS")
            if fps is None:
                logging.warning(f"Line {line_num}: no FPS found; skipping filter for this record.")
                fout.write(json.dumps(record) + "\n")
                skipped += 1
                continue

            bvps = record.get("BVPS")
            if bvps is None:
                logging.warning(f"Line {line_num}: no BVPS field; writing record unchanged.")
                fout.write(json.dumps(record) + "\n")
                skipped += 1
                continue

            try:
                record["BVPS"] = _apply_filter(bvps, fps=float(fps), band=band)
            except Exception as exc:
                logging.error(f"Line {line_num}: filter failed ({exc}); writing record unchanged.")
                skipped += 1
            else:
                processed += 1

            fout.write(json.dumps(record) + "\n")

    logging.info(f"Done. Filtered {processed} records, skipped {skipped}.")
    logging.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

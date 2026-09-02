"""Command-line interface for the label-free extraction pipeline (Task 10).

Four stdlib-``argparse`` commands:

- ``validate INPUT`` — documents and manifest v2 validate without running
  feature algorithms.
- ``convert INPUT OUTPUT [--force]`` — document format conversion through
  the existing codecs.
- ``extract MANIFEST OUTPUT_DIR [--pack PACK ...] [--force]`` — batch
  extraction writing ``recordings.csv``, ``utterances.csv``, ``issues.csv``,
  and ``provenance.json``; exits ``1`` when any issue has severity ``error``.
- ``list-features [--pack PACK] [--level LEVEL]`` — deterministic catalog CSV.

Validation, conversion, listing, and feature warnings exit ``0``; argparse
syntax errors and invalid global inputs/config/output conditions exit ``2``
with one concise stderr message and no traceback.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

from .catalog import list_features
from .document import InvalidDocumentError, load_document, save_document
from .extraction import (
    _canonical_packs,
    _load_task_spec,
    _resolve_speaker,
    _validate_manifest_v2,
    extract_batch,
)
from .result import FeatureExtractionError, InvalidConfigError
from .schema import MissingInputError, validate_task_spec

_OUTPUT_FILES = ("recordings.csv", "utterances.csv", "issues.csv", "provenance.json")
_DEFAULT_PACKS = ("acoustic", "adult_neuro")


def _contained_output_path(path: str | Path, kind: str) -> Path:
    """Reject output paths that resolve outside the current working directory."""
    path = Path(path)
    if not path.expanduser().resolve().is_relative_to(Path.cwd().resolve()):
        raise InvalidConfigError(f"{kind} escapes the working directory: {path}")
    return path


def _validate_command(input_path: str) -> str:
    """Validate a document or manifest v2 input; return the short OK line."""
    path = Path(input_path)
    if not path.is_file():
        raise MissingInputError(f"input file not found: {path}")
    if path.suffix.lower() == ".cha":
        load_document(path)
        return f"OK {path} (document)"
    if path.suffix.lower() != ".json":
        load_document(path)
        return f"OK {path} (document)"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidDocumentError(f"document is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "rows" not in data:
        load_document(path)
        return f"OK {path} (document)"
    rows = _validate_manifest_v2(data, path.parent)
    for row in rows:
        transcript_path = Path(row["transcript_path"])
        if not transcript_path.is_file():
            raise MissingInputError(f"input file not found: {transcript_path}")
        document = load_document(transcript_path)
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file():
            raise MissingInputError(f"input file not found: {audio_path}")
        if row["task_spec_path"] is not None:
            validate_task_spec(_load_task_spec(row["task_spec_path"]))
        for target in row["target_speakers"] or []:
            _resolve_speaker(document, target)
    return f"OK {path} (manifest v2, {len(rows)} rows)"


def _convert_command(input_path: str, output_path: str, force: bool) -> None:
    """Load/detect the input document and save it by the output extension."""
    output = _contained_output_path(output_path, "output path")
    document = load_document(input_path)
    save_document(document, output, force=force)


def _extract_command(
    manifest: str, output_dir: str, packs, force: bool, task_spec_path=None
) -> int:
    """Run extract_batch and write the four deterministic output files.

    Pack selection, the manifest, and output-path containment are validated
    before any directory is created; pre-existing expected files are checked
    without ``mkdir`` so invalid global input never leaves an output path
    behind. The output directory must resolve inside the working directory.
    """
    canonical_packs = _canonical_packs(packs)  # invalid selection exits before touching disk
    out = _contained_output_path(output_dir, "output directory")
    existing = [out / name for name in _OUTPUT_FILES if (out / name).exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing output files: "
            + ", ".join(str(path) for path in existing)
            + " (pass --force)"
        )
    task_spec = _load_task_spec(task_spec_path) if task_spec_path is not None else None
    bundle = extract_batch(manifest, packs=canonical_packs, task_spec=task_spec)
    out.mkdir(parents=True, exist_ok=True)
    bundle.recordings.to_csv(
        out / "recordings.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    bundle.utterances.to_csv(
        out / "utterances.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    bundle.issues.to_csv(out / "issues.csv", index=False, encoding="utf-8", lineterminator="\n")
    provenance_path = _contained_output_path(out / "provenance.json", "output path")
    provenance_path.write_text(
        json.dumps(dict(bundle.provenance), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return 1 if "error" in set(bundle.issues["severity"]) else 0


def _list_features_command(pack, level, **metadata_filters) -> str:
    """Render the catalog CSV: header and all definition metadata."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        [
            "key",
            "pack",
            "level",
            "unit",
            "population",
            "reference",
            "prerequisites",
            "formula_version",
        ]
    )
    for name, values in metadata_filters.items():
        for value in values:
            list_features(**{name: value})
    definitions = list_features(pack=pack, level=level)
    for definition in definitions:
        if any(
            values
            and (
                getattr(definition, name) not in values
                if name in {"domain", "language_scope", "evidence_level"}
                else not set(getattr(definition, f"{name}s")) & set(values)
            )
            for name, values in metadata_filters.items()
        ):
            continue
        writer.writerow(
            [
                definition.key,
                definition.pack,
                definition.level,
                definition.unit,
                definition.population,
                definition.reference,
                "|".join(definition.prerequisites),
                definition.formula_version,
            ]
        )
    return buffer.getvalue()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="say-features")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a document or manifest v2 input")
    validate.add_argument("input")

    convert = sub.add_parser("convert", help="convert a document between formats")
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument("--force", action="store_true")

    extract = sub.add_parser("extract", help="extract a manifest v2 batch into CSV + JSON output")
    extract.add_argument("manifest")
    extract.add_argument("output_dir")
    extract.add_argument("--pack", action="append", dest="packs", default=[])
    extract.add_argument("--task-spec", default=None)
    extract.add_argument("--force", action="store_true")

    listing = sub.add_parser("list-features", help="list catalog features as CSV")
    listing.add_argument("--pack", default=None)
    listing.add_argument("--level", default=None)
    listing.add_argument("--domain", action="append", default=[])
    listing.add_argument("--language-scope", action="append", default=[])
    listing.add_argument("--task", action="append", default=[])
    listing.add_argument("--disorder", action="append", default=[])
    listing.add_argument("--evidence-level", action="append", default=[])

    return parser


def main(argv=None) -> int:
    """Entry point; returns the process exit status (0, 1, or 2)."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            print(_validate_command(args.input))
            return 0
        if args.command == "convert":
            _convert_command(args.input, args.output, args.force)
            return 0
        if args.command == "extract":
            packs = tuple(args.packs) if args.packs else _DEFAULT_PACKS
            return _extract_command(
                args.manifest, args.output_dir, packs, args.force, args.task_spec
            )
        if args.command == "list-features":
            sys.stdout.write(
                _list_features_command(
                    args.pack,
                    args.level,
                    domain=args.domain,
                    language_scope=args.language_scope,
                    task=args.task,
                    disorder=args.disorder,
                    evidence_level=args.evidence_level,
                )
            )
            return 0
        return 2
    except (FeatureExtractionError, OSError, ValueError) as exc:
        print(f"say-features: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]

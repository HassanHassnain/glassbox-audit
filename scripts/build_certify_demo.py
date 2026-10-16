#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "demo"
RESULTS_DIR = ROOT / "results" / "certification"
CARD_JSON = RESULTS_DIR / "toy-certify-demo.card.json"
CARD_HTML = DOCS_DIR / "certify-demo.card.html"
CARD_PDF = RESULTS_DIR / "toy-certify-demo.card.pdf"
STDOUT_PATH = DOCS_DIR / "certify-demo.stdout.txt"
TRANSCRIPT_PATH = DOCS_DIR / "certify-demo-transcript.md"
STORYBOARD_PATH = DOCS_DIR / "certify-demo.storyboard.json"
VTT_PATH = DOCS_DIR / "certify-demo.vtt"
VIDEO_PATH = DOCS_DIR / "certify-demo.mp4"


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    title: str
    terminal: str
    narration: str


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_certify() -> tuple[dict, str]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    display_command = [
        "python",
        "-m",
        "glassbox_audit.cli",
        "certify",
        "--artifacts",
        "artifacts/demo",
        "--output-json",
        relative(CARD_JSON),
        "--output-html",
        relative(CARD_HTML),
        "--output-pdf",
        relative(CARD_PDF),
        "--sign",
        "--git-ref",
        "c3-certify-demo-v1",
    ]
    command = [sys.executable, *display_command[1:]]
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    stdout = result.stdout.strip()
    STDOUT_PATH.write_text(
        "$ " + " ".join(display_command) + "\n\n" + stdout + "\n",
        encoding="utf-8",
    )
    return json.loads(CARD_JSON.read_text(encoding="utf-8")), stdout


def summarize_rungs(card: dict) -> list[str]:
    labels = {
        "separation": "separation",
        "held_out_causal": "held-out causal",
        "specificity": "specificity",
        "capability_cost": "capability cost",
        "matched_controls": "matched controls",
    }
    rows = []
    for key in card["overall_verdict"]["required_rungs"]:
        verdict = card["rungs"][key]["passed"]
        rows.append(f"{labels.get(key, key)}: {'PASS' if verdict else 'FAIL'}")
    return rows


def wrap_lines(text: str, width: int = 86) -> str:
    wrapped: list[str] = []
    for line in text.splitlines():
        if not line:
            wrapped.append("")
            continue
        wrapped.extend(textwrap.wrap(line, width=width, replace_whitespace=False) or [""])
    return "\n".join(wrapped)


def build_segments(card: dict, stdout: str) -> list[Segment]:
    digest = card["signature"]["digest"]
    rung_lines = "\n".join(f"  {row}" for row in summarize_rungs(card))
    stdout_lines = "\n".join("  " + line for line in stdout.splitlines()[:11])
    return [
        Segment(
            0,
            12,
            "Glassbox certify in 90 seconds",
            "\n".join(
                [
                    "$ python -m glassbox_audit.cli certify \\",
                    "    --artifacts artifacts/demo \\",
                    f"    --output-json {relative(CARD_JSON)} \\",
                    f"    --output-html {relative(CARD_HTML)} \\",
                    f"    --output-pdf {relative(CARD_PDF)} \\",
                    "    --sign",
                ]
            ),
            "The demo starts from frozen toy audit artifacts and asks Glassbox to render a signed Intervention Audit Card.",
        ),
        Segment(
            12,
            27,
            "Inputs are fixed before certification",
            "\n".join(
                [
                    f"model: {card['model']}",
                    f"behavior: {card['behavior']}",
                    f"evidence class: {card['evidence_class']}",
                    f"records: {card['dataset']['n_records']}",
                    f"splits: train={card['protocol']['train_records']}, validation={card['protocol']['validation_records']}, test={card['protocol']['test_records']}",
                    f"corpus hash: {card['corpus_hash'][:16]}...",
                ]
            ),
            "The certifier reads the model identity, corpus hash, protocol counts, and frozen evidence before it scores any rung.",
        ),
        Segment(
            27,
            42,
            "The CLI emits a compact machine-readable verdict",
            stdout_lines,
            "The command prints the artifact directory, certified method, output paths, verdict, and card signature.",
        ),
        Segment(
            42,
            60,
            "Each required rung gets an explicit pass or fail",
            "\n".join(
                [
                    f"certified method: {card['certified_evidence_method']}",
                    f"overall passed: {card['overall_verdict']['passed']}",
                    "",
                    rung_lines,
                ]
            ),
            "The demo card fails because the SAE feature ablation does not pass held-out causal effect or matched controls.",
        ),
        Segment(
            60,
            75,
            "The card is rendered for humans and automation",
            "\n".join(
                [
                    f"JSON: {relative(CARD_JSON)}",
                    f"HTML: {relative(CARD_HTML)}",
                    f"PDF:  {relative(CARD_PDF)}",
                    f"stdout: {relative(STDOUT_PATH)}",
                    f"digest: {digest[:24]}...",
                ]
            ),
            "Glassbox writes a JSON contract for tooling, an HTML card for review, a PDF snapshot, and a digest tied to the demo ref.",
        ),
        Segment(
            75,
            90,
            "Takeaway",
            "\n".join(
                [
                    "Certification is not a rubber stamp.",
                    "A weak intervention fails loudly.",
                    "The artifacts are deterministic and reviewable.",
                    "",
                    f"Open: {relative(CARD_HTML)}",
                ]
            ),
            "The useful part is the discipline: every acceptance or rejection is linked back to frozen files and hashes.",
        ),
    ]


def timestamp(seconds: int) -> str:
    return f"00:{seconds // 60:02d}:{seconds % 60:02d}.000"


def write_transcript(segments: list[Segment]) -> None:
    lines = [
        "# C3 Certify Demo Transcript",
        "",
        "A 90-second narrated demo of `glassbox certify` running on the committed toy audit artifacts.",
        "",
        "| Time | Narration | Screen |",
        "|---|---|---|",
    ]
    for segment in segments:
        time_range = f"{timestamp(segment.start)}-{timestamp(segment.end)}"
        screen = "<br>".join(segment.terminal.splitlines())
        lines.append(f"| {time_range} | {segment.narration} | `{screen}` |")
    TRANSCRIPT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vtt(segments: list[Segment]) -> None:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.extend(
            [
                f"{timestamp(segment.start)} --> {timestamp(segment.end)}",
                segment.narration,
                "",
            ]
        )
    VTT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_storyboard(card: dict, segments: list[Segment]) -> None:
    story = {
        "schema_version": "1.0",
        "duration_seconds": 90,
        "source_artifacts": "artifacts/demo",
        "card_json": relative(CARD_JSON),
        "card_html": relative(CARD_HTML),
        "card_pdf": relative(CARD_PDF),
        "video": relative(VIDEO_PATH),
        "caption_file": relative(VTT_PATH),
        "overall_passed": card["overall_verdict"]["passed"],
        "certified_evidence_method": card["certified_evidence_method"],
        "segments": [
            {
                "start": segment.start,
                "end": segment.end,
                "title": segment.title,
                "narration": segment.narration,
                "terminal": segment.terminal,
            }
            for segment in segments
        ],
    }
    STORYBOARD_PATH.write_text(json.dumps(story, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:")


def render_video(segments: list[Segment]) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    with tempfile.TemporaryDirectory(prefix="glassbox-certify-demo-") as temp:
        temp_dir = Path(temp)
        filters = [
            "drawbox=x=0:y=0:w=iw:h=ih:color=0x10151F:t=fill",
            "drawbox=x=48:y=112:w=1184:h=430:color=0x0B1020@0.94:t=fill",
            "drawbox=x=48:y=570:w=1184:h=96:color=0x172033@0.96:t=fill",
            "drawtext=text='Glassbox certify':fontcolor=0xF5F7FA:fontsize=40:x=58:y=38",
            "drawtext=text='90-second audit-card demo':fontcolor=0x9FB0C6:fontsize=24:x=395:y=49",
        ]
        for index, segment in enumerate(segments):
            terminal_path = temp_dir / f"terminal-{index}.txt"
            caption_path = temp_dir / f"caption-{index}.txt"
            terminal_path.write_text(
                segment.title + "\n\n" + wrap_lines(segment.terminal),
                encoding="utf-8",
            )
            caption_path.write_text(wrap_lines(segment.narration, width=94), encoding="utf-8")
            enable = f"between(t\\,{segment.start}\\,{segment.end})"
            filters.append(
                "drawtext="
                f"textfile={escape_filter_value(str(terminal_path))}:"
                "fontcolor=0xDCE7F3:fontsize=25:"
                "x=82:y=145:line_spacing=8:"
                f"enable='{enable}'"
            )
            filters.append(
                "drawtext="
                f"textfile={escape_filter_value(str(caption_path))}:"
                "fontcolor=0xF5F7FA:fontsize=24:"
                "x=70:y=602:line_spacing=8:"
                f"enable='{enable}'"
            )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x10151F:s=1280x720:r=2:d=90",
                "-vf",
                ",".join(filters),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                relative(VIDEO_PATH),
            ],
            cwd=ROOT,
            check=True,
        )
    return True


def main() -> None:
    card, stdout = run_certify()
    segments = build_segments(card, stdout)
    write_transcript(segments)
    write_vtt(segments)
    write_storyboard(card, segments)
    video_rendered = render_video(segments)
    print(
        json.dumps(
            {
                "card_json": relative(CARD_JSON),
                "card_html": relative(CARD_HTML),
                "card_pdf": relative(CARD_PDF),
                "transcript": relative(TRANSCRIPT_PATH),
                "storyboard": relative(STORYBOARD_PATH),
                "captions": relative(VTT_PATH),
                "video": relative(VIDEO_PATH) if video_rendered else None,
                "overall_passed": card["overall_verdict"]["passed"],
                "certified_evidence_method": card["certified_evidence_method"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

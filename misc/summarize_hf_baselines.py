from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
RUN_RE = re.compile(r"^(?P<family>yolo(?:v)?\d+)(?P<scale>[a-z]+)?(?:u)?_seed(?P<seed>\d+)$", re.IGNORECASE)
LATEX_SCALE_BY_VERSION = {
    "YOLO11": {"n", "l", "x"},
    "YOLOv10": {"n", "l", "x"},
    "YOLOv5": {"n", "l", "x"},
    "YOLOv8": {"n", "l", "x"},
    "YOLOv9": {"t", "c", "e"},
}
LATEX_VERSION_ORDER = ["YOLOv5", "YOLOv8", "YOLOv9", "YOLOv10", "YOLO11"]
LATEX_SCALE_ORDER = {
    "YOLO11": ["n", "l", "x"],
    "YOLOv10": ["n", "l", "x"],
    "YOLOv5": ["n", "l", "x"],
    "YOLOv8": ["n", "l", "x"],
    "YOLOv9": ["t", "c", "e"],
}
MODEL_COMPLEXITY = {
    ("YOLO11", "n"): (2.6, 6.6),
    ("YOLO11", "l"): (25.4, 87.6),
    ("YOLO11", "x"): (57.0, 196.0),
    ("YOLOv10", "n"): (2.8, 8.7),
    ("YOLOv10", "l"): (25.9, 127.9),
    ("YOLOv10", "x"): (31.8, 171.8),
    ("YOLOv5", "n"): (2.7, 7.8),
    ("YOLOv5", "l"): (53.2, 135.6),
    ("YOLOv5", "x"): (97.3, 247.3),
    ("YOLOv8", "n"): (3.2, 8.9),
    ("YOLOv8", "l"): (43.7, 165.7),
    ("YOLOv8", "x"): (68.2, 258.5),
    ("YOLOv9", "t"): (2.1, 8.5),
    ("YOLOv9", "c"): (25.6, 104.0),
    ("YOLOv9", "e"): (58.2, 193.0),
}


@dataclass(frozen=True)
class TestRow:
    repo_id: str
    repo_dir: str
    dataset_group: str
    run_name: str
    yolo_version: str
    scale: str
    seed: int | None
    map50: float | None
    map5095: float | None
    precision: float | None
    recall: float | None
    source: str


def parse_repo_ids(path: Path) -> list[str]:
    repos: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if REPO_RE.match(line) and line not in seen:
            repos.append(line)
            seen.add(line)
    return repos


def repo_dir_name(repo_id: str) -> str:
    return repo_id.split("/", 1)[1]


def dataset_group_from_text(text: str) -> str:
    lowered = text.lower()
    if "missing" in lowered:
        return "missing"
    if "full" in lowered:
        return "full"
    return "unknown"


def parse_run_name(run_name: str) -> tuple[str, str, int | None]:
    match = RUN_RE.match(run_name)
    if not match:
        return "Unknown", "", None

    family = match.group("family").lower()
    scale = match.group("scale") or ""
    seed = int(match.group("seed"))

    if scale.endswith("u") and family == "yolov5":
        scale = scale[:-1]

    if family == "yolo11":
        version = "YOLO11"
    elif family.startswith("yolov"):
        version = "YOLO" + family.removeprefix("yolo")
    else:
        version = family.upper()

    return version, scale, seed


def as_float(row: dict[str, str], *names: str) -> float | None:
    lowered = {k.strip().lower(): v for k, v in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is None or value == "":
            continue
        try:
            return float(value)
        except ValueError:
            return None
    return None


def read_test_summaries(root: Path, repo_ids: list[str]) -> tuple[list[TestRow], list[str]]:
    rows: list[TestRow] = []
    notes: list[str] = []

    for repo_id in repo_ids:
        repo_dir = root / repo_dir_name(repo_id)
        summaries = sorted(repo_dir.rglob("test_summary_*.csv"))
        if not summaries:
            notes.append(f"{repo_id}: missing test_summary_*.csv")
            continue

        for summary in summaries:
            group = dataset_group_from_text(f"{repo_id} {summary.name}")
            with summary.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                count_before = len(rows)
                for raw in reader:
                    run_name = (raw.get("run_name") or raw.get("name") or "").strip()
                    if not run_name:
                        continue
                    version, scale, seed = parse_run_name(run_name)
                    rows.append(
                        TestRow(
                            repo_id=repo_id,
                            repo_dir=repo_dir.name,
                            dataset_group=group,
                            run_name=run_name,
                            yolo_version=version,
                            scale=scale,
                            seed=seed,
                            map50=as_float(raw, "mAP50", "metrics/mAP50(B)"),
                            map5095=as_float(raw, "mAP50-95", "metrics/mAP50-95(B)"),
                            precision=as_float(raw, "precision", "metrics/precision(B)"),
                            recall=as_float(raw, "recall", "metrics/recall(B)"),
                            source=str(summary.relative_to(root)),
                        )
                    )
                if len(rows) == count_before:
                    notes.append(f"{repo_id}: no readable rows in {summary.relative_to(root)}")

    return rows, notes


def collect_train_logs(root: Path, repo_ids: list[str]) -> dict[str, list[str]]:
    logs: dict[str, list[str]] = {}
    for repo_id in repo_ids:
        repo_dir = root / repo_dir_name(repo_id)
        run_names: list[str] = []
        for path in sorted(repo_dir.rglob("results.csv")):
            if path.is_file():
                run_names.append(path.parent.name)
        logs[repo_id] = run_names
    return logs


def add_train_test_notes(notes: list[str], rows: list[TestRow], train_logs: dict[str, list[str]]) -> None:
    test_runs_by_repo: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        test_runs_by_repo[row.repo_id].add(row.run_name)

    for repo_id, train_runs in train_logs.items():
        train_set = set(train_runs)
        missing_test = sorted(train_set - test_runs_by_repo.get(repo_id, set()))
        missing_train = sorted(test_runs_by_repo.get(repo_id, set()) - train_set)
        if missing_test:
            notes.append(f"{repo_id}: train results without test summary rows: {', '.join(missing_test)}")
        if missing_train:
            notes.append(f"{repo_id}: test summary rows without train results.csv: {', '.join(missing_train)}")


def fmt(value: float | None) -> str:
    return "" if value is None or math.isnan(value) else f"{value:.4f}"


def fmt_mean_std(values: list[float]) -> str:
    avg = mean(values)
    if avg is None:
        return ""
    sd = stdev(values)
    if sd is None:
        return f"{avg:.4f}"
    return f"{avg:.4f} +- {sd:.4f}"


def fmt_latex_mean_std(values: list[float]) -> str:
    avg = mean(values)
    if avg is None:
        return "--"
    sd = stdev(values)
    if sd is None:
        return f"{avg * 100:.2f}"
    return f"{avg * 100:.2f} $\\pm$ {sd * 100:.2f}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    avg = mean(values)
    assert avg is not None
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def markdown_table(headers: list[str], body: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def grouped_model_rows(rows: list[TestRow]) -> dict[str, dict[tuple[str, str], list[TestRow]]]:
    by_version: dict[str, dict[tuple[str, str], list[TestRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_version[row.yolo_version][(row.dataset_group, row.scale)].append(row)
    return by_version


def build_latex(rows: list[TestRow]) -> str:
    lines = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{caption}",
        r"\begin{document}",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\caption{Baseline YOLO test results (mean $\pm$ std).}",
        r"\label{tab:yolo-baselines}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccc}",
        r"\toprule",
        r"Model & mAP50 & mAP50-95 & Params (M) & GFLOPs \\",
        r"\midrule",
        "",
    ]

    grouped_rows = grouped_model_rows(rows)
    for version in LATEX_VERSION_ORDER:
        version_grouped = grouped_rows.get(version)
        if not version_grouped:
            continue
        allowed_scales = LATEX_SCALE_BY_VERSION.get(version, set())
        scale_rank = {scale: idx for idx, scale in enumerate(LATEX_SCALE_ORDER.get(version, []))}
        filtered_keys = [
            key
            for key in sorted(version_grouped, key=lambda item: scale_rank.get(item[1], 999))
            if key[1] in allowed_scales
        ]
        if not filtered_keys:
            continue
        lines.append(rf"\multicolumn{{5}}{{c}}{{{latex_escape(version)}}} \\")
        lines.append(r"\midrule")

        for group, scale in filtered_keys:
            group_rows = version_grouped[(group, scale)]
            model_name = f"{version}-{scale}" if scale else version
            params, gflops = MODEL_COMPLEXITY.get((version, scale), (None, None))
            map50_values = [r.map50 for r in group_rows if r.map50 is not None]
            map5095_values = [r.map5095 for r in group_rows if r.map5095 is not None]
            lines.append(
                " & ".join(
                    [
                        latex_escape(model_name),
                        fmt_latex_mean_std(map50_values),
                        fmt_latex_mean_std(map5095_values),
                        "--" if params is None else f"{params:.1f}",
                        "--" if gflops is None else f"{gflops:.1f}",
                    ]
                )
                + r" \\"
            )

        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular*}", r"\end{table}", "", r"\end{document}"])
    return "\n".join(lines) + "\n"


def build_markdown(root: Path, rows: list[TestRow], notes: list[str], train_logs: dict[str, list[str]]) -> str:
    train_counts = {repo_id: len(run_names) for repo_id, run_names in train_logs.items()}
    lines: list[str] = [
        "# Baseline YOLO Results",
        "",
        f"- Download folder: `{root.name}/`",
        f"- Test summary rows: {len(rows)}",
        f"- Train `results.csv` files: {sum(train_counts.values())}",
        "",
        "## Downloaded Repositories",
        "",
    ]

    repo_body = []
    for repo_id, count in train_counts.items():
        test_count = sum(1 for row in rows if row.repo_id == repo_id)
        repo_body.append([repo_id, repo_dir_name(repo_id), str(test_count), str(count)])
    lines.extend(markdown_table(["Repo", "Folder", "Test rows", "Train logs"], repo_body))
    lines.append("")

    grouped: dict[tuple[str, str, str], list[TestRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.yolo_version, row.dataset_group, row.scale)].append(row)

    lines.extend(["## Aggregate By Model Scale", ""])
    aggregate_body: list[list[str]] = []
    for key in sorted(grouped):
        version, group, scale = key
        group_rows = grouped[key]
        map50_values = [r.map50 for r in group_rows if r.map50 is not None]
        map5095_values = [r.map5095 for r in group_rows if r.map5095 is not None]
        precision_values = [r.precision for r in group_rows if r.precision is not None]
        recall_values = [r.recall for r in group_rows if r.recall is not None]
        seeds = sorted({r.seed for r in group_rows if r.seed is not None})
        aggregate_body.append(
            [
                group,
                f"{version}-{scale}" if scale else version,
                ",".join(str(seed) for seed in seeds),
                str(len(group_rows)),
                fmt_mean_std(map50_values),
                fmt_mean_std(map5095_values),
                fmt_mean_std(precision_values),
                fmt_mean_std(recall_values),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Group",
                "Model",
                "Seeds",
                "Runs",
                "mAP50",
                "mAP50-95",
                "Precision",
                "Recall",
            ],
            aggregate_body,
        )
    )
    lines.append("")

    by_version: dict[str, list[TestRow]] = defaultdict(list)
    for row in rows:
        by_version[row.yolo_version].append(row)

    for version in sorted(by_version):
        lines.extend([f"## {version}", ""])
        version_grouped: dict[tuple[str, str], list[TestRow]] = defaultdict(list)
        for row in by_version[version]:
            version_grouped[(row.dataset_group, row.scale)].append(row)

        detail_body = []
        for group, scale in sorted(version_grouped):
            version_rows = version_grouped[(group, scale)]
            map50_values = [r.map50 for r in version_rows if r.map50 is not None]
            map5095_values = [r.map5095 for r in version_rows if r.map5095 is not None]
            precision_values = [r.precision for r in version_rows if r.precision is not None]
            recall_values = [r.recall for r in version_rows if r.recall is not None]
            seeds = sorted({r.seed for r in version_rows if r.seed is not None})
            repos = sorted({r.repo_dir for r in version_rows})
            detail_body.append(
                [
                    group,
                    f"{version}-{scale}" if scale else version,
                    ",".join(str(seed) for seed in seeds),
                    str(len(version_rows)),
                    fmt_mean_std(map50_values),
                    fmt_mean_std(map5095_values),
                    fmt_mean_std(precision_values),
                    fmt_mean_std(recall_values),
                    ", ".join(repos),
                ]
            )
        lines.extend(
            markdown_table(
                ["Group", "Model", "Seeds", "Runs", "mAP50", "mAP50-95", "Precision", "Recall", "Repo folder"],
                detail_body,
            )
        )
        lines.append("")

    if notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def download_repos(repo_ids: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for repo_id in repo_ids:
        local_dir = output_dir / repo_dir_name(repo_id)
        print(f"Downloading {repo_id} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF YOLO baselines and summarize metrics.")
    parser.add_argument("--hf-list", default="hf_download.txt", type=Path)
    parser.add_argument("--output-dir", default="baseline_reuslts", type=Path)
    parser.add_argument("--summary-name", default="BASELINE_RESULTS_SUMMARY.md")
    parser.add_argument("--tex-name", default="BASELINE_RESULTS_TABLES.tex")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    repo_ids = parse_repo_ids(args.hf_list)
    if not repo_ids:
        raise SystemExit(f"No Hugging Face repo ids found in {args.hf_list}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_repos(repo_ids, args.output_dir)

    rows, notes = read_test_summaries(args.output_dir, repo_ids)
    train_logs = collect_train_logs(args.output_dir, repo_ids)
    add_train_test_notes(notes, rows, train_logs)
    if not rows:
        notes.append("No test summary rows were found in any downloaded repository.")

    summary = build_markdown(args.output_dir, rows, notes, train_logs)
    summary_path = args.output_dir / args.summary_name
    summary_path.write_text(summary, encoding="utf-8")
    tex_path = args.output_dir / args.tex_name
    tex_path.write_text(build_latex(rows), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tex_path}")
    print(f"Rows: {len(rows)}")
    print(f"Train logs: {sum(len(run_names) for run_names in train_logs.values())}")


if __name__ == "__main__":
    main()

"""Build and execute the tutorial notebooks (no Jupyter server needed).

Usage, from the repository root:  python examples/build_notebooks.py
Each code cell runs in one shared namespace; printed text and the value of the last expression
are stored as outputs. ``Chart`` values are stored as SVG images (light theme) so they render on
GitHub, in Jupyter and in Colab.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
REPO = "https://github.com/ahmed-ibrahim-asl/hybrid-guard"
COLAB = "https://colab.research.google.com/github/ahmed-ibrahim-asl/hybrid-guard/blob/main/examples/{name}"

SETUP = '''try:
    import hybrid_guard
except ImportError:  # Colab or a fresh environment
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hybrid-guard"], check=True)
    import hybrid_guard
import numpy as np
import pandas as pd
from hybrid_guard import charts
print("hybrid-guard", hybrid_guard.__version__)'''


def _outputs(value, stdout: str) -> list:
    outs = []
    if stdout:
        outs.append(nbformat.v4.new_output("stream", name="stdout", text=stdout))
    if value is None:
        return outs
    from hybrid_guard.charts import Chart

    if isinstance(value, Chart):
        data = {"text/plain": f"<Chart {value.key}>"}
        if value.kind == "svg":
            data["image/svg+xml"] = value.svg(theme="light")
        else:
            data["text/html"] = value.html(theme="light")
        outs.append(nbformat.v4.new_output("display_data", data=data))
    elif hasattr(value, "_repr_html_") and hasattr(value, "to_string"):
        outs.append(nbformat.v4.new_output("execute_result", data={"text/html": value._repr_html_(), "text/plain": value.to_string()},
                                           execution_count=None))
    else:
        outs.append(nbformat.v4.new_output("execute_result", data={"text/plain": repr(value)}, execution_count=None))
    return outs


def build(name: str, cells: list[tuple[str, str]]) -> None:
    os.chdir(HERE)
    namespace: dict = {"__name__": "__main__"}
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    count = 0
    for kind, source in cells:
        if kind == "md":
            nb.cells.append(nbformat.v4.new_markdown_cell(source.format(colab=COLAB.format(name=name), repo=REPO)))
            continue
        count += 1
        tree = ast.parse(source)
        last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        body = ast.Module(body=tree.body[:-1] if last else tree.body, type_ignores=[])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(body, name, "exec"), namespace)
            value = eval(compile(ast.Expression(last.value), name, "eval"), namespace) if last else None
        cell = nbformat.v4.new_code_cell(source, execution_count=count)
        cell.outputs = _outputs(value, buffer.getvalue())
        for out in cell.outputs:
            if out.get("output_type") == "execute_result":
                out["execution_count"] = count
        nb.cells.append(cell)
    nbformat.write(nb, HERE / name)
    print(f"wrote {name} ({count} code cells)")


WIND = [
    ("md", "# hybrid-guard on real wind-farm data\n\n[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})\n\n"
           "This notebook checks physics-informed wind-power models on one year (2025) of three Siemens turbines at Hill of Towie, "
           "Scotland. It follows the two questions of `hybrid-guard`:\n\n"
           "1. **Before training:** can a bounded hybrid correct the physics prior at all?\n"
           "2. **Before deployment:** is a new model safe row by row, or does it only look good on average?\n\n"
           "Data: Hill of Towie open dataset (RES on behalf of TRIG), CC BY 4.0. Power is per unit of the 2.3 MW rating. "
           "The predictions are the frozen models of the accompanying paper; on this three-turbine sample the numbers differ "
           "slightly from the paper's full-farm results."),
    ("code", SETUP),
    ("code", 'from pathlib import Path\n'
             'local = Path("data/hill_of_towie_2025_sample.csv.gz")\n'
             'source = local if local.exists() else "https://raw.githubusercontent.com/ahmed-ibrahim-asl/hybrid-guard/main/examples/data/hill_of_towie_2025_sample.csv.gz"\n'
             'df = pd.read_csv(source, parse_dates=["timestamp"])\n'
             'y, t = df["power"].to_numpy(), df["timestamp"]\n'
             'df.head()'),
    ("md", "## 1. Before training: can a bounded hybrid fix the physics?\n\n"
           "A bounded hybrid predicts the physics prior plus a correction of at most δ. If the prior is off by more than δ on a row, "
           "no training can fix that row. `capacity_diagnostics` measures this from training data alone."),
    ("code", 'from hybrid_guard import capacity_curve, capacity_diagnostics, gap_distribution\n'
             'prior = df["physics_no_wake"].to_numpy()\n'
             'capacity_diagnostics(y, prior, delta=0.10)'),
    ("code", 'charts.capacity(capacity_curve(y, prior, np.round(np.arange(0, 0.82, 0.02), 2)))'),
    ("code", 'charts.gap(gap_distribution(y, prior))'),
    ("md", "## 2. Before deployment: the coverage-matched acceptance test\n\n"
           "`cmat` compares a candidate with the incumbent model. It accepts only if the average error stays within the margin, "
           "enough rows change materially, more of them improve than get worse, few are harmed (with confidence), and, for a model "
           "that acts on selected rows, it beats random selection at the same coverage."),
    ("code", 'from hybrid_guard import cmat, random_matched\n'
             'import json\n'
             'meta_path = Path("data/hill_of_towie_2025_sample.json")\n'
             'threshold = json.loads(meta_path.read_text())["harm_threshold"] if meta_path.exists() else 0.1316\n'
             'selected = df["harm_score"].to_numpy() <= threshold\n'
             'transfer, wake = df["data_model_transfer"].to_numpy(), df["physics_wake_mean"].to_numpy()\n'
             'results = {\n'
             '    "Retrained local model": cmat(y, df["data_model_old"], df["data_model_local"], timestamps=t),\n'
             '    "Physics-features hybrid": cmat(y, df["data_model_local"], df["hybrid_physics_features"], timestamps=t),\n'
             '    "Harm classifier": cmat(y, transfer, wake, selected, timestamps=t),\n'
             '    "Random selection": random_matched(y, transfer, wake, selected.mean(), timestamps=t),\n'
             '}\n'
             'print(results["Harm classifier"].summary())'),
    ("code", 'charts.accuracy_safety(results)'),
    ("code", 'charts.conditions(results)'),
    ("code", 'charts.changes(results)'),
    ("code", 'charts.leaderboard_table(results)'),
    ("md", "On this three-turbine sample the harm classifier is rejected: its harm upper bound is above 0.10. On the full 21-turbine farm the paper reports an acceptance. A smaller sample gives wider bounds, and CMAT does not certify what the evidence cannot support."),
    ("md", "## 3. Choosing a threshold and finding where a model fails"),
    ("code", 'from hybrid_guard import coverage_harm_curve, choose_threshold\n'
             'curve = coverage_harm_curve(y, transfer, wake, df["harm_score"])\n'
             'charts.coverage_harm({"Harm classifier": curve}, chosen={"Harm classifier": selected.mean()})'),
    ("code", 'pick = choose_threshold(y, transfer, wake, df["harm_score"], timestamps=t, safety_margin=0.03)\n'
             'print(pick["reason"], "->", pick["threshold"])'),
    ("code", 'from hybrid_guard import bands, harm_by_condition\n'
             'wake_state = np.where(df["waked"] == 1, "waked", "free stream")\n'
             'table = harm_by_condition(y, df["data_model_local"], df["hybrid_physics_features"],\n'
             '                          rows=wake_state, columns=bands(df["wind_speed"], [3, 5, 8, 11, 14, 25], " m/s"))\n'
             'charts.harm_map(table, row_order=["free stream", "waked"],\n'
             '                column_order=["3-5 m/s", "5-8 m/s", "8-11 m/s", "11-14 m/s", "14-25 m/s"])'),
    ("md", "## 4. After deployment: watch it month by month"),
    ("code", 'from hybrid_guard import monitor\n'
             'watch = monitor(y, transfer, wake, selected, timestamps=t)\n'
             'charts.monitor_chart({"Harm classifier": watch})'),
    ("md", "## 5. One report with everything"),
    ("code", 'from hybrid_guard import save_report\n'
             'save_report("hill_of_towie_report.html", results, title="Hill of Towie 2025 (sample)",\n'
             '            charts=["accuracy_safety", "conditions", "changes", "leaderboard"],\n'
             '            extra=[charts.monitor_chart({"Harm classifier": watch}), charts.harm_map(table)])'),
    ("md", "## Cite\n\nIf you use the capacity diagnostics, cite the correction-capacity paper; if you use CMAT, cite the "
           "acceptance-test paper. Both are listed in the [README]({repo}#how-to-cite)."),
]

BEYOND = [
    ("md", "# hybrid-guard beyond wind: handing cases to an expert\n\n"
           "[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})\n\n"
           "Any system that switches between predictors has the same trap: a model hands some cases to an expert, a larger "
           "model or a human. An overall accuracy check averages over every case, so it cannot tell a skilful hand-off from a "
           "harmful one. This notebook uses `cmat_classification` on two examples: a simulated expert reviewer, where the answer "
           "is known, and real handwritten digits from scikit-learn."),
    ("code", SETUP),
    ("md", "## 1. A simulated expert reviewer\n\n"
           "A model is right 96% of the time on easy cases and 62% on hard ones, and it is less confident on the hard ones. "
           "A human reviewer is right 97% of the time on hard cases but only 85% on easy ones. Handing the reviewer the cases "
           "where the model is unsure should help; handing over a random set of cases of the same size should not."),
    ("code", 'from hybrid_guard import cmat_classification\n'
             'rng = np.random.default_rng(1)\n'
             'n = 20_000\n'
             'y = rng.integers(0, 2, n)\n'
             'hard = rng.random(n) < 0.25\n'
             'model = np.where(rng.random(n) < np.where(hard, 0.62, 0.96), y, 1 - y)\n'
             'reviewer = np.where(rng.random(n) < np.where(hard, 0.97, 0.85), y, 1 - y)\n'
             'confidence = np.where(hard, rng.uniform(0.5, 0.75, n), rng.uniform(0.7, 1.0, n))\n'
             'groups = np.arange(n) // 100           # cases arrive in batches of 100\n'
             'unsure = confidence < 0.7\n'
             'random_pick = rng.random(n) < unsure.mean()\n'
             'results = {\n'
             '    "Hand over unsure cases": cmat_classification(y, model, reviewer, unsure, groups=groups),\n'
             '    "Hand over random cases": cmat_classification(y, model, reviewer, random_pick, groups=groups),\n'
             '}\n'
             'for name, r in results.items():\n'
             '    print(name); print(r.summary()); print()'),
    ("code", 'charts.conditions(results)'),
    ("code", 'charts.changes(results)'),
    ("md", "Both hand-offs pass the aggregate check: the random one leaves the error rate where it was, well inside the "
           "one-point margin. Only CMAT shows the difference. The random hand-off changes the answer on hundreds of cases and "
           "about half of those changes are wrong, while handing over the unsure cases fixes almost every case it changes."),
    ("md", "## 2. Real data: handwritten digits\n\n"
           "A cheap model sees six principal components of each image; a random forest sees all pixels. The cheap model "
           "hands its unsure digits to the forest."),
    ("code", 'from sklearn.datasets import load_digits\n'
             'from sklearn.decomposition import PCA\n'
             'from sklearn.ensemble import RandomForestClassifier\n'
             'from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.model_selection import train_test_split\n'
             'X, yd = load_digits(return_X_y=True)\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, yd, test_size=0.6, random_state=0, stratify=yd)\n'
             'pca = PCA(n_components=6, random_state=0).fit(X_tr)\n'
             'cheap = LogisticRegression(max_iter=2000).fit(pca.transform(X_tr), y_tr)\n'
             'forest = RandomForestClassifier(n_estimators=300, random_state=0).fit(X_tr, y_tr)\n'
             'p = cheap.predict_proba(pca.transform(X_te))\n'
             'digits = cmat_classification(y_te, p.argmax(axis=1), forest.predict(X_te), p.max(axis=1) < 0.8,\n'
             '                             groups=np.arange(len(y_te)))\n'
             'print(digits.summary())'),
    ("md", "The hand-off cuts the error rate a lot, so an accuracy check would accept it at once. CMAT does not: with about a "
           "thousand test digits, the forest overrode the cheap model on fewer than a hundred of them, and the upper bound of the "
           "share made wrong is still above 10%. The evidence is too thin to certify the hand-off case by case. That is the "
           "intended behaviour: collect more labelled cases, or improve the confidence signal, before relying on it."),
]

if __name__ == "__main__":
    build("01_wind_quickstart.ipynb", WIND)
    build("02_beyond_wind_expert_handoff.ipynb", BEYOND)

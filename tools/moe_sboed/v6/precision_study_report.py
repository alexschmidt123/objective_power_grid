"""Development selection and frozen-bank reporting of matched terminal checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np


def read_arm(root, i, bank):
    path = Path((root / f"{i}.path").read_text().strip())
    with np.load(path / f"{bank}_evaluation.npz", allow_pickle=False) as data:
        scores = data["scores"].copy()
        receipt = json.loads(str(data["receipt"]))
    manifest = json.loads((path / "manifest.json").read_text())
    arch = ("observed_history_moe","relative_history_moe")[i // 2]
    method = ("moe-sboed","moe-sboed-uniform")[i % 2]
    if (manifest["architecture"], receipt["method"], receipt["seed"], manifest["objective"],
        manifest["local_precision_weight"], manifest["train_contrasts"], manifest["initialization"]) != (
        arch,method,101,"terminal_spce_local_precision",.5,8000,"from_scratch"):
        raise ValueError(f"Incorrect arm {i}")
    if scores.shape!=(2000,) or not np.isfinite(scores).all():
        raise ValueError("Incomplete score vector")
    if abs(scores.mean()-receipt["eig_nats"])>1e-10:
        raise ValueError("Inconsistent score receipt")
    return scores, dict(run=str(path), architecture=arch, **receipt)


def select(root):
    rows=[]
    for i in range(4):
        _, receipt = read_arm(root,i,"development")
        rows.append(receipt)
    if any(r["protocol"]!=rows[0]["protocol"] for r in rows):
        raise ValueError("Development protocols differ")
    # Prespecified tie break favors the existing action head. Ignore all Uniform scores for selection.
    index = 0 if rows[0]["eig_nats"] >= rows[2]["eig_nats"] else 2
    (root/"selection.json").write_text(json.dumps(dict(learned_index=index,development=rows),indent=2)+"\n")


def result(root):
    selection=json.loads((root/"selection.json").read_text())
    i=selection["learned_index"]
    x,a=read_arm(root,i,"canonical")
    y,b=read_arm(root,i+1,"canonical")
    if a["protocol"] != b["protocol"]:
        raise ValueError("Canonical protocols differ")
    diff=x-y
    mean=float(diff.mean())
    half=1.96*float(diff.std(ddof=1)/np.sqrt(len(diff)))
    out=dict(learned=a,uniform=b,target_nats=13,learned_over_13=bool(x.mean()>13),
             learned_minus_uniform=mean,paired_case_ci95=[mean-half,mean+half],
             mc_lower95_learned=float(x.mean()-1.96*x.std(ddof=1)/np.sqrt(len(x))),
             selection_rule="terminal learned development EIG, maximum across two declared architectures; earlier architecture on tie",
             training_seed_count=1)
    (root/"results.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps(out),flush=True)


if __name__=="__main__":
    {"select":select,"result":result}[sys.argv[1]](Path(sys.argv[2]))

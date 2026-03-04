#!/usr/bin/env python
"""
Analyse et export des runs MLflow.

Usage:
    python mlflow_analysis.py --list
    python mlflow_analysis.py --best  [exp_name]
    python mlflow_analysis.py --runs  [exp_name]
    python mlflow_analysis.py --csv   [exp_name] [output.csv]
"""

import argparse
import sys

try:
    import mlflow
    import pandas as pd
except ImportError:
    sys.exit("mlflow/pandas manquant. Installe avec: uv pip install mlflow pandas")


def get_exp(name):
    """Retourne l'objet experiment ou quitte si introuvable."""
    exp = mlflow.get_experiment_by_name(name)
    if not exp:
        sys.exit(f"Expérience '{name}' introuvable. Lance un run avec --mlflow d'abord.")
    return exp


def cmd_list(_args):
    """Liste toutes les expériences enregistrées."""
    for exp in mlflow.search_experiments():
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"{exp.name}  ({len(runs)} runs)")


def cmd_runs(args):
    """Liste les runs d'une expérience avec leur AUC et modèle."""
    exp = get_exp(args.runs)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    print(f"{'Run ID':36} | {'Status':10} | {'AUC':>6} | Model")
    print("-" * 70)
    for _, r in runs.iterrows():
        auc = r.get("metrics.test/auc", float("nan"))
        model = r.get("params.model", "?")
        print(f"{r.run_id:36} | {r.status:10} | {auc:6.4f} | {model}")


def cmd_best(args):
    """Affiche le run avec la meilleure AUC (test) d'une expérience."""
    exp = get_exp(args.best)
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.`test/auc` DESC"],
        max_results=1,
    )
    if runs.empty:
        sys.exit("Aucun run terminé trouvé.")
    r = runs.iloc[0]
    print(f"Meilleur run : {r.run_id}")
    print(f"  Modèle     : {r.get('params.model', '?')}")
    print(f"  LR         : {r.get('params.lr', '?')}")
    print(f"  Epochs     : {r.get('params.epochs', '?')}")
    print(f"  AUC        : {r.get('metrics.test/auc', float('nan')):.4f}")
    print(f"  Accuracy   : {r.get('metrics.test/acc', float('nan')):.4f}")
    print(f"  Sensitivity: {r.get('metrics.test/sensitivity', float('nan')):.4f}")
    print(f"  Specificity: {r.get('metrics.test/specificity', float('nan')):.4f}")


def cmd_csv(args):
    """Exporte tous les runs d'une expérience dans un CSV."""
    exp = get_exp(args.csv[0])
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    runs.to_csv(args.csv[1], index=False)
    print(f"{len(runs)} runs exportés dans {args.csv[1]}")


def main():
    p = argparse.ArgumentParser(description="Analyse des runs MLflow")
    p.add_argument("--list",  action="store_true",  help="Lister les expériences")
    p.add_argument("--runs",  metavar="EXP",         help="Lister les runs d'une exp")
    p.add_argument("--best",  metavar="EXP",         help="Meilleur run (test/auc)")
    p.add_argument("--csv",   nargs=2, metavar=("EXP", "FILE"), help="Export CSV")
    args = p.parse_args()

    if   args.list: cmd_list(args)
    elif args.runs: cmd_runs(args)
    elif args.best: cmd_best(args)
    elif args.csv:  cmd_csv(args)
    else:           p.print_help()


if __name__ == "__main__":
    main()

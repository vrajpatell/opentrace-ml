"""Summarize an extracted public RDD2022 annotation directory."""

import argparse

from opentrace_ml.datasets import load_rdd2022_annotations


parser = argparse.ArgumentParser()
parser.add_argument("rdd_root", help="Path to an extracted RDD2022 folder")
args = parser.parse_args()

annotations = load_rdd2022_annotations(args.rdd_root)
print(annotations["label"].value_counts())


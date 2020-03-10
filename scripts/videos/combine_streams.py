import os
import glob
import json
import itertools
import argparse
from argparse import RawTextHelpFormatter

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Script to combine the class predictions for rgb and flow streams.",
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("json_path", help="Path to the json files.", type=str)
    parser.add_argument("-save_path", help="Save path for the combined RGB-flow accuracy", default=None)
    args = parser.parse_args()
    return args


def compute_pred(proba, targets):
    video_preds = {k: np.argmax(v) for k, v in proba.items()}

    n_videos = len(video_preds)
    accuracy = sum([video_preds[k] == targets[k] for k in proba.keys()]) / n_videos * 100
    return accuracy


def main(args):
    json_files = glob.glob(os.path.join(args.json_path + "/*.json"))

    rgb_json = [file for file in json_files if "rgb" in file]
    flow_json = [file for file in json_files if "flow" in file]
    ground_truth_json = os.path.join(args.json_path, "ground_truth.json")

    acc_rgb, acc_flow, acc_two_stream = [], [], []

    with open(ground_truth_json, "r") as file:
        ground_truth = json.load(file)

    for flow_json, rgb_json in itertools.product(flow_json, rgb_json):
        with open(rgb_json, "r") as file:
            rgb_proba = json.load(file)

        with open(flow_json, "r") as file:
            flow_proba = json.load(file)

        acc_rgb.append(compute_pred(rgb_proba, ground_truth))
        acc_flow.append(compute_pred(flow_proba, ground_truth))

        flow_rgb_proba = {k: (np.array(rgb_proba[k]) + np.array(flow_proba[k])) / 2 for k in flow_proba}
        acc_two_stream.append(compute_pred(flow_rgb_proba, ground_truth))

        print("Accuracy: RGB = {0:2.2f}\tFlow = {1:2.2f}\ttwo stream = {2:2.2f}".format(acc_rgb[-1], acc_flow[-1],
                                                                                        acc_two_stream[-1]))

    print("\nBest:  RGB = {0:2.2f}\tFlow = {1:2.2f}\ttwo stream = {2:2.2f}\n".format(np.max(acc_rgb), np.max(acc_flow),
                                                                                     np.max(acc_two_stream)))

    if args.save_path is not None:
        df = pd.DataFrame(np.array([acc_rgb, acc_flow, acc_two_stream]).T, columns=["rgb", "flow", "two_stream"])
        df.to_csv(os.path.join(args.save_path, "two_stream_acc.csv"), sep="\t", index=False)
        print("Two stream results saved in ", args.save_path)
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)

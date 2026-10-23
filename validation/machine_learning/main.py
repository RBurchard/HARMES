from HARMES.evaluation.evaluation import evaluate_models, evaluate_transfer, evaluate_modality_transfer, evaluate_envs, evaluate_humid_only
from HARMES.preprocessing.load_and_process_raw_data import run_export

import argparse


def main(pretrain=False, modality_transfer=False, env=False, humidity=False, do_export=False):
    if pretrain:
        evaluate_transfer()
    elif modality_transfer:
        evaluate_modality_transfer()
    elif env:
        evaluate_envs()
    elif humidity:
        evaluate_humid_only()
    elif not do_export:
        evaluate_models()
    if do_export:
        run_export()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrain', action='store_true')
    parser.add_argument('-m', '--modality_transfer', action='store_true')
    parser.add_argument('-e', '--environment', action='store_true')
    parser.add_argument('-u', '--humidity', action='store_true')
    parser.add_argument('-s', '--export', action='store_true')
    args = parser.parse_args()
    main(args.pretrain, args.modality_transfer, args.environment, args.humidity, args.export)

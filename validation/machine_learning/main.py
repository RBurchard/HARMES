from HARMES.evaluation.evaluation import evaluate_models
from HARMES.preprocessing.load_and_process_raw_data import run_export
import sys



def main(do_eval=True, do_export=False):
    if do_eval:
        evaluate_models()
    if do_export:
        run_export()


if __name__ == "__main__":
    #### stupidly parse command line arguments.
    # defaults
    do_eval = True
    do_export = False
    if len(sys.argv) > 1:
        if "--no-eval" in sys.argv:
            do_eval = False
        if "--export" in sys.argv:
            do_export = True
   
    main(do_eval=do_eval, do_export=do_export)

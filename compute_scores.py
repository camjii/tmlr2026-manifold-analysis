import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.utils import resample
from tqdm import tqdm

from shape_happens import Runner
from shape_happens.utils import ActivationDataset, SupervisedMDS


def process_layer(args):
    (control, layer, label_col, target_col, activations, labels, reduction_method, n_components,
     manifold, k, repetitions, preprocess_func, global_metadata) = args
    
    # Hack to detect if we are debugging to avoid multiprocessing issues
    is_debugging = 'debugpy' in sys.modules or 'pydevd' in sys.modules
    
    if repetitions == 0:
        print("Repetitions is set to 0, initiating bootstrap mode...")
        repetitions = 1
        bootstrap = True
    else:
        bootstrap = False

    if preprocess_func is not None and isinstance(preprocess_func, list) and len(preprocess_func) == 1:
        preprocess_func = preprocess_func[0]
    
    norm = Normalizer()

    if reduction_method == 'PCA':
        rmodel = PCA(n_components=n_components)
    elif reduction_method == 'tSNE':
        rmodel = TSNE(n_components=n_components)
    elif reduction_method == 'Isomap':
        rmodel = Isomap(n_components=n_components)
    elif reduction_method == 'PLS':
        rmodel = PLSRegression(n_components=n_components)
    elif reduction_method == 'LDA':
        rmodel = LinearDiscriminantAnalysis(n_components=n_components)
    elif reduction_method == 'SMDS':
        rmodel = SupervisedMDS(n_components=n_components, manifold=manifold)
    else:
        raise ValueError(f"Unknown reduction method: {reduction_method}")

    try:
        rep_results = []
        indices = np.arange(len(labels))
        for i in range(repetitions):
            if bootstrap:
                cv_splits = []
                for b in range(k):
                    train_idx = resample(indices, replace=True, n_samples=len(labels), random_state=i * k + b)
                    test_idx = np.array(list(set(indices) - set(train_idx)))
                    cv_splits.append((train_idx, test_idx))
            else:
                cv_splits = KFold(n_splits=k, random_state=i, shuffle=True)

            steps = []
            if reduction_method == 'PCA':
                steps.append(('norm', norm))
                steps.append(('pca', rmodel))
            else:
                steps.append(('model', rmodel))
            
            pipeline = Pipeline(steps)

            cv_results = cross_validate(
                pipeline, 
                activations[:, layer], 
                labels, 
                cv=cv_splits, 
                n_jobs=-1 if not is_debugging else 1, # Use all available cores
                scoring=None,
            )

            fold_scores = cv_results['test_score'].tolist()
            
            rep_results.append({
                'preprocess_func': preprocess_func,
                'n_samples': len(labels),
                'n_components': n_components,
                'bootstrap': bootstrap,
                'k': k,
                'repetition_id': i,
                'manifold': manifold,
                'layer': layer,
                'target_col': target_col,
                'reduction_method': reduction_method,
                'score': float(np.mean(fold_scores)),
                'fold_scores': fold_scores,
                'label_col': label_col,
                'control': control,
                **global_metadata
            })

        return rep_results  

    except Exception as e:
        raise RuntimeError(f"Error in {args} (layer {layer}, target_col {target_col}: {e})")


class ScoreRunner(Runner):
    def __init__(self, config_path=None, save_path="results/scores/combined_scores.csv",
                 overwrite=False):
        self.save_path = save_path
        self.overwrite = overwrite
        super().__init__(config_path=config_path)

    def score_activations(self, **kwargs):
        id = self.hash_args(kwargs)

        path = kwargs["path"]
        label_col = kwargs["label_col"]
        reduction_method = kwargs["reduction_method"]

        model_name = kwargs.get("model_name", None)
        k = kwargs.get("k", 5)
        repetitions = kwargs.get("repetitions", 1)
        target_columns = kwargs.get("target_columns", None)
        layers = kwargs.get("layers", None)
        n_components = kwargs.get("n_components", 2)
        manifold = kwargs.get("manifold", None)
        preprocess_func = kwargs.get("preprocess_func", None)
        label_shift = kwargs.get("label_shift", 0)
        max_samples = kwargs.get("max_samples", None)
        control = kwargs.get("control", False)
        
        print(f"Scoring activations for {kwargs}")
        
        if '.pt' in path:
            pt_path = path
            folder_path = path.replace('.pt', '')
        else:
            folder_path = path
            pt_path = path + '.pt'

        try:
            ad = ActivationDataset.load(pt_path, model_name=model_name)
        except FileNotFoundError:
            ad = ActivationDataset.load(folder_path, model_name=model_name)
            

        if layers is None:
            layers = range(1, ad.activations['correct_answer'].shape[1])

        if target_columns is None:
            target_columns = ['correct_answer', 'last_prompt_token'] + ad.global_metadata['extra_columns']
        if isinstance(target_columns, str):
            target_columns = [target_columns]

        if isinstance(preprocess_func, str):
            preprocess_func = [preprocess_func]

        if preprocess_func is None:
            preprocess_func_lambdas = None
        else:
            preprocess_func_lambdas = []
            for func in preprocess_func or []:
                if func == 'datetime_to_dayofyear':
                    preprocess_func_lambdas.append(lambda x: pd.to_datetime(x).day_of_year)
                elif func == 'datetime_to_month':
                    preprocess_func_lambdas.append(lambda x: pd.to_datetime(x).month)
                elif func == 'datetime_to_year':
                    preprocess_func_lambdas.append(lambda x: np.abs(pd.to_datetime(x).year + label_shift))
                elif func == 'datetime_to_hour':
                    preprocess_func_lambdas.append(lambda x: pd.to_datetime(x).hour)
                elif func == 'log':
                    preprocess_func_lambdas.append(lambda x: np.log(x + 1))

        all_scores = []
        for target_col in target_columns:
            activations, labels = ad.get_slice(
                target_name=target_col,
                columns=label_col,
                preprocess_funcs=preprocess_func_lambdas,
                filter_incorrect=True
            )
            labels = np.squeeze(labels)

            if max_samples is not None and activations.shape[0] > max_samples:
                activations = activations[:max_samples]
                labels = labels[:max_samples]

            if control: # If this is a control task, shuffle the labels
                labels = np.random.permutation(labels)

            # Prepare args list
            args_list = [
                (
                    control,
                    layer,
                    label_col,
                    target_col,
                    activations,
                    labels,
                    reduction_method,
                    n_components,
                    manifold,
                    k,
                    repetitions,
                    preprocess_func,
                    ad.global_metadata
                )
                for layer in layers
            ]
            for args in tqdm(args_list, total=len(args_list), desc=f"Target: {target_col}"):
                results = process_layer(args)
                all_scores.extend(results)

        return pd.DataFrame(all_scores).to_csv(f"results/scores/{id}.csv", header=True, index=False)

    def run_experiment(self, args):
        return self.score_activations(**args)

    def combine_results(self, results_args):
        # Combine results from multiple experiments (e.g., aggregate metrics)
        print("Combining results...")
        ids = [self.hash_args(args) for args in results_args]
        combined_df = pd.concat([pd.read_csv(f"results/scores/{id}.csv") for id in ids], ignore_index=True)
        combined_df.to_csv(self.save_path, index=False)
        print(f"Results combined and saved to {self.save_path}")

    def results_exist(self, args):
        # Check if the results for the given args already exist
        if self.overwrite:
            return False
        id = self.hash_args(args)
        return os.path.exists(f"results/scores/{id}.csv")
    
    def validate_args(self, args):
        if 'duration' in args['path'] and args['manifold'] != 'euclidean' and isinstance(args['label_col'], list):
            print(f"Skipping {args['path']} with manifold {args['manifold']} and label_col {args['label_col']}")
            return False
        return super().validate_args(args)

    def run_all(self, multiprocessing=True, shuffle=False):
        full_argsets = self.merge_args(self.global_args, self.grid_args, self.local_args)

        if shuffle:
            random.shuffle(full_argsets)

        for args in tqdm(full_argsets, desc="Completed experiments", total=len(full_argsets)):
            if self.results_exist(args):
                print(f"Skipping existing experiment: {args}")
                continue
            if multiprocessing:
                p = mp.Process(target=self.run_experiment, args=(args,))
                p.start()
                p.join()
            else:
                self.run_experiment(args)

        self.combine_results(full_argsets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file containing global, grid, and local configs.")
    parser.add_argument("--save_path", type=str, default="results/scores/combined_scores.csv",
                        help="Path to save the results.")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite existing results.")
    parser.add_argument("--shuffle", action='store_true',
                    help="Shuffle the order in which experiments are computed. Useful for parallel tasks.")
    args = parser.parse_args()

    runner = ScoreRunner(config_path=args.config, save_path=args.save_path, overwrite=args.overwrite)
    runner.run_all(multiprocessing=False, shuffle=args.shuffle)
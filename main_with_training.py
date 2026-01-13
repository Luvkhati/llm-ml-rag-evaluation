# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:24:12 2025

@author: arnab
"""

"""
os                         : Helps work with file paths, folders, and operating system operations
json                       : Used to read and write configuration or result data in JSON format
pickle                     : Saves and loads Python objects like models and results
ast                        : Safely converts text data into Python objects (like dictionaries)
yaml                       : Reads configuration settings from YAML files
numpy                      : Performs numerical and array-based computations
matplotlib.pyplot          : Creates plots and visual graphs for results analysis
Path                       : Handles file and directory paths in an object-oriented way

Extract_random_test_dataset        : Randomly selects test data points from the dataset
Sample_test_dataset                : Selects test data based on distance or outlier strategy
load_model_data                    : Loads trained ML model, training data, and test data
NN_epoch_data_editting             : Prepares neural network training data by handling epochs
split_test_data                    : Separates dataset into input (X) and output (Y) parts
Test_data_point_extractor          : Converts a single test row into readable text format
Prompt_creator_Clusterer           : Builds LLM prompts using clustering-based RAG context
Baseline_prediction_function       : Generates baseline explanations like LIME or DiCE
LLM_answers                        : Sends prompts to LLMs and collects their responses
Evaluation_all_answers             : Evaluates LLM predictions against ML model outputs
format_llm_answers_with_labels     : Formats LLM answers by attaching feature or output names
Temp_ground_truth_to_string        : Converts true output values into readable string format
Classification_counterfactual_test_dataset : Prepares counterfactual test data for classification tasks
"""

import os
import json
import pickle
import ast
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from Main_modules import (
    Extract_random_test_dataset,
    Sample_test_dataset,
    load_model_data,
    NN_epoch_data_editting,
    split_test_data,
    Test_data_point_extractor,
    Prompt_creator_Clusterer,
    Baseline_prediction_function,
    LLM_answers,
    Evaluation_all_answers,
    format_llm_answers_with_labels,
    Temp_ground_truth_to_string,
    Classification_counterfactual_test_dataset
)
from Plotting_functions import create_dot_plot, create_dot_plot_plotly
#from RAG_classes import Baseline_RAG, KNN_RAG, VecDB_RAG, HybridVecRAG, Autoencoder_RAG
from Clustering_classes import Clustering_classifier
from joblib import load

# Stores the name of a supported clustering algorithm (currently Gaussian) (Unused for now)
Cluster_classifier_CLASS_MAP = {"Gausian"}


"""Loads the configuration from a YAML file and updates folder paths based on the selected use case and experiment.
This helps organize model files and results automatically for different experiments.
"""

def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    
    base_ml_folder = Path(cfg["ML_files_folder"])
    
    use_case, ml_task_type = next(iter(cfg["Use_Case"].items()))

    # Append the subfolder corresponding to the chosen use case
    cfg["ML_files_folder"] = base_ml_folder / use_case

    # Handle saving_folder if present
    if "saving_folder" in cfg and "Experiment" in cfg:
        base_folder = Path(cfg["saving_folder"])
        Use_case, ML_task_type = list(cfg["Use_Case"].items())[0]
        experiment_folder = base_folder / cfg["Experiment"] / Use_case
        cfg["saving_folder"] = experiment_folder

    return cfg

"""
Checks whether all required configuration values are present and valid before running the experiment.
Also ensures train/test sizes are correct and creates the results folder if it does not exist.
"""

def validate_config(cfg: dict) -> None:
    assert cfg["LLM_models"], "LLM_models cannot be empty."
    assert cfg["ML_models"], "ML_models cannot be empty."
    assert cfg["Metrics"], "Metrics cannot be empty."
    assert cfg["RAG_Class_variables"], "RAG_Class_variables cannot be empty."
    assert cfg["Output_variable_dict"], "Output_variable_dict cannot be empty."

    max_outputs = max(int(k) for k in cfg["Output_variable_dict"].keys())
    assert (
        len(cfg["output_upgrade_CF"]) >= max_outputs
    ), f"output_upgrade_CF must have >= {max_outputs} values."

    train_n = int(cfg["Train_subset"])
    test_n = int(cfg["Test_subset"])
    assert train_n > 0 and test_n > 0, "Train/Test subset sizes must be > 0."

    cfg["saving_folder"].mkdir(parents=True, exist_ok=True)

"""
Runs the complete experiment pipeline across use cases, models, metrics, and output settings.
Loads trained ML models, selects test data, builds clusters, and prepares RAG-based prompts.
Sends prompts to LLMs and also generates baseline predictions for comparison.
Evaluates LLM outputs against ML model predictions using selected metrics.
Stores detailed results, prompts, and intermediate data for analysis.
Optionally saves all outputs, reports, and configuration details to disk.
"""

def run(cfg: dict) -> dict:
    results = {}
    prompt_dict = {}
    
    for Use_case, ML_task_type in cfg["Use_Case"].items():
        print(f"\n ------Use Case: {Use_case} ----")
        
        for num_outputs, folder_name in cfg["Output_variable_dict"].items():
            num_outputs = int(num_outputs)
            print(f"\n ------Interpreting {num_outputs} output regressors----")

            ml_folder = cfg["ML_files_folder"] / folder_name
            
            output_upgrade = cfg["output_upgrade_CF"][Use_case][:num_outputs]  # CQI, UL_bitrate, DL_bitrate

            for model_name in cfg["ML_models"]:
                print(f"\n ------Interpreting {model_name}----")
                
                #NOTE: X Label encoders must be a blank dictionary atleast in case there are no categorical variables in Input
                if ML_task_type=='classification':
                    label_encoder_X=load(str(ml_folder) + "\\label_encoders.pkl")
                    label_encoder_Y=load(str(ml_folder) + "\\MLP_Y_scaler.save")
                else:
                    label_encoder_X=None
                    label_encoder_Y=None
                

                # load training/test/model artifacts
                ML_training_data, meta_data_filename, ML_test_data, model_file = (
                    load_model_data(model_name, str(ml_folder) + os.sep)
                )
                
                #try dropping secondary features of RF training data
                if Use_case !='Healthcare':
                    if model_name in ('DT', 'RF'):
                        ML_training_data = ML_training_data.iloc[:, :-1]
                    
                    if model_name in( "MLP", 'CNN_LSTM', 'CNN', 'Transformer'):
                        ML_training_data = NN_epoch_data_editting(
                            ML_training_data, replace=True, num_epochs=20
                        )
                

                
                X_labels = ML_test_data.columns[:-num_outputs].tolist()
                Y_labels = ML_test_data.columns[-num_outputs:].tolist()
                Predicted_labels=[f"{s}_predicted" for s in Y_labels]
                Immutables= cfg["Immutables"][Use_case]
                
                
                
                
                # Select cluster-train/test slices: random or outlier-based
                selection_strategy = cfg.get("Test_selection_strategy", "random")
                standardize = bool(cfg.get("Test_selection_standardize", True))
                
                
                #Extract Train and test data for classifier
                if  selection_strategy == "random":
                    Cluster_classifier_test_data = Extract_random_test_dataset(
                        ML_test_data,  number_of_test_points=cfg["Test_subset"]
                    )
                    print("[Selection] Using random strategy.")
               
                else:
                    Cluster_classifier_test_data, dist_test_data = Sample_test_dataset(
                        test_df=ML_test_data,
                        training_df=ML_training_data,
                        X_labels=X_labels,
                        len1=cfg["Train_subset"],
                        len2=cfg["Test_subset"],
                        mode=selection_strategy,
                        X_cat_encoder=label_encoder_X,
                        Y_cat_encoder=label_encoder_Y,
                        standardize=standardize,
                        random_state=int(cfg.get("seed", 42)),
                        cap_high_q= 0.80,
                    )
                    print(f"[Selection] Using {selection_strategy} strategy (standardize={standardize}).")
                
                
                
                
                
                for metric in cfg["Metrics"]:
                    print(f"\n ------for {metric}----")
                    
                    if metric=='Counterfactuals' and ML_task_type=='classification':
                        Cluster_classifier_test_data, preds_filtered = Classification_counterfactual_test_dataset(test_dataset=ML_test_data,
                                                                          model_name=model_name,
                                                                          X_encoders=label_encoder_X,
                                                                          Y_encoders=label_encoder_Y,
                                                                          target_cols=Y_labels,
                                                                          y_excluded_values=output_upgrade,
                                                                          n_rows=cfg["Test_subset"],
                                                                          model_file_path=model_file,  # or pass `model=` instead
                                                                          random_state=42,
                                                                      )
                    else:
                        preds_filtered=None
                                                         
                        
    
                    
                    Cluster_test_data_X, Cluster_test_data_Y = split_test_data(
                        Cluster_classifier_test_data, X_labels, Y_labels
                    )                   
                    # ----------------------------------------------------------
                    # BLOCK 1: Cluster builder
                    # ----------------------------------------------------------
                    Cluster_class_obj =  Clustering_classifier(metric, X_labels, Y_labels, Predicted_labels, Immutables)
                    
                    Clustering_strategy=cfg.get("Clustering_algorithm", "Gaussian")
                    Expert_algorithm=cfg.get("Expert_algorithm", "OLS")
                    
                    # build RAG DB based on metric
                    if metric in ("Accuracy", "Feature_importance"):
                        Cluster_class_obj.Create_Clusters(ML_training_data, X_labels, X_cat_encoder=label_encoder_X, Y_cat_encoder=label_encoder_Y, metric=metric, Clusterer=Clustering_strategy, expert=Expert_algorithm, max_clusters=2)
                    elif metric == "Counterfactuals":
                        Cluster_class_obj.Create_Clusters(ML_training_data, Immutables+Predicted_labels, metric, X_cat_encoder=label_encoder_X,Y_cat_encoder=label_encoder_Y, Clusterer=Clustering_strategy, expert=Expert_algorithm, max_clusters=8)

                   #blank variables to store answers
                    LLM_answers_per_MLmodel_metric = {llm: [] for llm in cfg["LLM_models"]}
                    Baseline_LLM_answers_per_MLmodel_metric = {llm: [] for llm in cfg["LLM_models"]}
                    GT_feature_attributions = {llm: [] for llm in cfg["LLM_models"]}
                    LLM_answers_baseline = {llm: [] for llm in cfg["LLM_models"]}
                    Clusterer_QA = []
                    
                    
                    # ----------------------------------------------------------
                    # BLOCK 2: LLM Prompt ilding and LLM response
                    # ----------------------------------------------------------
                    for i in range(len(Cluster_classifier_test_data)):
                        print(f'{i}th test data point of  {len(Cluster_classifier_test_data)} points')
                        
                        x_row = Cluster_test_data_X.iloc[i]
                        y_row = Cluster_test_data_Y.iloc[i]
                        x_str, y_str = Test_data_point_extractor(x_row, y_row)

                        prompt_and_baseline_args = {
                            "Use_case":Use_case,
                            "ML_task_type":ML_task_type,
                            "Dice_grid":cfg["Dice_grid"][Use_case],
                            "X_cat_encoder": label_encoder_X,
                            "Y_cat_encoder": label_encoder_Y,
                            "Inputs":X_labels,
                            "Outputs":Y_labels,
                            "Targets":Predicted_labels,
                            "Immutables":Immutables,
                            "metric": metric,
                            "ML_model_name": model_name,
                            "ML_model_file_path": model_file,
                            "training_data": ML_training_data,
                            "output_variable_number": num_outputs,
                            "output_upgrade_CF": output_upgrade,
                            "Cluster_classifier_object": Cluster_class_obj,
                            "LLM_outputs":cfg["LLM_outputs"]
                        }
                        
                        role, content, q_and_rag = Prompt_creator_Clusterer(prompt_and_baseline_args, x_str, y_str)
                        
                        debug(cfg.get("Cluster_classifier_debug", False), role, content, Cluster_class_obj)
                       

                        LLM_Answers = LLM_answers(cfg["LLM_models"], role, content)
                        
                        # ----------------------------------------------------------
                        # BLOCK 3: Baseline prediction
                        # ----------------------------------------------------------
                        Baseline_answers = Baseline_prediction_function(cfg["LLM_models"], prompt_and_baseline_args, x_row, random_sample_length=1000) 
                        
                        LLM_predictions, LLM_FI, LLM_model_error, LLM_dist_domain= Split_dictionary_LLM(metric, LLM_Answers,cfg["LLM_outputs"])
                        Baseline_predictions, Baseline_FI= Split_dictionary_Baseline(metric, Baseline_answers)
                        
                        formatted_answer=logger_formating(metric, LLM_predictions, Y_labels, X_labels)
                        Clusterer_QA.append(q_and_rag + formatted_answer)

                        #Prompt logging
                        prompt_dict[f"{model_name}_{metric}_{Cluster_class_obj.__class__.__name__}_{num_outputs} variables"] = (role + content)
                        #Storing LLM answers in a dictionary
                        LLM_answers_per_MLmodel_metric= Store_LLM_answers(LLM_predictions, LLM_answers_per_MLmodel_metric)
                        BaseLine_LLM_answers_per_MLmodel_metric= Store_LLM_answers(Baseline_predictions,  Baseline_LLM_answers_per_MLmodel_metric)
                        
                        
                        

                    eval_args = {
                        "ML_task_type":ML_task_type,
                        'ML_training_dataframe':ML_training_data,
                        'classification_X_encoder':label_encoder_X,
                        'classification_Y_encoder':label_encoder_Y,
                        "metric": metric,
                        "LLM_predictions": LLM_answers_per_MLmodel_metric,
                        "test_data_X": Cluster_test_data_X,
                        "test_data_Y": Cluster_test_data_Y,
                        "output_variable_number": num_outputs,
                        "model": model_file,
                        "model_name": model_name,
                        "X_labels": X_labels,
                        "Y_labels": Y_labels,
                        "Targets":Predicted_labels,
                        "Immutables": Immutables
                    }
                    
                    
                    # ----------------------------------------------------------
                    # BLOCK 4: Evaluate Results
                    # ----------------------------------------------------------
                    for LLM_model in cfg["LLM_models"]:
                        
                        #LLM Evaluation
                        metrics_report, LLM_preds, ML_preds, LLM_raw, dropped_test_answers = Evaluation_all_answers(
                            LLM_model,
                            eval_args,
                            increment=output_upgrade,
                        )
                        print('Metric report of LLM-MOE\n\n')
                        print(metrics_report)
                        
                        #Baseline evaluation
                        eval_args["LLM_predictions"] = BaseLine_LLM_answers_per_MLmodel_metric
                        
                        metrics_report_bl, LLM_preds_bl, ML_preds_bl, LLM_raw_bl, dropped_test_answers_bl = Evaluation_all_answers(
                            LLM_model,
                            eval_args,
                            increment=output_upgrade,
                        )
                        
                        print('Metric report of Baseline\n\n')
                        print(metrics_report_bl) 
  
                            
                    key = (
                        Use_case,
                        ML_task_type,
                        metric,
                        model_name,
                        num_outputs,
                        LLM_model,
                        getattr(Cluster_class_obj, "training_enabled", False),
                        cfg["User_Train_enable"],
                    )
                    results[key] = {
                        "metrics_report": metrics_report,
                        "metrics_report_bl": metrics_report_bl,
                        "ground_truth": ML_preds,
                        "llm_results": LLM_preds,
                        "llm_results_baseline": LLM_preds_bl,
                        "LLM_raw_answers": LLM_raw,
                        "LLM_raw_answers_baseline": LLM_raw_bl,
                        "dropped_test_answers": dropped_test_answers,
                        "test_data": Cluster_classifier_test_data,
                        "dist_test_data": dist_test_data,
                        "RAG_outputs":Clusterer_QA,
                        "X_labels": X_labels,
                        "Y_labels": Y_labels,
                        "output_upgrade": output_upgrade,
                        "CF_test_ML_predictions":preds_filtered
                    }

                    print(
                        f"---- logged {model_name}_{metric}_{num_outputs} ----"
                    )


    # ---- Save artifacts ----
    if cfg['Save_Yes_or_No']=='Yes':
        saving_folder = Path(cfg["saving_folder"])
        if "Ablation_results" in saving_folder.parts:
            LLM_model_name = cfg["LLM_models"][0]
            pickle_path = saving_folder / f"{LLM_model_name}_results.pkl"

        else:
            pickle_path = cfg["saving_folder"] / "results.pkl"

        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)

        any_value = next(iter(results.values()))
        any_value["test_data"].to_csv(
            cfg["saving_folder"] / "RAG_test_data.csv", index=False
        )

        with open(cfg["saving_folder"] / "prompts.pkl", "wb") as f:
            pickle.dump(prompt_dict, f)
        with open(cfg["saving_folder"] / "config_used.json", "w") as f:
            json.dump(_stringify_classes_in_cfg(cfg), f, indent=2)
    
    #Run function returns result variable
    return results


# Formats LLM answers for logging by attaching either output labels (for accuracy) or input labels (for counterfactuals)
def logger_formating(metric, Answers, Y_labels, X_labels):
    #Only for logging
    if metric == "Accuracy":
        formatted = format_llm_answers_with_labels(Answers, Y_labels)
    else:  # Counterfactuals
        formatted = format_llm_answers_with_labels(Answers, X_labels)
    
    return formatted

import re

# Extracts the first number found in a text string, or returns an empty string if no number is present
def extract_first_number_or_blank(text):
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return m.group(0) if m else ""

# Extracts simple prediction values from Mistral LLM responses and ignores advanced details like feature importance or errors
def Split_dictionary_LLM_mistral(metric, Answers, LLM_outputs):
    predictions = {}
    feature_importances = {}
    Model_error = {}
    Distribution_domain = {}

    for model, values in Answers.items():
        dict_str = values[0] if isinstance(values, list) else values

        if metric == 'Accuracy':
            predictions[model] = extract_first_number_or_blank(dict_str)
        else:
            predictions[model] = ""

        feature_importances[model] = ""
        Model_error[model] = ""

    return predictions, feature_importances, Model_error, Distribution_domain

# Parses structured LLM responses to extract predictions, feature importance, and error details based on the task type
def Split_dictionary_LLM(metric, Answers, LLM_outputs):
    #needs some error handling
    
    # Create two new dicts
    predictions = {}
    feature_importances = {}
    Model_error = {}
    Distribution_domain = {}

    for model, values in Answers.items():
        # values could be a list or a string
        if isinstance(values, list):
            dict_str = values[0]
        else:
            dict_str = values
        
        parsed = ast.literal_eval(dict_str)

        if metric=='Accuracy':
            predictions[model] = parsed["prediction"]
            
        elif metric=='Counterfactuals':
            predictions[model] = parsed["counterfactual_input"]
        

        if LLM_outputs in ("Two_output", "More_output"):
            feature_importances[model] = parsed["feature_importances"]
        else:
            feature_importances[model] = []
            
        
        if LLM_outputs =="More_output":
            Model_error[model] = parsed["local_error"]
        else:
            Model_error[model] = []
            
    return predictions, feature_importances, Model_error,  Distribution_domain

# Parses baseline method outputs to extract predictions and feature importance for comparison with LLM results
def Split_dictionary_Baseline(metric, Answers):
    #needs some error handling
    
    # Create two new dicts
    predictions = {}
    feature_importances = {}

    for model, values in Answers.items():
        # values could be a list or a string
        if isinstance(values, list):
            dict_str = values[0]
        else:
            dict_str = values
        
        parsed = ast.literal_eval(dict_str)

        if metric=='Accuracy':
            predictions[model] = parsed["prediction"]
            
        elif metric=='Counterfactuals':
            predictions[model] = parsed["counterfactual_input"]
        
        feature_importances[model] = parsed["feature_importances"]
        
    return predictions, feature_importances

# Stores each LLM prediction by appending it to the corresponding model’s result list
def Store_LLM_answers(Answers, Answers_dict):
    for k, v in Answers.items():
        if k in Answers_dict:
            Answers_dict[k].append(v)
    
    return Answers_dict

# Prints the LLM prompt and the retrieved cluster data to help debug how RAG context is being used
def debug(debug_key, role, content, Cluster_class_obj):
    if debug_key:
        print(
            "\n================ QUERY SENT TO LLM ================\n"
        )
        print(role + content)
        print(
            "\n================ RAG RETURNED DATAFRAME ===========\n"
        )

        if (
            hasattr(Cluster_class_obj, "_last")
            and "retrieved_df" in Cluster_class_obj._last
        ):
            print(
                Cluster_class_obj._last["retrieved_df"].to_string(
                    index=False
                )
            )
        else:
            idxs = (getattr(Cluster_class_obj, "_last", {}) or {}).get(
                "indices", []
            )
            k = min(8, len(idxs))
            if k:
                df = Cluster_class_obj.full_DB.iloc[idxs[:k]]
                print(df.to_string(index=False))
            else:
                print("[RAG] No neighbors captured yet.")
                
# Converts non-JSON-friendly values in the config (like paths or classes) into readable strings before saving
def _stringify_classes_in_cfg(cfg: dict) -> dict:
    """Helper to write config to JSON (classes -> names, Paths -> str)."""
    out = dict(cfg)
    #out["Cluster_classifier"] = [cls.__name__ for cls in cfg["Cluster_classifier"]]
    out["ML_files_folder"] = str(cfg["ML_files_folder"])
    out["saving_folder"] = str(cfg["saving_folder"])
    return out


"""
Creates visual plots to compare LLM predictions, ML model predictions, and baseline method results.
Automatically organizes and saves graphs for each model, metric, and output variable.
Only regression results are plotted; classification plots are currently skipped.
"""

def plot_results(results: dict, saving_folder: Path) -> None:
    plots_base_dir = saving_folder / "plots"
    

    for key, value in results.items():
        use_case, ML_task_type, metric, model, num_outputs, llm, retrieval, *_ = key
        y_labels = value["Y_labels"]
        gt = np.array(value["ground_truth"])
        pr = np.array(value["llm_results"])
        pr_baseline = np.array(value["llm_results_baseline"])
        
        if metric=='Accuracy':
            baseline_algo="LIME"
            
        elif metric=='Counterfactuals':
            baseline_algo="DICE"

        if gt.ndim == 1:
            gt = gt.reshape(-1, 1)
        if pr.ndim == 1:
            pr = pr.reshape(-1, 1)

        for i in range(num_outputs):
            
            if ML_task_type=='regression':
                plot_saving_dir = plots_base_dir / f'{model}'/ f'{num_outputs}_output_regressor'
                
                if cfg['Save_Yes_or_No']=='Yes':
                    plotly_saving_dir= plot_saving_dir / 'plotly'
                    plotly_saving_dir.mkdir(parents=True, exist_ok=True)
                    
                    matplotlib_saving_dir= plot_saving_dir / 'matplotlib'
                    matplotlib_saving_dir.mkdir(parents=True, exist_ok=True)
                    png = (
                        matplotlib_saving_dir
                        / f"{metric}_{model}_{llm}_out_{i}_of_{num_outputs}_with_{retrieval}.png"
                    )
                    html = (
                        plotly_saving_dir
                        / f"{metric}_{model}_{llm}_out_{i}_of_{num_outputs}_with_{retrieval}.html"
                    )
                elif cfg['Save_Yes_or_No']=='No':
                    png=None
                    html=None
                
                
                y1, y2, y3 = pr[:, i], gt[:, i], pr_baseline[:, i]
                label_i = y_labels[i] if i < len(y_labels) else f"output_{i}"
                title = (
                    f"{metric} of {model} model with {num_outputs} output variables "
                    f"using {retrieval} and {llm} for {label_i}"
                )

                    
                create_dot_plot(
                    y1,
                    f"{llm} _Predictions",
                    y2,
                    f"{model}_model predictions",
                    y3,
                    f'Baseline_{baseline_algo}_predictions',
                    title=str(title),
                    save_path=str(png),
                )
                create_dot_plot_plotly(
                    y1,
                    f"{llm} _Predictions",
                    y2,
                    f"{model}_model predictions",
                    y3,
                    f'Baseline_{baseline_algo}_predictions',
                    title=str(title),
                    save_path=str(html),
                )
            
            elif ML_task_type=='classification':
                pass
            
            
"""
Entry point of the program that loads the configuration, validates it, runs the full experiment,
and finally generates plots for the results.
"""

if __name__ == "__main__":
    cfg = load_config("config.yml")
    validate_config(cfg)
    results = run(cfg)
    plot_results(results, cfg["saving_folder"])

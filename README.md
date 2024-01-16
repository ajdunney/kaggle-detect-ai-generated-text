# Get Comp Dataset
mlflow run . -e load_raw_data --env-manager=local --experiment-name kaggle-detect-ai-generated-text

# Get Additional Dataset
mlflow run . -e load_raw_data -P dataset_type=dataset -P dataset_name=thedrcat/daigt-v3-train-dataset --env-manager=local --experiment-name kaggle-detect-ai-generated-text

# Train Distilbert
mlflow run C:\Users\ludov\kaggle\detect-ai-gen-text -e train -b local -P model_name=distilbert

# Train Roberta
mlflow run C:\Users\ludov\kaggle\detect-ai-gen-text -e train -b local -P config_path=proj_config_roberta.yaml -P model_name=roberta-base
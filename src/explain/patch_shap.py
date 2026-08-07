# src/explain/patch_shap.py
import os

filepath = "/home/ray/anaconda3/lib/python3.12/site-packages/shap/explainers/_tree.py"
if os.path.exists(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    old_str = 'float(learner_model_param["base_score"])'
    new_str = '(float(learner_model_param["base_score"][1:-1]) if isinstance(learner_model_param["base_score"], str) and learner_model_param["base_score"].startswith("[") else float(learner_model_param["base_score"]))'

    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(filepath, "w") as f:
            f.write(content)
        print("PATCH SUCCESSFUL: SHAP TreeExplainer patched inside the container.")
    else:
        print("PATCH ALREADY APPLIED OR TARGET STRING NOT FOUND.")
else:
    print(f"Error: {filepath} not found inside the container.")

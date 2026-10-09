import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://mlflow:5000")
client = MlflowClient()

versions = client.search_model_versions("name='fraud_detector'")
data = []
for v in versions:
    r = client.get_run(v.run_id)
    val_strat = r.data.params.get("validation_strategy", "N/A")
    f2 = r.data.metrics.get("f2_class_1", "N/A")
    f1 = r.data.metrics.get("f1_class_1", "N/A")
    prec = r.data.metrics.get("prec_class_1", "N/A")
    rec = r.data.metrics.get("rec_class_1", "N/A")
    mtype = r.data.params.get("model_type", r.data.tags.get("model_type", "N/A"))
    data.append(
        {
            "version": int(v.version),
            "aliases": getattr(v, "aliases", []),
            "model_type": mtype,
            "val_strategy": val_strat,
            "F2": f2,
            "F1": f1,
            "Precision": prec,
            "Recall": rec,
            "created": v.creation_timestamp,
        }
    )

df = pd.DataFrame(data).sort_values("version")
print(df.to_string())

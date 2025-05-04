import hydra
from omegaconf import DictConfig
from pathlib import Path
from plastic_id.data.datasets import PlasticDataset
from plastic_id.models import get_model
from plastic_id.evaluation.metrics import evaluate
from plastic_id.utils.timer import timed


@hydra.main(
    config_path="../../configs/experiment", config_name="baseline", version_base=None
)
def main(cfg: DictConfig):
    ds = PlasticDataset(Path(cfg.data.csv_path))
    model = get_model(cfg.model.name, cfg.model.params)

    with timed("fit"):
        model.fit(ds.X_train, ds.y_train)
    with timed("predict"):
        y_pred = model.predict(ds.X_test)

    metrics = evaluate(ds.y_test, y_pred)
    print(metrics)


if __name__ == "__main__":
    main()

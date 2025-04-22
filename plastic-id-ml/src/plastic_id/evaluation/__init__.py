# ─── append at the very bottom of the file ─────────────────────────────
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def save_model(model, tag: str, out_dir: Path = Path("artifacts")) -> Path:
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{tag}.joblib"
    joblib.dump(model, path)
    return path


def save_reports(y_true, y_pred, tag: str, out_dir: Path = Path("artifacts")):
    out_dir.mkdir(exist_ok=True)

    # ─ text report ──────────────────────────────────────────
    rep = classification_report(y_true, y_pred, output_dict=True)
    (out_dir / f"{tag}_report.json").write_text(json.dumps(rep, indent=2))

    # ─ confusion matrix image ──────────────────────────────
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, xticks_rotation="vertical", colorbar=False
    )
    plt.tight_layout()
    img_path = out_dir / f"{tag}_cm.png"
    plt.savefig(img_path, dpi=200)
    plt.close()
    return img_path

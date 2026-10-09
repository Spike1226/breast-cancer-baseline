"""学习项目：标准化、逻辑回归、阈值分析与独立测试评估。"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # 保存图片，不弹出会阻塞执行的窗口
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def evaluate_threshold(y_true, p_malignant, threshold):
    """0=恶性（阳性），1=良性；统计量全部来自传入的数据。"""
    y_pred = np.where(p_malignant >= threshold, 0, 1)
    fn = int(((y_true == 0) & (y_pred == 1)).sum())
    fp = int(((y_true == 1) & (y_pred == 0)).sum())
    malignant_count = int((y_true == 0).sum())
    benign_count = int((y_true == 1).sum())
    return {
        "threshold": threshold,
        "fn": fn,
        "fp": fp,
        "accuracy": float((y_pred == y_true).mean()),
        "sensitivity": 1 - fn / malignant_count if malignant_count else None,
        "specificity": 1 - fp / benign_count if benign_count else None,
    }


def main():
    output_dir = Path(__file__).resolve().parent
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 先留出20%测试集，再从剩余80%中取25%作为验证集。
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 保留原稿的模型设置。
    model = LogisticRegression(max_iter=100)
    model.fit(X_train_scaled, y_train)
    malignant_column = int(np.flatnonzero(model.classes_ == 0)[0])
    p_val = model.predict_proba(X_val_scaled)[:, malignant_column]
    p_test = model.predict_proba(X_test_scaled)[:, malignant_column]

    validation_rows = [evaluate_threshold(y_val, p_val, t) for t in [0.5, 0.3, 0.1]]
    # 默认基线阈值在查看测试结果之前固定，不根据测试结果修改。
    final_threshold = 0.5
    test_row = evaluate_threshold(y_test, p_test, final_threshold)
    fpr_val, tpr_val, _ = roc_curve(y_val, p_val, pos_label=0)
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test, pos_label=0)
    summary = {
        "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "validation_thresholds": validation_rows,
        "validation_auc": float(auc(fpr_val, tpr_val)),
        "test": test_row,
        "test_auc": float(auc(fpr_test, tpr_test)),
    }
    print(json.dumps(summary, indent=2))
    (output_dir / "results" / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_val, tpr_val, label=f"Validation AUC = {summary['validation_auc']:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance reference")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Logistic Regression: Validation ROC", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "figures" / "roc_validation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

X = data.data
y = data.target

print("X的形状:", X.shape)
print("y的形状:", y.shape)
print("标签名称：", data.target_names)

print(X[0],y[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state= 42,
    stratify = y

)

print("训练集特征的形状：", X_train.shape)
print("训练集标签的形状：", y_train.shape)
print("测试集特征的形状：", X_test.shape )
print("测试集标签的形状：", y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
   test_size  = 0.25,
   random_state= 42,
   stratify = y_train 
) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("标准化前：", X_train[0, :3])
print("标准化后：", X_train_scaled[0, :3])
print(X_train_scaled.shape)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=100)

model.fit(X_train_scaled, y_train)
y_val_pred = model.predict(X_val_scaled)

print(y_val_pred.shape)
print("真实标签：", y_val[:10])
print("预测标签：", y_val_pred[:10])

correct = (y_val_pred == y_val)
print(correct[:10])

accuracy = correct.mean()
print("验证集正确率：", accuracy)

wrong = ~correct

print("错误的真实标签：", y_val[wrong])
print("错误的预测标签：", y_val_pred[wrong])

false_negative = (y_val == 0) & (y_val_pred == 1)
print("假阴性数量：", false_negative.sum())

false_positive = (y_val_pred ==0) & (y_val ==1)
print("假阳性数量：", false_positive.sum())

malignant_count = (y_val == 0).sum()
sensitivity = (malignant_count - false_negative.sum()) / (malignant_count)
print("敏感度", sensitivity)

benign_count = (y_val == 1).sum()
specificity = (benign_count- false_positive.sum()) / (benign_count)
print("特异度", specificity)

val_proba = model.predict_proba(X_val_scaled)

print("类别顺序：", model.classes_)
print("前五个样本的概率：", val_proba[:5])
print("预测错误样本的概率：", val_proba[wrong])

import numpy as np
p_malignant = val_proba[:, 0]

for threshold in [0.5, 0.3, 0.1] :

 y_val_pred_new = np.where(p_malignant >= threshold, 0, 1)

 fp_new = ((y_val == 1) & (y_val_pred_new == 0)).sum()
 fn_new = ((y_val ==0) & (y_val_pred_new == 1)).sum()
 sensitivity_new = (malignant_count - fn_new) / (malignant_count)
 specificity_new = (benign_count - fp_new) / (benign_count)
 print("当前阈值：", threshold)
 print("假阴性数量：", fn_new)
 print("假阳性数量：", fp_new)
 fpr = fp_new / benign_count
 tpr = sensitivity_new
 
 print("ROC点的横坐标：", fpr)
 print("ROC点的纵坐标：", tpr)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr_values, tpr_values, thresholds = roc_curve(
    y_val,
    p_malignant,
    pos_label = 0
 
)

roc_auc = auc(fpr_values, tpr_values)
print("验证集 ROC AUC：", roc_auc)
plt.plot(fpr_values, tpr_values, label = "LogisticRegression")
plt.plot([0,1],[0,1], "--", color = "gray")

plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.title("validation ROC curve")
plt.legend()
plt.show()

final_threshold = 0.5
X_test_scaled = scaler.transform(X_test)

test_proba = model.predict_proba(X_test_scaled)
p_test_malignant = test_proba[:, 0]                                

y_test_pred = np.where(
    p_test_malignant >= final_threshold, 0, 1
)

fn_test = ((y_test == 0) & (y_test_pred ==1)).sum()
fp_test = ((y_test ==1) & (y_test_pred == 0)).sum()

malignant_test_count = (y_test == 0).sum()
benign_test_count = (y_test == 1).sum()

accuracy_test = (y_test_pred == y_test).mean()
sensitivity_test = 1 - fn_test / malignant_test_count
specificity_test = 1 - fp_test / benign_test_count
print("测试集阈值：", final_threshold)
print("测试集假阴性：", fn_test)
print("测试集假阳性：", fp_test)
print("测试集准确率：", accuracy_test)
print("测试集敏感度：", sensitivity_test)
print("测试集特异度：", specificity_test)

fpr_test, tpr_test, thresholds_test = roc_curve(
    y_test,
    p_test_malignant,
    pos_label = 0
)

auc_test = auc(fpr_test, tpr_test)

print("测试集 ROC AUC：", auc_test)


  
                                                                                                                                                                                                                                                                                                                                                                                
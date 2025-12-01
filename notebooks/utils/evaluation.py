import numpy as np

def confusion_counts(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def precision(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    return TP / (TP + FP + 1e-15)

def recall(y_true, y_pred):
    TP, TN, FP, FN = confusion_counts(y_true, y_pred)
    return TP / (TP + FN + 1e-15)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-15)

def eval_model(y_test, y_test_pred):
    prec = precision(y_test, y_test_pred)
    rec = recall(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    return prec, rec, f1

def eval_regression(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R2: {r2:.3f}")
    
    return mse, rmse, mae, r2
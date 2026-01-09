
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from geopy.distance import geodesic
from tqdm import tqdm
import joblib
import os


train_csv = r""
test_csv  = r""
model_dir = r""
os.makedirs(model_dir, exist_ok=True)

features   = []
target_col = ''
k_neighbors = 
bandwidth   = 
k_model     = 


train_df = pd.read_csv(train_csv, encoding='gbk')
test_df  = pd.read_csv(test_csv,  encoding='gbk')
for df in [train_df, test_df]:
    if 'ID' not in df.columns:
        df['ID'] = df.index


model_index_map = {}

for i in tqdm(range(len(train_df)), desc="Train Local Models"):
    center_point = (train_df.loc[i, 'WD'], train_df.loc[i, 'JD'])
    dists = train_df.apply(lambda row: geodesic((row['WD'], row['JD']), center_point).meters, axis=1)
    sub_idx = dists.nsmallest(k_neighbors).index

    X_local = train_df.loc[sub_idx, features].values
    y_local = train_df.loc[sub_idx, target_col].values
    local_weights = np.exp(-(dists.loc[sub_idx].values ** 2) / (2 * bandwidth ** 2))
    local_weights = local_weights / local_weights.sum()


    base = DecisionTreeRegressor(max_depth=6, random_state=42)
    try:

        model = AdaBoostRegressor(
            estimator=base,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
    except TypeError:

        model = AdaBoostRegressor(
            base_estimator=base,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

    model.fit(X_local, y_local, sample_weight=local_weights)

    model_path = os.path.join(model_dir, f"local_model_{i}.pkl")
    joblib.dump(model, model_path)
    model_index_map[i] = model_path

rf_local_preds = []

for i in tqdm(range(len(test_df)), desc="Predict Using Local Models"):
    test_point = (test_df.loc[i, 'WD'], test_df.loc[i, 'JD'])


    dists = train_df.apply(lambda row: geodesic((row['WD'], row['JD']), test_point).meters, axis=1)
    nearest_idx = dists.nsmallest(k_model).index

    pred_list, weight_list = [], []

    for idx in nearest_idx:
        model_path = model_index_map[idx]
        model = joblib.load(model_path)

        feat = test_df.loc[i, features].values.reshape(1, -1)
        pred_i = model.predict(feat)[0]

        dist_to_center = geodesic((train_df.loc[idx, 'WD'], train_df.loc[idx, 'JD']), test_point).meters
        weight_i = np.exp(-(dist_to_center ** 2) / (2 * bandwidth ** 2))

        pred_list.append(pred_i)
        weight_list.append(weight_i)

    weight_list = np.array(weight_list)
    weight_list = weight_list / weight_list.sum()
    final_pred = np.dot(pred_list, weight_list)
    rf_local_preds.append(final_pred)


def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ubrmse = np.sqrt(np.mean((y_pred - y_true - np.mean(y_pred - y_true)) ** 2))
    print(f"\n{name} 模型评估：")
    print(f" R²     = {r2:.3f}")
    print(f" RMSE   = {rmse:.3f}")
    print(f" MAE    = {mae:.3f}")
    print(f" ubRMSE = {ubrmse:.3f}")
    return r2, rmse, mae, ubrmse


test_df['GWRF_CatBoost'] = rf_local_preds   # 按你的原表头命名不改列名
test_df['真实值'] = test_df[target_col].values
evaluate(test_df['真实值'], test_df['GWRF_CatBoost'], name="高斯核融合AdaBoost")


output_df = test_df[['ID', 'WD', 'JD', '真实值', 'GWRF_CatBoost']].copy()
output_df['预测误差'] = output_df['GWRF_CatBoost'] - output_df['真实值']
output_path = r""
output_df.to_csv(output_path, index=False, encoding='utf-8-sig')


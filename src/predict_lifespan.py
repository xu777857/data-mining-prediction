# predict_lifespan.py

import pandas as pd
import joblib

def predict_new_lifespan(new_data_dict):

    # 1. 加载保存的模型、Scaler 和特征列名
    try:
        model = joblib.load('mlp_lifespan_predictor.joblib')
        scaler = joblib.load('scaler.joblib')
        feature_names = joblib.load('feature_names.joblib') # 加载训练时的特征列名
        print("模型、Scaler 和特征列名加载成功！")
    except FileNotFoundError as e:
        print(f"错误：找不到文件。请先运行主训练脚本生成所需文件。 缺失文件: {e.filename}")
        return None

    # 2. 准备新数据
    # 将字典转换为DataFrame
    new_data_df = pd.DataFrame([new_data_dict])
    print("\n输入的原始数据:")
    print(new_data_df)

    # 3. 对新数据进行独热编码（与训练时完全相同的方式）
    new_data_encoded = pd.get_dummies(new_data_df, drop_first=True)
    print("\n独热编码后的数据:")
    print(new_data_encoded)

    # 4. 对齐特征列（关键步骤！）
    # 确保新数据的特征列与模型期望的完全一致
    # reindex会自动添加缺失的列并填充为0，同时删除多余的列
    new_data_aligned = new_data_encoded.reindex(columns=feature_names, fill_value=0)
    print("\n对齐特征列后的数据 (与训练时一致):")
    print(new_data_aligned)

    # 5. 数据预处理：使用加载的scaler进行标准化
    new_data_scaled = scaler.transform(new_data_aligned)
    print("\n新数据已准备并标准化完成。")

    # 6. 执行预测
    prediction = model.predict(new_data_scaled)
    
    # 7. 解读结果
    predicted_lifespan = prediction[0]
    print(f"\n--- 预测结果 ---")
    print(f"根据提供的生活方式数据，预测该个体的寿命为: {predicted_lifespan:.2f} 岁")
    
    return predicted_lifespan

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义一个新个体的特征（键名必须与原始DataFrame的列名匹配）
    # 【修正】你的字典键名有多余的空格，如 ' avg_rest_hours_per_day'，这会导致列名不匹配。
    new_individual = {
        'gender': 'Male',
        'occupation_type': 'Professional',
        'avg_work_hours_per_day': 9.5,
        'avg_rest_hours_per_day': 7,  # 修正：去掉了前面的空格
        'avg_sleep_hours_per_day': 7.0,
        'avg_exercise_hours_per_day': 1.5
    }

    # 调用函数进行预测
    predict_new_lifespan(new_individual)

    # 再预测一个生活方式不同的人
    print("\n" + "="*30 + "\n")
    another_individual = {
        'gender': 'Female', # 试试另一种性别
        'occupation_type': 'Teacher',
        'avg_work_hours_per_day': 10,
        'avg_rest_hours_per_day': 6, # 修正：去掉了前面的空格
        'avg_sleep_hours_per_day': 6.5,
        'avg_exercise_hours_per_day': 1.5
    }
    predict_new_lifespan(another_individual)

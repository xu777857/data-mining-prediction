# predict_lifespan_rf.py

import pandas as pd
import joblib

def predict_lifespan_with_rf(new_data_dict):

    # 1. 加载保存的模型和特征名称
    MODEL_FILENAME = 'random_forest_lifespan_predictor.joblib'
    FEATURES_FILENAME = 'feature_names.joblib'
    
    try:
        model = joblib.load(MODEL_FILENAME)
        feature_names = joblib.load(FEATURES_FILENAME)
        print("模型和特征名称加载成功！")
    except FileNotFoundError:
        print("错误：找不到模型或特征文件。请先运行主训练脚本生成这些文件。")
        return None

    # 2. 准备新数据
    # 将字典转换为DataFrame
    new_data_df = pd.DataFrame([new_data_dict])
    print("\n原始输入数据:")
    print(new_data_df)

    # 3. 数据预处理：对新数据进行独热编码
    # 这一步必须和训练时完全一样
    new_data_encoded = pd.get_dummies(new_data_df, drop_first=True)
    
    # 4. 对齐特征
    # 确保新数据的特征与模型训练时的特征完全一致
    # reindex会自动添加缺失的列并用0填充，删除多余的列
    new_data_for_prediction = new_data_encoded.reindex(columns=feature_names, fill_value=0)
    
    print("\n预处理并对齐特征后的数据 (用于模型输入):")
    print(new_data_for_prediction)

    # 5. 执行预测
    prediction = model.predict(new_data_for_prediction)
    
    # 6. 解读结果
    predicted_lifespan = prediction[0]
    print(f"\n--- 预测结果 ---")
    print(f"根据提供的生活方式数据，预测该个体的寿命为: {predicted_lifespan:.2f} 岁")
    
    return predicted_lifespan

# --- 使用示例 ---
if __name__ == "__main__":
    # 定义一个新个体的特征 (键名必须与原始CSV文件的列名匹配)
    # 假设原始数据列名为 'gender', 'occupation_type', 'avg_work_hours_per_day' 等
    individual_A = {
        'gender': 'Male',
        'occupation_type': 'Professional',
        'avg_work_hours_per_day': 9.5,
        'avg_sleep_hours_per_day': 7.0,
        'avg_exercise_hours_per_day': 1.5
    }

    print("正在为个体A进行预测...")
    predict_lifespan_with_rf(individual_A)

    # 再预测一个生活方式不同的人
    print("\n" + "="*40 + "\n")
    individual_B = {
        'gender': 'Female',
        'occupation_type': 'Manual',
        'avg_work_hours_per_day': 12.0,
        'avg_sleep_hours_per_day': 6.0,
        'avg_exercise_hours_per_day': 0.5
    }
    print("正在为个体B进行预测...")
    predict_lifespan_with_rf(individual_B)

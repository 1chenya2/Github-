import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# 加载模型
model_path = "stacking_regressor_model.pkl"
stacking_regressor = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Stacking 模型预测与 SHAP 可视化", page_icon="📊")

st.title("📊 Stacking 模型预测与 SHAP 可视化分析")
st.write("""
通过输入特征值进行模型预测，并结合 SHAP 分析结果，了解特征对模型预测的贡献。
""")

# 左侧侧边栏输入区域
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 定义特征输入范围
wavelet_HHL_glrlm_RunLengthNonUniformityNormalized= st.sidebar.number_input("特征 wavelet.HHL.glrlm.RunLengthNonUniformityNormalized (范围: 0.3380315-0.4071359)", min_value=0.3380315, max_value=0.4071359, value=0.392841974686494)
depression = st.sidebar.number_input("特征 depression (范围: 0-1)", min_value=0, max_value=1, value=1)
ALB = st.sidebar.number_input("特征 ALB (范围: 30.400000-50.30000)", min_value=30.400000, max_value=50.30000, value=36.5)
wavelet_HLH_ngtdm_Contrast = st.sidebar.number_input("特征 wavelet.HLH.ngtdm.Contrast (范围:  0.0007196355-0.007903385)", min_value=0.0007196355, max_value=0.007903385, value=0.00200425913154255)
original_firstorder_Median = st.sidebar.number_input("特征 original.firstorder.Median (范围: 13-59)", min_value=13, max_value=59, value=29)
original_ngtdm_Strength = st.sidebar.number_input("特征 original.ngtdm.Strength (范围: 0.006529775-0.6016773)", min_value=0.006529775, max_value=0.6016773, value=0.0238386651523205)
wavelet_LLH_glcm_Imc2 = st.sidebar.number_input("特征 wavelet.LLH.glcm.Imc2 (范围: 0.5296347-0.9590042)", min_value=0.5296347, max_value=0.9590042, value=0.842884819810307)
wavelet_LLH_firstorder_Maximum = st.sidebar.number_input("特征 wavelet.LLH.firstorder.Maximum (范围: 180.8378 到 1161.716)", min_value=180.8378, max_value=1161.716, value=245.586205830349)
wavelet_LLL_firstorder_Energy = st.sidebar.number_input("特征 wavelet.LLL.firstorder.Energy (范围: 93352630 到 1078847000)", min_value=93352630, max_value=1078847000, value=217064766)
wavelet_HHL_firstorder_Median = st.sidebar.number_input("特征 wavelet.HHL.firstorder.Median (范围: -0.01568906 到 0.1198817)", min_value=-0.01568906, max_value=0.1198817, value=0.052452842888068)
original_glcm_ClusterShade = st.sidebar.number_input("特征 original.glcm.ClusterShade (范围: -42.909 到 6.599659)", min_value=-42.909, max_value= 6.599659, value=-0.788808389265009)
wavelet_HLH_glszm_GrayLevelNonUniformityNormalized = st.sidebar.number_input("特征 wavelet.HLH.glszm.GrayLevelNonUniformityNormalized (范围: 0.1647905 到 0.4181931)", min_value=0.1647905, max_value=0.4181931, value=0.288887906871038)
wavelet_HLH_glrlm_LongRunHighGrayLevelEmphasis = st.sidebar.number_input("特征 wavelet.HLH.glrlm.LongRunHighGrayLevelEmphasis (范围: 148.5932 到 1724.463)", min_value=148.5932, max_value=1724.463, value=422.496396998562)
original_firstorder_90Percentile = st.sidebar.number_input("特征 original.firstorder.90Percentile (范围: 38 到 77)", min_value=38, max_value=77, value=49)
original_firstorder_Energy = st.sidebar.number_input("特征 original.firstorder.Energy (范围: 11933140 到 128495400)", min_value=11933140 , max_value=128495400, value=24847591)
wavelet_HLH_glszm_LargeAreaHighGrayLevelEmphasis= st.sidebar.number_input("特征 wavelet.HLH.glszm.LargeAreaHighGrayLevelEmphasis (范围: 178720 到 10806650)", min_value=178720, max_value=10806650, value=932798)
wavelet_LLH_glcm_MCC = st.sidebar.number_input("特征 wavelet.LLH.glcm.MCC (范围:0.7759031 到 0.984111)", min_value=0.7759031, max_value=0.984111, value=0.906120059179769)
Operative_time = st.sidebar.number_input("特征 Operative.time (范围: 49 到 550)", min_value=49, max_value= 550, value=550)
wavelet_LLH_gldm_GrayLevelNonUniformity = st.sidebar.number_input("特征 wavelet.LLH.gldm.GrayLevelNonUniformity (范围: 3164.433 到 20873.98)", min_value=3164.433, max_value=20873.98, value=5863.36161806208)

# 添加预测按钮
predict_button = st.sidebar.button("进行预测")

# 主页面用于结果展示
if predict_button:
    st.header("预测结果")
    try:
        # 将输入特征转换为模型所需格式
        input_array = np.array([wavelet_HHL_glrlm_RunLengthNonUniformityNormalized, depression, ALB, wavelet_HLH_ngtdm_Contrast, original_firstorder_Median, original_ngtdm_Strength, wavelet_LLH_glcm_Imc2, wavelet_LLH_firstorder_Maximum,wavelet_LLL_firstorder_Energy,wavelet_HHL_firstorder_Median,original_glcm_ClusterShade,wavelet_HLH_glszm_GrayLevelNonUniformityNormalized,wavelet_HLH_glrlm_LongRunHighGrayLevelEmphasis,original_firstorder_90Percentile,original_firstorder_Energy,wavelet_HLH_glszm_LargeAreaHighGrayLevelEmphasis,wavelet_LLH_glcm_MCC,Operative_time,wavelet_LLH_gldm_GrayLevelNonUniformity]).reshape(1, -1)

        # 模型预测
        prediction = stacking_regressor.predict(input_array)[0]

        # 显示预测结果
        st.success(f"预测结果：{prediction:.2f}")
    except Exception as e:
        st.error(f"预测时发生错误：{e}")

# 可视化展示
st.header("SHAP 可视化分析")
st.write("""
以下图表展示了模型的 SHAP 分析结果，包括第一层基学习器、第二层元学习器以及整个 Stacking 模型的特征贡献。
""")

# 第一层基学习器 SHAP 可视化
st.subheader("1. 第一层基学习器")
st.write("基学习器（RandomForest、XGB、LGBM 等）的特征贡献分析。")
first_layer_img = "summary_plot.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="第一层基学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第一层基学习器的 SHAP 图像文件。")

# 第二层元学习器 SHAP 可视化
st.subheader("2. 第二层元学习器")
st.write("元学习器（Linear Regression）的输入特征贡献分析。")
meta_layer_img = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer of Stacking Regressor.png"
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="第二层元学习器的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到第二层元学习器的 SHAP 图像文件。")

# 整体 Stacking 模型 SHAP 可视化
st.subheader("3. 整体 Stacking 模型")
st.write("整个 Stacking 模型的特征贡献分析。")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="整体 Stacking 模型的 SHAP 贡献分析", use_column_width=True)
except FileNotFoundError:
    st.warning("未找到整体 Stacking 模型的 SHAP 图像文件。")

# 页脚
st.markdown("---")
st.header("总结")
st.write("""
通过本页面，您可以：
1. 使用输入特征值进行实时预测。
2. 直观地理解第一层基学习器、第二层元学习器以及整体 Stacking 模型的特征贡献情况。
这些分析有助于深入理解模型的预测逻辑和特征的重要性。
""")

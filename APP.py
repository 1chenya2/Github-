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
depression = st.sidebar.number_input("特征 depression  (范围: 0-1)", min_value=0, max_value=1, value=1)
wavelet_HHL_glrlm_RunLengthNonUniformityNormalized = st.sidebar.number_input("特征 wavelet.HHL.glrlm.RunLengthNonUniformityNormalized  (范围: 0.338031-0.4071359)", min_value=0.338031, max_value=0.4071359, value=0.392841974686494)
wavelet_HHL_firstorder_Median = st.sidebar.number_input("特征 wavelet.HHL.firstorder.Median  (范围: -0.015689-0.1198817)", min_value=-0.015689, max_value=0.1198817, value=0.052452842888068)
wavelet_HHL_glszm_ZonePercentage = st.sidebar.number_input("特征 wavelet.HHL.glszm.ZonePercentage (范围: 0.003914-0.03386989)", min_value=0.003914, max_value=0.03386989, value=0.0144873000940733)
wavelet_HLH_glcm_MCC = st.sidebar.number_input("特征 wavelet.HLH.glcm.MCC (范围: 0.309678-0.7596631)", min_value=0.309678, max_value=0.7596631, value=0.634320591738265)
original_glrlm_GrayLevelNonUniformity = st.sidebar.number_input("特征 original.glrlm.GrayLevelNonUniformity (范围: 1266.161072-4581.778)", min_value=1266.161072, max_value=4581.778, value=2156.59078743461)
wavelet_LLL_glcm_ClusterProminence = st.sidebar.number_input("特征 wavelet.LLL.glcm.ClusterProminence (范围: 1316.540948-144454.3)", min_value=1316.540948, max_value=144454.3, value=5461.81471258063)
wavelet_HLH_glszm_LargeAreaHighGrayLevelEmphasis = st.sidebar.number_input("特征 wavelet.HLH.glszm.LargeAreaHighGrayLevelEmphasis (范围: 178720.643290到10806650.0)", min_value=178720.643290, max_value=10806650.0, value=932798.531407035)
wavelet_HLH_gldm_LargeDependenceHighGrayLevelEmphasis = st.sidebar.number_input("特征 wavelet.HLH.gldm.LargeDependenceHighGrayLevelEmphasis  (范围: 431.904614-5722.353)", min_value=431.904614, max_value=5722.353, value=1328.63372530573)
original_shape_VoxelVolume = st.sidebar.number_input("特征 original.shape.VoxelVolume  (范围: 11102.428979-96702.95)", min_value=11102.428979, max_value=96702.95, value=49674.0341186523)
original_glcm_Autocorrelation = st.sidebar.number_input("特征 original.glcm.Autocorrelation  (范围: 9.951094-57.70954)", min_value=9.951094, max_value=57.70954, value=13.4272036676023)
wavelet_LLH_glcm_MCC = st.sidebar.number_input("特征 wavelet.LLH.glcm.MCC   (范围: 0.775903-0.9841110)", min_value=0.775903, max_value=0.9841110, value=0.906120059179769)
ALB = st.sidebar.number_input("特征 ALB  (范围: 30.400000-50.30000)", min_value=30.400000, max_value=50.30000, value=36.5)
Operative_time = st.sidebar.number_input("特征 Operative.time   (范围: 49-550)", min_value=49, max_value=550, value=550)
# 添加预测按钮
predict_button = st.sidebar.button("进行预测")

# 主页面用于结果展示
if predict_button:
    st.header("预测结果")
    try:
        # 将输入特征转换为模型所需格式
        input_array = np.array([depression, wavelet_HHL_glrlm_RunLengthNonUniformityNormalized, wavelet_HHL_firstorder_Median, wavelet_HHL_glszm_ZonePercentage, wavelet_HLH_glcm_MCC, original_glrlm_GrayLevelNonUniformity,wavelet_LLL_glcm_ClusterProminence, wavelet_HLH_glszm_LargeAreaHighGrayLevelEmphasis,wavelet_HLH_gldm_LargeDependenceHighGrayLevelEmphasis,original_shape_VoxelVolume,original_glcm_Autocorrelation,wavelet_LLH_glcm_MCC,ALB,Operative_time]).reshape(1, -1)

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

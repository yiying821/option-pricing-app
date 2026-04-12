# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:45:55 2026

@author: erich
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

# 1. 核心期权定价引擎
def black_76_price_iv_matrix(F_grid, K, T, r, sigma_grid, opt_type):
    T = max(T, 1e-6)
    d1 = (np.log(F_grid / K) + (0.5 * sigma_grid ** 2) * T) / (sigma_grid * np.sqrt(T))
    d2 = d1 - sigma_grid * np.sqrt(T)
    
    if opt_type == 0: # Call
        price = np.exp(-r * T) * (F_grid * norm.cdf(d1) - K * norm.cdf(d2))
        intrinsic = np.maximum(0, F_grid - K)
    else: # Put
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F_grid * norm.cdf(-d1))
        intrinsic = np.maximum(0, K - F_grid)
        
    return np.where(T <= 1e-6, intrinsic, price)

# 2. UI 界面构建
st.set_page_config(page_title="期权价格矩阵预估", layout="wide")
st.title("📈 期权三维风险平铺矩阵")

# 侧边栏输入参数
with st.sidebar:
    st.header("基础合约参数")
    
    # 新增下拉框联动功能
    contract_type = st.selectbox("选择合约", ["沪深300", "上证50", "中证1000"])
    
    # 根据选择的合约自动生成默认值
    if contract_type == "沪深300":
        default_p_start, default_p_end, default_p_step = 4200, 4800, 100
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 2
    elif contract_type == "中证1000":
        default_p_start, default_p_end, default_p_step = 7300, 8500, 100
        default_iv_start, default_iv_end, default_iv_step = 15, 50, 3
    else: # 上证50
        default_p_start, default_p_end, default_p_step = 2700, 3200, 100
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 3

    K = st.number_input("行权价 (K)", value=4850.0, step=10.0)
    opt_type_str = st.selectbox("期权类型", ["认购 (Call)", "认沽 (Put)"])
    opt_type = 0 if opt_type_str == "认购 (Call)" else 1
    
    st.header("宏观/时间参数")
    # 只保留当前距离到期的实际天数
    initial_T_days = st.number_input("剩余交易天数", value=40, step=1)
    
    # 隐藏的默认参数
    r = 0.01             # 默认无风险利率
    q = 0.0              # 默认股息率（当前Black-76函数直接基于F计算，已内置逻辑）
    time_base = 255.0    # 默认年化时间基准
    
    st.subheader("标的价格范围")
    price_start = st.number_input("起始价格", value=default_p_start)
    price_end = st.number_input("结束价格", value=default_p_end)
    price_step = st.number_input("价格步长", value=default_p_step)
    
    st.subheader("隐含波动率(IV)范围")
    iv_start = st.number_input("起始 IV (%)", value=default_iv_start)
    iv_end = st.number_input("结束 IV (%)", value=default_iv_end)
    iv_step = st.number_input("IV 步长 (%)", value=default_iv_step)

# 3. 计算与展示
if st.button("开始计算矩阵", type="primary"):
    # 为了防止除以0或死循环的安全保护
    if price_step == 0: price_step = 100
    if iv_step == 0: iv_step = 1
    
    price_range = np.arange(price_start, price_end + (1 if price_step>0 else -1), price_step)
    iv_range = np.arange(iv_start/100, (iv_end+1)/100, iv_step/100)
    days_passed_list = [0, 1, 3, 5, 7, 10, 30]
    
    for day in days_passed_list:
        if day >= initial_T_days: 
            continue # 如果假设流逝天数超过了总剩余天数，则跳过不展示
            
        # 使用隐藏的 255 个交易日基准进行时间折算
        T_current = (initial_T_days - day) / time_base
        IV_grid, F_grid = np.meshgrid(iv_range, price_range)
        price_matrix = black_76_price_iv_matrix(F_grid, K, T_current, r, IV_grid, opt_type)
        
        df_matrix = pd.DataFrame(
            price_matrix.round(0),
            index=[f"{p}" for p in price_range],
            columns=[f"{int(iv*100)}%" for iv in iv_range]
        )
        
        st.subheader(f"⏳ 假设已过 【{day} 天】 (剩余 {initial_T_days - day} 天)")
        st.dataframe(df_matrix, use_container_width=True)

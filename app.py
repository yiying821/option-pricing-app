# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:45:55 2026

@author: erich
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, datetime

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

# 2. 计算交易日天数的辅助函数 (排除周末)
def get_trading_days_delta(start_date, end_date):
    if end_date < start_date:
        return 0
    # 生成日期范围
    days = pd.date_range(start=start_date, end=end_date)
    # 排除周六(5)和周日(6)
    trading_days = days[days.dayofweek < 5]
    # 返回差值（不含起点当天则 -1，根据惯例通常算头不算尾或算尾不算头）
    return max(0, len(trading_days) - 1)

# 3. UI 界面构建
st.set_page_config(page_title="期权价格矩阵预估", layout="wide")
st.title("📈 期权三维风险平铺矩阵")

with st.sidebar:
    st.header("基础合约参数")
    contract_type = st.selectbox("选择合约类型", ["沪深300", "上证50", "中证1000"])
    
    if contract_type == "沪深300":
        default_p_start, default_p_end, default_p_step = 4200, 4800, 100
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 2
    elif contract_type == "中证1000":
        default_p_start, default_p_end, default_p_step = 7300, 8500, 100
        default_iv_start, default_iv_end, default_iv_step = 15, 50, 3
    else: 
        default_p_start, default_p_end, default_p_step = 2700, 3200, 100
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 3

    K = st.number_input("行权价 (K)", value=4850.0, step=10.0)
    opt_type_str = st.selectbox("期权类型", ["认购 (Call)", "认沽 (Put)"])
    opt_type = 0 if opt_type_str == "认购 (Call)" else 1
    
    st.header("时间模拟参数")
    initial_T_days = st.number_input("剩余交易天数", value=40, min_value=1, step=1)
    
    # --- 修改点：日期选择器 ---
    use_date = st.checkbox("按目标日期模拟", value=False)
    target_n_days = 0
    
    if use_date:
        target_date = st.date_input("选择未来目标日期", value=date.today())
        today = date.today()
        # 计算交易日差额
        target_n_days = get_trading_days_delta(today, target_date)
        st.caption(f"📅 预估从今天起到目标日期约有 **{target_n_days}** 个交易日")
    else:
        target_n_days = st.number_input("假设已过去 N 天 (输入0展示默认序列)", value=0, min_value=0)

    # 隐藏默认参数
    r = 0.01             
    time_base = 255.0    
    
    st.subheader("标的价格范围")
    price_start = st.number_input("起始价格", value=default_p_start)
    price_end = st.number_input("结束价格", value=default_p_end)
    price_step = st.number_input("价格步长", value=default_p_step)
    
    st.subheader("隐含波动率(IV)范围")
    iv_start = st.number_input("起始 IV (%)", value=default_iv_start)
    iv_end = st.number_input("结束 IV (%)", value=default_iv_end)
    iv_step = st.number_input("IV 步长 (%)", value=default_iv_step)

# 4. 计算与展示
if st.button("开始计算矩阵", type="primary"):
    if price_step == 0: price_step = 100
    if iv_step == 0: iv_step = 1
    
    # 确定要展示的天数列表
    if target_n_days == 0 and not use_date:
        days_passed_list = [0, 1, 3, 5, 7, 10, 30]
    else:
        days_passed_list = [0, target_n_days]
    
    price_range = np.arange(price_start, price_end + (1 if price_step>0 else -1), price_step)
    iv_range = np.arange(iv_start/100, (iv_end+1)/100, iv_step/100)
    
    for day in days_passed_list:
        actual_day_passed = min(day, initial_T_days)
        T_current = (initial_T_days - actual_day_passed) / time_base
        
        IV_grid, F_grid = np.meshgrid(iv_range, price_range)
        price_matrix = black_76_price_iv_matrix(F_grid, K, T_current, r, IV_grid, opt_type)
        
        df_matrix = pd.DataFrame(
            price_matrix.round(0),
            index=[f"{p}" for p in price_range],
            columns=[f"{int(iv*100)}%" for iv in iv_range]
        )
        
        st.subheader(f"⏳ 假设已过 【{actual_day_passed} 天】 (剩余 {initial_T_days - actual_day_passed} 天)")
        if use_date and day == target_n_days:
            st.info(f"目标日期模拟：{target_date}")
            
        st.dataframe(df_matrix, use_container_width=True)

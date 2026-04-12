# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:45:55 2026

@author: erich
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, timedelta
import calendar

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

# 2. 辅助函数：计算交易日天数 (排除周末)
def get_trading_days_delta(start_date, end_date):
    if end_date <= start_date:
        return 0
    days = pd.date_range(start=start_date, end=end_date)
    trading_days = days[days.dayofweek < 5]
    return max(0, len(trading_days) - 1)

# 3. 辅助函数：获取特定月份的第三个星期五
def get_third_friday(year, month):
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    monthcal = c.monthdatescalendar(year, month)
    # 找到所有星期五 (index 4)
    fridays = [day for week in monthcal for day in week if day.weekday() == calendar.FRIDAY and day.month == month]
    return fridays[2] # 返回第三个

# 4. 生成未来6个月的到期日列表
def get_expiry_dates():
    expiry_list = []
    today = date.today()
    current_year = today.year
    current_month = today.month
    
    for i in range(0, 7): # 包含本月及未来6个月
        target_month = (current_month + i - 1) % 12 + 1
        target_year = current_year + (current_month + i - 1) // 12
        expiry_date = get_third_friday(target_year, target_month)
        
        # 只保留还没过期的日期
        if expiry_date >= today:
            expiry_list.append(expiry_date)
            
    return expiry_list

# 5. UI 界面构建
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
    
    # 获取到期日列表
    expiry_options = get_expiry_dates()
    # 默认值逻辑：如果本月已过第三个周五，列表第一项就是下月；如果没过，列表第二项是下月
    today = date.today()
    this_month_friday = get_third_friday(today.year, today.month)
    default_idx = 1 if (len(expiry_options) > 1 and today <= this_month_friday) else 0
    
    selected_expiry = st.selectbox("选择到期日期 (每月第三个周五)", 
                                   options=expiry_options, 
                                   index=min(default_idx, len(expiry_options)-1))
    
    # 自动计算剩余交易天数
    initial_T_days = get_trading_days_delta(today, selected_expiry)
    st.info(f"📅 距离该到期日剩余 **{initial_T_days}** 个交易日")

    # 场景模拟：已过去N天
    use_date = st.checkbox("按目标模拟日期", value=False)
    target_n_days = 0
    if use_date:
        target_date = st.date_input("选择模拟日期", value=today)
        target_n_days = get_trading_days_delta(today, target_date)
        st.caption(f"从今天起到模拟日期过去 {target_n_days} 交易日")
    else:
        target_n_days = st.number_input("假设已过去 N 天 (输入0展示默认序列)", value=0, min_value=0)

    # 隐藏默认参数
    r, time_base = 0.01, 255.0    
    
    st.subheader("标的价格范围")
    price_start = st.number_input("起始价格", value=default_p_start)
    price_end = st.number_input("结束价格", value=default_p_end)
    price_step = st.number_input("价格步长", value=default_p_step)
    
    st.subheader("隐含波动率(IV)范围")
    iv_start = st.number_input("起始 IV (%)", value=default_iv_start)
    iv_end = st.number_input("结束 IV (%)", value=default_iv_end)
    iv_step = st.number_input("IV 步长 (%)", value=default_iv_step)

# 6. 计算与展示
if st.button("开始计算矩阵", type="primary"):
    if price_step == 0: price_step = 100
    
    # 确定天数序列
    days_passed_list = [0, 1, 3, 5, 7, 10, 30] if (target_n_days == 0 and not use_date) else [0, target_n_days]
    
    price_range = np.arange(price_start, price_end + (1 if price_step>0 else -1), price_step)
    iv_range = np.arange(iv_start/100, (iv_end+1)/100, iv_step/100)
    
    for day in days_passed_list:
        # 保护：流逝天数不能超过总剩余交易天数
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
        st.dataframe(df_matrix, use_container_width=True)

# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm

# 1. 核心期权定价引擎 (Black-Scholes 模型)
def black_scholes_matrix(S, K, T, r, q, sigma, opt_type):
    """
    S: 标的现货价格
    K: 行权价
    T: 剩余期限(年)
    r: 无风险利率
    q: 股息率
    sigma: 隐含波动率
    opt_type: 0 为 Call, 1 为 Put
    """
    # 处理临近到期或已到期的情况
    if T <= 1e-6:
        if opt_type == 0:
            return np.maximum(0, S - K)
        else:
            return np.maximum(0, K - S)
            
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if opt_type == 0: # Call
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else: # Put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
    return price

# 2. UI 界面构建
st.set_page_config(page_title="中证1000期权定价矩阵", layout="wide")
st.title("📈 股指期权三维风险矩阵")
st.caption("注：针对 MO2609 等远月合约，请务必准确填写‘初始剩余天数’（2609合约通常大于500天）")

# 侧边栏输入参数
with st.sidebar:
    st.header("1. 基础合约参数")
    K = st.number_input("行权价 (K)", value=9600.0, step=50.0)
    opt_type_str = st.selectbox("期权类型", ["认购 (Call)", "认沽 (Put)"])
    opt_type = 0 if opt_type_str == "认购 (Call)" else 1
    
    st.header("2. 宏观/时间参数")
    initial_T_days = st.number_input("当前距离到期的实际天数", value=540, step=1)
    r = st.number_input("无风险利率 (r) - 例: 0.02 为 2%", value=0.025, format="%.4f")
    q = st.number_input("指数股息率 (q) - 例: 0.015 为 1.5%", value=0.020, format="%.4f")
    day_convention = st.selectbox("年化时间基准 (天/年)", [365, 360, 243], index=0)
    
    st.header("3. 矩阵扫描范围")
    col1, col2 = st.columns(2)
    with col1:
        price_start = st.number_input("价格起始", value=6000)
        price_end = st.number_input("价格结束", value=5500)
    with col2:
        iv_start = st.number_input("IV起始(%)", value=20)
        iv_end = st.number_input("IV结束(%)", value=40)
    
    price_step = st.number_input("价格步长", value=-100)
    iv_step = st.number_input("IV 步长(%)", value=5)

# 3. 计算与展示
if st.button("生成定价预测矩阵", type="primary"):
    # 构建范围（处理步长正负问题）
    p_step = -abs(price_step) if price_start > price_end else abs(price_step)
    price_range = np.arange(price_start, price_end + (1 if p_step > 0 else -1), p_step)
    
    iv_range = np.arange(iv_start/100, (iv_end + 0.1)/100, iv_step/100)
    
    # 模拟时间流逝
    days_passed_list = [0, 30, 45,60,90, 180, 360]
    
    tabs = st.tabs([f"经过 {d} 天" for d in days_passed_list])
    
    for i, day in enumerate(days_passed_list):
        with tabs[i]:
            remaining_days = initial_T_days - day
            if remaining_days < 0:
                st.warning(f"合约已在 {day} 天前到期")
                continue
                
            T_current = remaining_days / float(day_convention)
            
            # 生成网格
            IV_grid, S_grid = np.meshgrid(iv_range, price_range)
            
            # 向量化计算
            price_matrix = black_scholes_matrix(S_grid, K, T_current, r, q, IV_grid, opt_type)
            
            # 转换为 DataFrame
            df_matrix = pd.DataFrame(
                price_matrix.round(1),
                index=[f"标的:{p}" for p in price_range],
                columns=[f"IV:{int(iv*100)}%" for iv in iv_range]
            )
            
            st.subheader(f"⏳ 剩余时间: {remaining_days} 天 (T = {T_current:.4f} 年)")
            st.dataframe(df_matrix, use_container_width=True)

st.divider()
st.info("""
**使用建议：**
1. **MO2609 合约**：属于远期深度价外合约（假设目前指数在 5000-6000 左右），其价格对 **r** 和 **q** 非常敏感。
2. **波动率偏斜**：远月合约的 IV 通常比近月稳定，但 9600 行权价属于虚值，其实际 IV 可能显著高于平值合约。
3. **单位换算**：显示的数值为期权点数，实际价值需乘以 100（MO合约乘数）。
""")

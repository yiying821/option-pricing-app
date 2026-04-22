# -*- coding: utf-8 -*-
"""
【纯股指现货定价版】
适配标的：A股股指期权（沪深300/上证50/中证1000）→ 标的物为【指数价格】，非期货价格
定价模型：Black-Scholes 欧式现货期权（含股息率，交易所标准模型）
核心修正：全流程基于指数点位计算，修复价格虚高问题
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, datetime
import calendar

# ===================== 核心定价函数：股指现货期权（指数价格专用）=====================
def bs_index_option(S_index, K, T, r, q, sigma, opt_type):
    """
    专用函数：根据【股指现货价格】计算期权价格
    S_index: 股指现货点位（指数价格，核心输入）
    K: 行权价
    T: 年化剩余时间
    r: 无风险利率
    q: 指数年化股息率
    sigma: 隐含波动率
    opt_type: 0=认购 1=认沽
    """
    T = max(T, 1e-6)
    # Black-Scholes 股指现货期权核心公式
    d1 = (np.log(S_index / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if opt_type == 0:
        price = S_index * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S_index * np.exp(-q * T) * norm.cdf(-d1)
    
    # 到期后返回内在价值
    intrinsic = np.maximum(0, S_index - K) if opt_type == 0 else np.maximum(0, K - S_index)
    return np.where(T <= 1e-6, intrinsic, price)

# ===================== A股市场标准工具函数 =====================
def get_fourth_wednesday(year, month):
    """A股股指期权到期日：到期月第四个星期三（官方规则）"""
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    monthcal = c.monthdatescalendar(year, month)
    wednesdays = [day for week in monthcal for day in week if day.weekday() == 2 and day.month == month]
    return wednesdays[3] if len(wednesdays) >= 4 else wednesdays[-1]

def get_expiry_dates():
    """生成有效到期日列表"""
    expiry_list = []
    today = date.today()
    current_year, current_month = today.year, today.month
    for i in range(7):
        m = (current_month + i - 1) % 12 + 1
        y = current_year + (current_month + i - 1) // 12
        expiry = get_fourth_wednesday(y, m)
        if expiry >= today:
            expiry_list.append(expiry)
    return expiry_list

def get_trading_days(start, end):
    """计算剩余交易日（A股：242天/年，标准算法）"""
    days = pd.date_range(start, end, freq='B')
    return len(days)

# ===================== UI界面 =====================
st.set_page_config(page_title="股指期权定价（指数现货版）", layout="wide")
st.title("📊 股指期权定价工具【基于：股指现货（指数）价格】")
st.markdown("### 适配标的：沪深300/上证50/中证1000 股指期权 | 定价依据：**指数实时点位**")

with st.sidebar:
    st.header("1. 基础参数（指数现货版）")
    contract = st.selectbox("合约类型", ["沪深300", "上证50", "中证1000"])
    
    # 市场默认参数（修复价格虚高：低利率+高股息率+合理IV）
    if contract == "沪深300":
        default_index = 4750  # 默认指数点位（现货）
        default_K = 4750
        iv_low, iv_high, iv_step = 12, 20, 1
        q_default = 0.030  # 股息率3.0%
        r_default = 0.015  # 无风险利率1.5%（降低，避免虚高）
    elif contract == "中证1000":
        default_index = 7700
        default_K = 7700
        iv_low, iv_high, iv_step = 18, 28, 1
        q_default = 0.032
        r_default = 0.016
    else:
        default_index = 2950
        default_K = 2950
        iv_low, iv_high, iv_step = 12, 20, 1
        q_default = 0.030
        r_default = 0.015

    # 核心输入：【股指现货价格】（指数点位）
    S_index = st.number_input("📈 股指现货价格（指数点位）", value=default_index, step=50)
    K = st.number_input("行权价", value=default_K, step=50)
    opt_type = st.radio("期权类型", ["认购", "认沽"], index=0)
    opt_code = 0 if opt_type == "认购" else 1

    st.header("2. 利率参数（修正虚高）")
    r = st.number_input("无风险利率(%)", value=r_default*100, step=0.1)/100
    q = st.number_input("指数股息率(%)", value=q_default*100, step=0.1)/100

    st.header("3. 到期时间")
    expiry_list = get_expiry_dates()
    selected_expiry = st.selectbox("到期日（第四个周三）", expiry_list)
    today = date.today()
    remaining_days = get_trading_days(today, selected_expiry)
    T = remaining_days / 242  # A股标准年化时间（无复杂计算，避免虚高）
    st.info(f"剩余交易日：{remaining_days} 天 | 年化时间：{T:.3f}")

    st.header("4. 波动率&价格区间")
    iv_start = st.number_input("起始IV(%)", value=iv_low)
    iv_end = st.number_input("结束IV(%)", value=iv_high)
    iv_step = st.number_input("IV步长(%)", value=iv_step)

# ===================== 计算逻辑 =====================
if st.button("计算期权价格", type="primary"):
    # 生成参数网格
    iv_range = np.arange(iv_start/100, iv_end/100 + 0.001, iv_step/100)
    index_range = np.array([S_index])  # 固定：仅用【指数现货价格】计算

    # 定价计算
    iv_grid, s_grid = np.meshgrid(iv_range, index_range)
    price = bs_index_option(s_grid, K, T, r, q, iv_grid, opt_code)

    # 输出结果
    st.markdown(f"## 定价结果")
    st.markdown(f"**标的：{contract} 指数 | 现货价格：{S_index} 点**")
    st.markdown(f"**期权类型：{opt_type} | 行权价：{K} | 到期：{selected_expiry}**")
    
    df = pd.DataFrame(
        price.round(2),
        index=[f"指数:{int(S_index)}"],
        columns=[f"IV:{int(x*100)}%" for x in iv_range]
    )
    st.dataframe(df, use_container_width=True)

    # 偏差提示
    st.caption("✅ 本结果**纯基于指数现货价格**计算，无期货参数干扰，贴合交易所定价规则")
# -*- coding: utf-8 -*-
"""
修正说明：
1. 模型替换：Black-76 → Black-Scholes（适配A股股指现货期权，加入股息率）
2. 到期日规则：第三个周五 → 第四个周三（符合沪深300/上证50/中证1000股指期权规则）
3. 时间参数：年化基数255→242（A股实际交易日）、精确到小时的年化时间计算
4. 波动率：修复IV区间截断错误、避免步长溢出
5. 新增：股息率参数（按合约类型适配默认值）、无风险利率可选（默认用对应期限合理值）
"""
import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, datetime, timedelta
import calendar

# 1. 核心期权定价引擎（替换为Black-Scholes，适配A股股指现货期权）
def black_scholes_price_iv_matrix(S_grid, K, T, r, q, sigma_grid, opt_type):
    T = max(T, 1e-6)  # 避免T=0导致除零错误
    # Black-Scholes核心公式（加入股息率q）
    d1 = (np.log(S_grid / K) + (r - q + 0.5 * sigma_grid ** 2) * T) / (sigma_grid * np.sqrt(T))
    d2 = d1 - sigma_grid * np.sqrt(T)
    
    if opt_type == 0:  # Call 认购
        price = np.exp(-r * T) * (S_grid * np.exp(-q * T) * norm.cdf(d1) - K * norm.cdf(d2))
        intrinsic = np.maximum(0, S_grid - K)
    else:  # Put 认沽
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - S_grid * np.exp(-q * T) * norm.cdf(-d1))
        intrinsic = np.maximum(0, K - S_grid)
        
    # 到期时返回内在价值，否则返回计算价格
    return np.where(T <= 1e-6, intrinsic, price)

# 2. 辅助函数：计算交易日天数 (排除周末)
def get_trading_days_delta(start_date, end_date):
    if end_date <= start_date:
        return 0
    days = pd.date_range(start=start_date, end=end_date)
    trading_days = days[days.dayofweek < 5]
    return max(0, len(trading_days) - 1)

# 3. 辅助函数：获取A股股指期权到期日（到期月第四个星期三）
def get_fourth_wednesday(year, month):
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    monthcal = c.monthdatescalendar(year, month)
    # 筛选当月所有星期三（weekday=2）
    wednesdays = [day for week in monthcal for day in week if day.weekday() == calendar.WEDNESDAY and day.month == month]
    # 不足4个则取最后一个，否则取第四个
    return wednesdays[3] if len(wednesdays) >= 4 else wednesdays[-1]

# 4. 生成未来6个月的到期日列表（适配A股规则）
def get_expiry_dates():
    expiry_list = []
    today = date.today()
    current_year = today.year
    current_month = today.month
    
    for i in range(0, 7):  # 包含本月及未来6个月
        target_month = (current_month + i - 1) % 12 + 1
        target_year = current_year + (current_month + i - 1) // 12
        expiry_date = get_fourth_wednesday(target_year, target_month)
        
        # 只保留未过期的日期
        if expiry_date >= today:
            expiry_list.append(expiry_date)
            
    return expiry_list

# 5. 辅助函数：精确计算年化到期时间（包含小时/分钟，适配A股交易时间）
def get_annualized_T(start_datetime, expiry_datetime, time_base=242):
    """
    start_datetime: 当前时间（datetime）
    expiry_datetime: 到期时间（datetime，固定为到期日15:00）
    time_base: 年化交易日基数（A股实际约242天）
    """
    if expiry_datetime <= start_datetime:
        return 1e-6
    # 计算总秒数差
    total_seconds = (expiry_datetime - start_datetime).total_seconds()
    # A股交易时间：9:30-15:00，每天5.5小时=19800秒
    annual_trading_seconds = time_base * 5.5 * 3600
    # 年化时间（按交易时间占比）
    return total_seconds / annual_trading_seconds

# 6. UI 界面构建
st.set_page_config(page_title="期权价格矩阵预估", layout="wide")
st.title("📈 期权三维风险平铺矩阵（A股股指适配版）")

with st.sidebar:
    st.header("基础合约参数")
    contract_type = st.selectbox("选择合约类型", ["沪深300", "上证50", "中证1000"])
    
    # 根据合约类型设置默认参数（适配A股实际行权价区间）
    if contract_type == "沪深300":
        default_K = 4850
        default_p_start, default_p_end, default_p_step = 4200, 4800, 50  # 步长从100→50（匹配实际行权价间隔）
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 2
        default_q = 0.025  # 沪深300年化股息率
        default_r = 0.02   # 对应期限国债逆回购利率
    elif contract_type == "中证1000":
        default_K = 7800
        default_p_start, default_p_end, default_p_step = 7300, 8500, 50
        default_iv_start, default_iv_end, default_iv_step = 15, 50, 3
        default_q = 0.03   # 中证1000年化股息率
        default_r = 0.022  # 中证1000对应无风险利率
    else:  # 上证50
        default_K = 3000
        default_p_start, default_p_end, default_p_step = 2700, 3200, 50
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 3
        default_q = 0.028  # 上证50年化股息率
        default_r = 0.019  # 上证50对应无风险利率

    # 基础参数输入
    K = st.number_input("行权价 (K)", value=int(default_K), step=50)  # 步长匹配实际行权价
    opt_type_str = st.selectbox("期权类型", ["认购 (Call)", "认沽 (Put)"])
    opt_type = 0 if opt_type_str == "认购 (Call)" else 1
    
    # 新增：股息率和无风险利率可手动调整
    st.subheader("利率/股息率参数")
    r = st.number_input("无风险利率 (r, %)", value=default_r*100, step=0.1, format="%.1f") / 100
    q = st.number_input("年化股息率 (q, %)", value=default_q*100, step=0.1, format="%.1f") / 100
    
    st.header("时间模拟参数")
    # 获取修正后的到期日列表（第四个星期三）
    expiry_options = get_expiry_dates()
    # 默认到期日逻辑（适配新规则）
    today = date.today()
    this_month_wed = get_fourth_wednesday(today.year, today.month)
    default_idx = 1 if (len(expiry_options) > 1 and today <= this_month_wed) else 0
    
    selected_expiry = st.selectbox(
        "选择到期日期 (每月第四个星期三)", 
        options=expiry_options, 
        index=min(default_idx, len(expiry_options)-1)
    )
    # 到期时间固定为15:00（A股收盘）
    expiry_datetime = datetime.combine(selected_expiry, datetime.strptime("15:00", "%H:%M").time())
    current_datetime = datetime.now()
    
    # 计算剩余交易日和精确年化时间
    initial_T_days = get_trading_days_delta(today, selected_expiry)
    initial_T = get_annualized_T(current_datetime, expiry_datetime, time_base=242)
    st.info(f"""📅 到期日：{selected_expiry} 15:00
剩余交易日：{initial_T_days} 天
精确年化时间：{initial_T:.4f} 年""")

    # 场景模拟：已过去N天
    use_date = st.checkbox("按目标模拟日期", value=False)
    target_n_days = 0
    if use_date:
        target_date = st.date_input("选择模拟日期", value=today)
        target_n_days = get_trading_days_delta(today, target_date)
        st.caption(f"从今天起到模拟日期过去 {target_n_days} 交易日")
    else:
        target_n_days = st.number_input("假设已过去 N 天 (输入0展示默认序列)", value=0, min_value=0)

    # 标的价格范围（步长优化）
    st.subheader("标的价格范围")
    price_start = st.number_input("起始价格", value=default_p_start, step=50)
    price_end = st.number_input("结束价格", value=default_p_end, step=50)
    price_step = st.number_input("价格步长", value=default_p_step, step=10)
    
    # 隐含波动率(IV)范围（修复区间截断错误）
    st.subheader("隐含波动率(IV)范围")
    iv_start = st.number_input("起始 IV (%)", value=default_iv_start)
    iv_end = st.number_input("结束 IV (%)", value=default_iv_end)
    iv_step = st.number_input("IV 步长 (%)", value=default_iv_step)

# 7. 计算与展示
if st.button("开始计算矩阵", type="primary"):
    # 价格范围修正：避免步长为0，确保包含结束价格
    price_step = price_step if price_step != 0 else 50
    price_range = np.arange(price_start, price_end + price_step, price_step)
    
    # IV范围修正：避免溢出，精确截断到iv_end
    iv_range = np.arange(iv_start/100, iv_end/100 + iv_step/100, iv_step/100)
    iv_range = iv_range[iv_range <= iv_end/100]  # 截断超出上限的IV值
    
    # 确定天数序列
    days_passed_list = [0, 1, 3, 5, 7, 10, 30] if (target_n_days == 0 and not use_date) else [0, target_n_days]
    
    for day in days_passed_list:
        # 保护：流逝天数不能超过总剩余交易天数
        actual_day_passed = min(day, initial_T_days)
        # 模拟已过N天后的时间：按交易日推算目标时间
        target_datetime = current_datetime + timedelta(days=actual_day_passed)
        # 重新计算剩余年化时间
        T_current = get_annualized_T(target_datetime, expiry_datetime, time_base=242)
        
        # 构建网格并计算价格矩阵（使用修正后的Black-Scholes）
        IV_grid, S_grid = np.meshgrid(iv_range, price_range)
        price_matrix = black_scholes_price_iv_matrix(S_grid, K, T_current, r, q, IV_grid, opt_type)
        
        # 格式化输出（保留1位小数，更贴合实际价格）
        df_matrix = pd.DataFrame(
            price_matrix.round(1),
            index=[f"{int(p)}" for p in price_range],
            columns=[f"{int(iv*100)}%" for iv in iv_range]
        )
        
        st.subheader(f"⏳ 假设已过 【{actual_day_passed} 天】 (剩余 {initial_T_days - actual_day_passed} 天)")
        st.dataframe(df_matrix, use_container_width=True)
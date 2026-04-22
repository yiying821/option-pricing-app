import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import date, datetime, timedelta
import calendar

# =====================
# 【修改1：中金所实盘BS——使用期货远期F定价，不再用现货S】
# =====================
def cffex_option_price(F, K, T, r, sigma, opt_type):
    """
    中金所官方做市商用公式
    F : 股指期货远期价格（核心！替代现货）
    q 不再参与计算，远期已经包含股息与无风险利率
    """
    T = max(T, 1e-6)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if opt_type == 0:
        # 认购
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        # 认沽
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return price

# =====================
# 【修改2：交易所标准交易日统计，剔除周末】
# =====================
def get_trading_days(start_date, end_date):
    if end_date <= start_date:
        return 0
    # 直接获取A股交易日序列
    trade_days = pd.date_range(start=start_date, end=end_date, freq="B")
    return len(trade_days)

# 到期日规则（原逻辑正确，无修改）
def get_fourth_wednesday(year, month):
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    monthcal = c.monthdatescalendar(year, month)
    wednesdays = [day for week in monthcal for day in week if day.weekday() == 2 and day.month == month]
    return wednesdays[3] if len(wednesdays)>=4 else wednesdays[-1]

def get_expiry_dates():
    expiry_list = []
    today = date.today()
    for i in range(0,7):
        m = (today.month + i -1) %12 +1
        y = today.year + (today.month + i -1)//12
        exp_day = get_fourth_wednesday(y,m)
        if exp_day >= today:
            expiry_list.append(exp_day)
    return expiry_list

# UI界面（仅参数微调，主体无修改）
st.set_page_config(page_title="中金所做市商定价版｜股指期权", layout="wide")
st.title("📈 中金所标准｜股指期权矩阵（期货远期定价·实盘对齐）")

with st.sidebar:
    st.header("基础合约参数")
    contract_type = st.selectbox("选择合约类型", ["沪深300", "上证50", "中证1000"])

    if contract_type == "沪深300":
        default_K = 4850
        default_S_start, default_S_end, default_S_step = 4200, 5200, 50
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 2
        default_r = 0.02
        default_q = 0.012
    elif contract_type == "中证1000":
        default_K = 7800
        default_S_start, default_S_end, default_S_step = 7300, 8500, 50
        default_iv_start, default_iv_end, default_iv_step = 15, 30, 2
        default_r = 0.022
        default_q = 0.018
    else:
        default_K = 3000
        default_S_start, default_S_end, default_S_step = 2700, 3200, 50
        default_iv_start, default_iv_end, default_iv_step = 10, 25, 2
        default_r = 0.019
        default_q = 0.015

    K = st.number_input("行权价 (K)", value=int(default_K), step=50)
    opt_type_str = st.selectbox("期权类型", ["认购 (Call)", "认沽 (Put)"])
    opt_type = 0 if opt_type_str == "认购 (Call)" else 1

    st.subheader("利率/股息（用于计算期货远期F）")
    r = st.number_input("无风险利率 (r, %)", value=default_r*100, step=0.1, format="%.1f") / 100
    q = st.number_input("年化股息率 (q, %)", value=default_q*100, step=0.1, format="%.1f") / 100

    # 到期日选择
    expiry_options = get_expiry_dates()
    today = date.today()
    selected_expiry = st.selectbox("选择到期日期", options=expiry_options, index=0)

    # =====================
    # 【修改3：年化T彻底改为交易所标准：剩余交易日 / 242】
    # =====================
    remain_trade_days = get_trading_days(today, selected_expiry)
    T_origin = remain_trade_days / 242
    st.info(f"""📅 到期日：{selected_expiry}
剩余交易日：{remain_trade_days} 天
交易所标准年化T：{T_origin:.4f}""")

    # 交易日时间衰减
    st.subheader("时间衰减（交易日流逝）")
    pass_trade_days = st.number_input("已流逝交易日", value=0, min_value=0)

    # 标的现货价格区间
    st.subheader("现货指数价格区间（自动换算期货远期F）")
    S_start = st.number_input("起始现货", value=default_S_start, step=50)
    S_end = st.number_input("结束现货", value=default_S_end, step=50)
    S_step = st.number_input("现货步长", value=default_S_step, step=10)

    st.subheader("隐含波动率IV区间")
    iv_start = st.number_input("起始 IV (%)", value=default_iv_start)
    iv_end = st.number_input("结束 IV (%)", value=default_iv_end)
    iv_step = st.number_input("IV 步长 (%)", value=default_iv_step)

# =====================
# 【修改4：核心计算逻辑：现货S → 自动换算期货远期F 再算期权价格】
# =====================
if st.button("🚀 中金所实盘计算", type="primary"):
    S_range = np.arange(S_start, S_end + S_step, S_step)
    iv_range = np.arange(iv_start/100, iv_end/100 + iv_step/100, iv_step/100)

    # 真实剩余年化时间
    real_remain_day = max(remain_trade_days - pass_trade_days, 0)
    T_now = real_remain_day / 242

    # 构建结果表
    df_result = pd.DataFrame()

    for spot_S in S_range:
        row_data = []
        # 中金所公式：现货 → 换算股指期货远期价格F
        F = spot_S * np.exp((r - q) * T_now)
        
        for iv in iv_range:
            # 用【期货远期F】定价，完全对齐做市商
            opt_price = cffex_option_price(F, K, T_now, r, iv, opt_type)
            row_data.append(round(opt_price,2))

        df_result.loc[f"{int(spot_S)}现货｜F={round(F,1)}",
                     [f"{round(iv*100,1)}%" for iv in iv_range]] = row_data

    st.subheader(f"✅中金所做市商定价｜已流逝{pass_trade_days}交易日｜剩余{real_remain_day}天")
    st.dataframe(df_result, use_container_width=True)
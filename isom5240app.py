
# import part
import streamlit as st
from transformers import pipeline

# function part

# main part
# set up page configuration
st.set_page_config(
    page_title="ISOM5240: Sentiment Analysis App",
    page_icon="🤖",
    layout="contered"
)

# 1. 頁面標題
st.title("🤖 英文情緒分析工具")

# 2. 建立情緒分析 Pipeline（每次網頁刷新時直接載入）
sentiment_pipeline = pipeline("sentiment-analysis")

# 3. 預設範例文本
default_text = """Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI)."""

# 4. 文字輸入框
text = st.text_area("請輸入一段英文：", value=default_text, height=120)

# 5. 點擊按鈕執行
if st.button("開始分析"):
    if text.strip():
        # 執行推論
        result = sentiment_pipeline(text)
        
        # 拆解結構
        label = result[0]["label"]
        score = result[0]["score"]
        
        # 輸出結果
        st.write(f"**情緒結果 (Sentiment):** {label}")
        st.write(f"**信心度 (Score):** {score:.4f}")
    else:
        st.warning("請先輸入內容再按分析！")

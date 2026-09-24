# import part
import streamlit as st
from transformers import pipeline

# ── page configuration (必須是第一個 Streamlit 指令) ──
st.set_page_config(
    page_title="ISOM5240: Sentiment Analysis App",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# Function Part (工具箱定義)
# ==========================================

def load_sentiment_pipeline(model_name: str):
    """
    載入指定模型的情緒分析 Pipeline。
    """
    return pipeline("text-classification", model=model_name)


def analyze_sentiment(classifier, input_text: str):
    """
    接收文字原料並送入模型進行推論，回傳解包後的標籤與信心分數。
    """
    result = classifier(input_text)
    label = result[0]["label"]
    score = result[0]["score"]
    return label, score


# ==========================================
# Main Part (主流程與介面呈現)
# ==========================================

def main():
    # 1. 頁面標題
    st.title("🤖 英文情緒分析工具")

    # 2. 指定輕巧且準確的專業模型並載入
    MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_pipeline = load_sentiment_pipeline(model_name=MODEL_NAME)

    # 3. 預設範例文本
    default_text = """Deep Learning (DL) represents a highly promising approach to developing applications in Artificial Intelligence (AI)."""

    # 4. 文字輸入框
    text = st.text_area("請輸入一段英文：", value=default_text, height=120)

    # 5. 點擊按鈕執行
    if st.button("開始分析"):
        if text.strip():
            # 呼叫分析函式，拿到回傳成果
            label, score = analyze_sentiment(classifier=sentiment_pipeline, input_text=text)

            # 輸出結果
            st.write(f"**情緒結果 (Sentiment):** {label}")
            st.write(f"**信心度 (Score):** {score:.4f}")
        else:
            st.warning("請先輸入內容再按分析！")


if __name__ == "__main__":
    main()

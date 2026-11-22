from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import helper
import preprocessor
import sentiment

st.set_page_config(page_title="WhatsApp Chat & Sentiment Analyzer", layout="wide")
st.sidebar.title("WhatsApp Chat & Sentiment Analyzer")

uploaded_file = st.sidebar.file_uploader("Upload exported WhatsApp .txt file", type=["txt"])
use_sample = st.sidebar.checkbox("Use built-in sample chat", value=uploaded_file is None)

raw_data = None
if uploaded_file is not None:
    raw_data = uploaded_file.getvalue().decode("utf-8", errors="ignore")
elif use_sample and Path("sample_chat.txt").exists():
    raw_data = Path("sample_chat.txt").read_text(encoding="utf-8")

if raw_data:
    df = preprocessor.preprocess(raw_data)

    if df.empty:
        st.warning("No valid WhatsApp messages were detected in the uploaded file.")
    else:
        user_list = [u for u in df["user"].unique().tolist() if u != "group_notification"]
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.sidebar.selectbox("Analyze participant", user_list)

        if st.sidebar.button("Run Analysis", type="primary") or use_sample:
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

            st.subheader("Conversation Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Messages", num_messages)
            c2.metric("Total Words", words)
            c3.metric("Media Shared", num_media_messages)
            c4.metric("Links Shared", num_links)

            # Sentiment & Toxicity Breakdown
            filtered_df = df if selected_user == "Overall" else df[df["user"] == selected_user]
            sent_report = sentiment.analyze_records(filtered_df.to_dict("records"))

            st.subheader("Sentiment & Toxicity Telemetry")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Avg Polarity", f"{sent_report['average_score']:+.2f}")
            s2.metric("Positive", sent_report["distribution"]["positive"])
            s3.metric("Neutral", sent_report["distribution"]["neutral"])
            s4.metric("Negative", sent_report["distribution"]["negative"])
            s5.metric("Flagged Toxic", sent_report["toxic_messages"])

            col_l, col_r = st.columns(2)
            with col_l:
                fig, ax = plt.subplots()
                dist = sent_report["distribution"]
                ax.bar(dist.keys(), dist.values(), color=["#16A34A", "#64748B", "#DC2626"])
                ax.set_title("Sentiment Distribution")
                st.pyplot(fig)
            with col_r:
                st.write("Per-Participant Sentiment Summary")
                st.json(sent_report["by_user"])

            # Timelines
            st.subheader("Activity Timelines")
            t1, t2 = st.columns(2)
            with t1:
                timeline = helper.monthly_timeline(selected_user, df)
                fig, ax = plt.subplots()
                ax.plot(timeline["time"], timeline["message"], color="#15803D")
                plt.xticks(rotation="vertical")
                ax.set_title("Monthly Timeline")
                st.pyplot(fig)
            with t2:
                daily = helper.daily_timeline(selected_user, df)
                fig, ax = plt.subplots()
                ax.plot(daily["only_date"], daily["message"], color="#0F172A")
                plt.xticks(rotation="vertical")
                ax.set_title("Daily Timeline")
                st.pyplot(fig)

            # Heatmap
            st.subheader("Weekly Activity Heatmap")
            user_heatmap = helper.activity_heatmap(selected_user, df)
            if not user_heatmap.empty:
                fig, ax = plt.subplots()
                sns.heatmap(user_heatmap, ax=ax, cmap="YlGn")
                st.pyplot(fig)

            # Wordcloud & Common Words
            w1, w2 = st.columns(2)
            with w1:
                st.subheader("Wordcloud")
                df_wc = helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots()
                ax.imshow(df_wc)
                ax.axis("off")
                st.pyplot(fig)
            with w2:
                st.subheader("Most Common Words")
                common_df = helper.most_common_words(selected_user, df)
                if not common_df.empty:
                    fig, ax = plt.subplots()
                    ax.barh(common_df[0], common_df[1], color="#15803D")
                    st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ---------- Page config (use the same icon everywhere) ----------
st.set_page_config(
    page_title="Amazon Product Reviews Sentiment Analysis",
    page_icon="attached_assets/amazon_logo.png",  # make sure this file exists
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Helpers ----------
def title_with_icon(title_text: str, icon_path: str = "attached_assets/amazon_logo.png", icon_width: int = 108):
    """Renders a title with a small icon; falls back to emoji if the image is missing."""
    if Path(icon_path).exists():
        col_logo, col_title = st.columns([0.08, 1])
        with col_logo:
            st.image(icon_path, width=icon_width)
        with col_title:
            st.markdown(
                f"<h1 style='margin:0; line-height:1.1;'>{title_text}</h1>",
                unsafe_allow_html=True
            )
    else:
        st.title("🛒 " + title_text)

@st.cache_data
def load_data(sample_size=50000):
    df = pd.read_csv('attached_assets/Reviews.csv')
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    df_sample['Text'] = df_sample['Text'].fillna("")
    df_sample['Summary'] = df_sample['Summary'].fillna("")
    df_sample['Score'] = df_sample['Score'].fillna(df_sample['Score'].median())
    
    df_sample = df_sample.drop_duplicates(subset=['ProductId', 'Time', 'Text'])
    
    df_sample['Date'] = pd.to_datetime(df_sample['Time'], unit='s')
    df_sample['Year'] = df_sample['Date'].dt.year
    df_sample['Month'] = df_sample['Date'].dt.month
    df_sample['YearMonth'] = df_sample['Date'].dt.to_period('M').astype(str)
    
    return df_sample

@st.cache_data
def perform_sentiment_analysis(df):
    df = df.copy()
    
    def get_sentiment(text):
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0, 0
    
    sample_for_sentiment = df.head(10000)
    
    sentiments = sample_for_sentiment['Text'].apply(get_sentiment)
    sample_for_sentiment['Polarity'] = sentiments.apply(lambda x: x[0])
    sample_for_sentiment['Subjectivity'] = sentiments.apply(lambda x: x[1])
    
    sample_for_sentiment['Sentiment'] = sample_for_sentiment['Polarity'].apply(
        lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
    )
    
    df.loc[sample_for_sentiment.index, 'Polarity'] = sample_for_sentiment['Polarity']
    df.loc[sample_for_sentiment.index, 'Subjectivity'] = sample_for_sentiment['Subjectivity']
    df.loc[sample_for_sentiment.index, 'Sentiment'] = sample_for_sentiment['Sentiment']
    
    df['Polarity'] = df['Polarity'].fillna(0)
    df['Subjectivity'] = df['Subjectivity'].fillna(0)
    df['Sentiment'] = df['Sentiment'].fillna('Neutral')
    
    return df

@st.cache_data
def generate_wordcloud_data(df, sentiment_filter=None):
    if sentiment_filter:
        text = ' '.join(df[df['Sentiment'] == sentiment_filter]['Text'].head(1000))
    else:
        text = ' '.join(df['Text'].head(1000))
    
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                              colormap='viridis', max_words=100).generate(text)
        return wordcloud
    return None

def predict_sentiment(text):
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = 'Positive'
        elif polarity < -0.1:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        return sentiment, polarity, subjectivity
    except:
        return 'Neutral', 0, 0

# ---------- Navigation ----------
st.sidebar.title("🛒 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Data Exploration", 
        "Sentiment Analysis",
        "Trends & Patterns",
        "Sample Reviews",
        "ML Performance",
        "Data Processing",
        "Sentiment Predictor",
        "Product Analysis",
        "Period Comparison",
    ],
    key="nav_page"
)

df = load_data()
df_with_sentiment = perform_sentiment_analysis(df)

# ---------- Pages ----------
if page == "Overview":
    title_with_icon("Amazon Product Reviews – Sentiment Analysis")
    st.markdown("---")
    
    st.markdown("""
    ### Project Overview
    This dashboard presents a comprehensive sentiment analysis of **Amazon Product Reviews**. 
    The project analyzes customer reviews to uncover insights about product satisfaction, 
    sentiment trends, and review helpfulness.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews", f"{len(df):,}")
    with col2:
        st.metric("Average Rating", f"{df['Score'].mean():.2f}")
    with col3:
        st.metric("Time Span", f"{df['Year'].min()} - {df['Year'].max()}")
    with col4:
        st.metric("Unique Products", f"{df['ProductId'].nunique():,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(" Score Distribution")
        score_counts = df['Score'].value_counts().sort_index()
        fig = px.bar(x=score_counts.index, y=score_counts.values,
                     labels={'x': 'Rating', 'y': 'Count'},
                     color=score_counts.values,
                     color_continuous_scale='blues')
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Reviews Over Time")
        reviews_by_year = df.groupby('Year').size().reset_index(name='Count')
        fig = px.line(reviews_by_year, x='Year', y='Count', markers=True)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader(" Data Processing Pipeline")
    pipeline_cols = st.columns(5)
    with pipeline_cols[0]:
        st.info("**1. Data Loading**\n\n568K+ reviews")
    with pipeline_cols[1]:
        st.info("**2. Cleaning**\n\nDuplicates & nulls")
    with pipeline_cols[2]:
        st.info("**3. Feature Engineering**\n\nDates, sentiment")
    with pipeline_cols[3]:
        st.info("**4. Analysis**\n\nPatterns & trends")
    with pipeline_cols[4]:
        st.info("**5. Visualization**\n\nInteractive charts")
    
    st.markdown("---")
    st.subheader("Download Data")
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Raw Dataset (CSV)",
            data=csv,
            file_name="amazon_reviews_raw.csv",
            mime="text/csv"
        )
    with col2:
        if 'Sentiment' in df_with_sentiment.columns:
            csv_sentiment = df_with_sentiment.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download with Sentiment Analysis (CSV)",
                data=csv_sentiment,
                file_name="amazon_reviews_sentiment.csv",
                mime="text/csv"
            )

elif page == "Data Exploration":
    st.title("Data Exploration")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📈 Distributions", "🔗 Correlations"])
    with tab1:
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Numerical Features")
            numeric_stats = df[['Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator']].describe()
            st.dataframe(numeric_stats, use_container_width=True)
        with col2:
            st.markdown("##### Categorical Features")
            cat_stats = pd.DataFrame({
                'Unique Products': [df['ProductId'].nunique()],
                'Unique Users': [df['UserId'].nunique()],
                'Unique Profiles': [df['ProfileName'].nunique()],
                'Total Reviews': [len(df)]
            }).T
            cat_stats.columns = ['Count']
            st.dataframe(cat_stats, use_container_width=True)
        st.markdown("##### Top 10 Most Reviewed Products")
        top_products = df['ProductId'].value_counts().head(10).reset_index()
        top_products.columns = ['Product ID', 'Number of Reviews']
        st.dataframe(top_products, use_container_width=True)
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Helpfulness Numerator Distribution")
            fig = px.histogram(df[df['HelpfulnessNumerator'] <= 50], x='HelpfulnessNumerator', nbins=50,
                               color_discrete_sequence=['#636EFA'])
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Review Length Distribution")
            df['TextLength'] = df['Text'].str.len()
            fig = px.histogram(df[df['TextLength'] <= 2000], x='TextLength', nbins=50,
                               color_discrete_sequence=['#EF553B'])
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Score vs Helpfulness")
        help_ratio = df.copy()
        help_ratio['HelpfulnessRatio'] = help_ratio.apply(
            lambda x: x['HelpfulnessNumerator'] / x['HelpfulnessDenominator'] 
            if x['HelpfulnessDenominator'] > 0 else 0, axis=1
        )
        help_ratio = help_ratio[help_ratio['HelpfulnessDenominator'] >= 5]
        avg_help = help_ratio.groupby('Score')['HelpfulnessRatio'].mean().reset_index()
        fig = px.bar(avg_help, x='Score', y='HelpfulnessRatio',
                     labels={'HelpfulnessRatio': 'Average Helpfulness Ratio'},
                     color='HelpfulnessRatio', color_continuous_scale='Teal')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader("Feature Correlations")
        corr_data = df[['Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator']].corr()
        fig = px.imshow(corr_data, text_auto='.2f', color_continuous_scale='RdBu_r', aspect='auto')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Sentiment Analysis":
    st.title("Sentiment Analysis")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["😊 Sentiment Overview", "☁️ Word Clouds", "📊 Polarity Analysis"])
    with tab1:
        sentiment_counts = df_with_sentiment['Sentiment'].value_counts()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("😊 Positive", f"{sentiment_counts.get('Positive', 0):,}")
        with col2:
            st.metric("😐 Neutral", f"{sentiment_counts.get('Neutral', 0):,}")
        with col3:
            st.metric("😞 Negative", f"{sentiment_counts.get('Negative', 0):,}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                         color=sentiment_counts.index,
                         color_discrete_map={'Positive': '#00CC96','Neutral': '#FFA15A','Negative': '#EF553B'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Sentiment by Rating Score")
            sentiment_by_score = df_with_sentiment.groupby(['Score', 'Sentiment']).size().reset_index(name='Count')
            fig = px.bar(sentiment_by_score, x='Score', y='Count', color='Sentiment', barmode='group',
                         color_discrete_map={'Positive': '#00CC96','Neutral': '#FFA15A','Negative': '#EF553B'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Word Clouds by Sentiment")
        sentiment_option = st.selectbox("Select Sentiment", ['All Reviews', 'Positive', 'Neutral', 'Negative'])
        with st.spinner("Generating word cloud..."):
            if sentiment_option == 'All Reviews':
                wc = generate_wordcloud_data(df_with_sentiment)
            else:
                wc = generate_wordcloud_data(df_with_sentiment, sentiment_option)
            if wc:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.warning("Not enough data to generate word cloud")
    with tab3:
        st.subheader("Polarity and Subjectivity Analysis")
        df_sentiment_filtered = df_with_sentiment[
            (df_with_sentiment['Polarity'] != 0) | (df_with_sentiment['Subjectivity'] != 0)
        ].head(5000)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df_sentiment_filtered, x='Polarity', nbins=50,
                               color_discrete_sequence=['#AB63FA'],
                               labels={'Polarity': 'Sentiment Polarity'})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df_sentiment_filtered, x='Subjectivity', nbins=50,
                               color_discrete_sequence=['#19D3F3'],
                               labels={'Subjectivity': 'Sentiment Subjectivity'})
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Polarity vs Subjectivity Scatter")
        fig = px.scatter(df_sentiment_filtered, x='Polarity', y='Subjectivity',
                         color='Score', opacity=0.5, color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Trends & Patterns":
    st.title("Trends & Patterns")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📅 Time Series", "🔥 Product Insights", "👥 User Behavior"])
    with tab1:
        st.subheader("Review Trends Over Time")
        monthly_reviews = df.groupby('YearMonth').agg({'Score': 'mean', 'Id': 'count'}).reset_index()
        monthly_reviews.columns = ['YearMonth', 'AvgScore', 'Count']
        monthly_reviews = monthly_reviews.sort_values('YearMonth')
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=monthly_reviews['YearMonth'], y=monthly_reviews['Count'],
                                 name="Review Count", line=dict(color='#636EFA')), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly_reviews['YearMonth'], y=monthly_reviews['AvgScore'],
                                 name="Average Score", line=dict(color='#EF553B')), secondary_y=True)
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Number of Reviews", secondary_y=False)
        fig.update_yaxes(title_text="Average Score", secondary_y=True)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Score Distribution by Year")
        score_by_year = df.groupby(['Year', 'Score']).size().reset_index(name='Count')
        fig = px.bar(score_by_year, x='Year', y='Count', color='Score', color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Top Rated Products")
        product_stats = df.groupby('ProductId').agg({'Score': ['mean', 'count'], 'HelpfulnessNumerator': 'sum'}).reset_index()
        product_stats.columns = ['ProductId', 'AvgScore', 'ReviewCount', 'TotalHelpfulness']
        product_stats = product_stats[product_stats['ReviewCount'] >= 10]
        top_products = product_stats.nlargest(20, 'AvgScore')
        fig = px.bar(top_products, x='ProductId', y='AvgScore', color='ReviewCount',
                     hover_data=['ReviewCount', 'TotalHelpfulness'],
                     labels={'AvgScore': 'Average Score'}, color_continuous_scale='Greens')
        fig.update_layout(height=400, showlegend=True)
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Product Performance Matrix")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Products with 5-star average", f"{len(product_stats[product_stats['AvgScore'] == 5])}")
        with col2:
            st.metric("Products with 10+ reviews", f"{len(product_stats[product_stats['ReviewCount'] >= 10])}")
    with tab3:
        st.subheader("User Activity Analysis")
        user_stats = df.groupby('UserId').agg({'Score': ['mean', 'count']}).reset_index()
        user_stats.columns = ['UserId', 'AvgScore', 'ReviewCount']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Unique Users", f"{df['UserId'].nunique():,}")
            st.metric("Average Reviews per User", f"{user_stats['ReviewCount'].mean():.2f}")
        with col2:
            st.metric("Most Active User Reviews", f"{user_stats['ReviewCount'].max()}")
            st.metric("Median Reviews per User", f"{user_stats['ReviewCount'].median():.0f}")
        st.subheader("Review Count Distribution")
        review_dist = user_stats['ReviewCount'].value_counts().head(20).sort_index()
        fig = px.bar(x=review_dist.index, y=review_dist.values,
                     labels={'x': 'Number of Reviews', 'y': 'Number of Users'},
                     color=review_dist.values, color_continuous_scale='Purp')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Sample Reviews":
    st.title("Advanced Reviews Browser")
    st.markdown("---")
    st.subheader("Search and Filter")
    search_term = st.text_input("🔎 Search in review text (case-insensitive)", "")
    col1, col2, col3 = st.columns(3)
    with col1:
        score_filter = st.multiselect("Filter by Score",
                                      options=sorted(df['Score'].unique()),
                                      default=sorted(df['Score'].unique()))
    with col2:
        if 'Sentiment' in df_with_sentiment.columns:
            sentiment_filter = st.multiselect("Filter by Sentiment",
                                              options=['Positive', 'Neutral', 'Negative'],
                                              default=['Positive', 'Neutral', 'Negative'])
        else:
            sentiment_filter = []
    with col3:
        min_helpfulness = st.slider("Minimum Helpfulness Votes", min_value=0, max_value=50, value=0)
    filtered_df = df_with_sentiment[
        (df_with_sentiment['Score'].isin(score_filter)) &
        (df_with_sentiment['HelpfulnessNumerator'] >= min_helpfulness)
    ]
    if sentiment_filter and 'Sentiment' in df_with_sentiment.columns:
        filtered_df = filtered_df[filtered_df['Sentiment'].isin(sentiment_filter)]
    if search_term:
        filtered_df = filtered_df[df['Text'].str.contains(search_term, case=False, na=False)]
    st.markdown(f"**Found {len(filtered_df):,} matching reviews**")
    col1, col2 = st.columns(2)
    with col1:
        num_display = st.slider("Number of reviews to display", min_value=5, max_value=50, value=10)
    with col2:
        if len(filtered_df) > 0:
            csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download Filtered Reviews ({len(filtered_df):,} rows)",
                data=csv_filtered,
                file_name="filtered_reviews.csv",
                mime="text/csv"
            )
    if len(filtered_df) > 0:
        sample_reviews = filtered_df.sample(n=min(num_display, len(filtered_df)))
        for idx, row in sample_reviews.iterrows():
            with st.expander(f"⭐ {row['Score']} - {row['Summary'][:100]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Review:** {row['Text'][:500]}...")
                    st.markdown(f"**Product ID:** {row['ProductId']}")
                    st.markdown(f"**Date:** {row['Date'].strftime('%Y-%m-%d')}")
                with col2:
                    if 'Sentiment' in row and pd.notna(row['Sentiment']):
                        sentiment_color = {'Positive': '🟢','Neutral': '🟡','Negative': '🔴'}
                        st.markdown(f"**Sentiment:** {sentiment_color.get(row['Sentiment'], '')} {row['Sentiment']}")
                        if pd.notna(row['Polarity']):
                            st.markdown(f"**Polarity:** {row['Polarity']:.3f}")
                    st.markdown(f"**Helpful:** {row['HelpfulnessNumerator']}/{row['HelpfulnessDenominator']}")
    else:
        st.warning("No reviews match your filters. Try adjusting the search criteria.")

elif page == "ML Performance":
    st.title("Machine Learning Model Performance")
    st.markdown("---")
    st.markdown("""
    This section demonstrates the classification performance of sentiment analysis based on review scores.
    We treat scores 1-2 as Negative, 3 as Neutral, and 4-5 as Positive.
    """)
    df_ml = df_with_sentiment[df_with_sentiment['Sentiment'].isin(['Positive', 'Neutral', 'Negative'])].copy()
    df_ml['ScoreCategory'] = df_ml['Score'].apply(lambda x: 'Negative' if x <= 2 else ('Neutral' if x == 3 else 'Positive'))
    df_ml_sample = df_ml.head(5000)
    tab1, tab2, tab3 = st.tabs([" Classification Metrics", " Confusion Matrix", "Performance Analysis"])
    with tab1:
        st.subheader("Classification Report")
        y_true = df_ml_sample['ScoreCategory']
        y_pred = df_ml_sample['Sentiment']
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            accuracy = accuracy_score(y_true, y_pred)
            st.metric("Overall Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Macro Avg F1-Score", f"{report_dict['macro avg']['f1-score']:.3f}")
        with col3:
            st.metric("Weighted Avg F1-Score", f"{report_dict['weighted avg']['f1-score']:.3f}")
    with tab2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=['Negative', 'Neutral', 'Positive'])
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Negative', 'Neutral', 'Positive'], y=['Negative', 'Neutral', 'Positive'],
                        text_auto=True, color_continuous_scale='Blues', aspect='auto')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Interpretation:**
        - Diagonal values show correct predictions
        - Off-diagonal values show misclassifications
        - Darker colors indicate higher counts
        """)
    with tab3:
        st.subheader("Performance by Class")
        class_metrics = pd.DataFrame({
            'Class': ['Negative', 'Neutral', 'Positive'],
            'Precision': [report_dict['Negative']['precision'], report_dict['Neutral']['precision'], report_dict['Positive']['precision']],
            'Recall': [report_dict['Negative']['recall'], report_dict['Neutral']['recall'], report_dict['Positive']['recall']],
            'F1-Score': [report_dict['Negative']['f1-score'], report_dict['Neutral']['f1-score'], report_dict['Positive']['f1-score']]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Precision', x=class_metrics['Class'], y=class_metrics['Precision']))
        fig.add_trace(go.Bar(name='Recall', x=class_metrics['Class'], y=class_metrics['Recall']))
        fig.add_trace(go.Bar(name='F1-Score', x=class_metrics['Class'], y=class_metrics['F1-Score']))
        fig.update_layout(barmode='group', height=400, yaxis_title="Score", xaxis_title="Sentiment Class")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Data Processing":
    st.title("Data Preprocessing & SMOTE")
    st.markdown("---")
    st.markdown("""
    This section demonstrates the data preprocessing pipeline and class balancing using SMOTE 
    (Synthetic Minority Over-sampling Technique).
    """)
    tab1, tab2, tab3 = st.tabs(["Class Distribution", "SMOTE Balancing", "Before & After"])
    with tab1:
        st.subheader("Original Class Distribution")
        df_ml = df_with_sentiment[df_with_sentiment['Sentiment'].isin(['Positive', 'Neutral', 'Negative'])].copy()
        class_dist = df_ml['Sentiment'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=class_dist.values, names=class_dist.index, color=class_dist.index,
                         color_discrete_map={'Positive': '#00CC96','Neutral': '#FFA15A','Negative': '#EF553B'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=class_dist.index, y=class_dist.values, labels={'x': 'Sentiment', 'y': 'Count'},
                         color=class_dist.values, color_continuous_scale='Viridis')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({
            'Class': class_dist.index,
            'Count': class_dist.values,
            'Percentage': (class_dist.values / class_dist.values.sum() * 100).round(2)
        }), use_container_width=True)
    with tab2:
        st.subheader("SMOTE - Balanced Class Distribution")
        st.markdown("SMOTE creates synthetic samples for minority classes to balance the dataset.")
        df_smote_sample = df_ml.sample(n=min(5000, len(df_ml)), random_state=42)
        df_smote_sample = df_smote_sample[['Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Polarity', 'Subjectivity', 'Sentiment']].dropna()
        X = df_smote_sample[['Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Polarity', 'Subjectivity']]
        y = df_smote_sample['Sentiment']
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        y_resampled_labels = le.inverse_transform(y_resampled)
        resampled_dist = pd.Series(y_resampled_labels).value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=resampled_dist.values, names=resampled_dist.index, color=resampled_dist.index,
                         color_discrete_map={'Positive': '#00CC96','Neutral': '#FFA15A','Negative': '#EF553B'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=resampled_dist.index, y=resampled_dist.values, labels={'x': 'Sentiment', 'y': 'Count'},
                         color=resampled_dist.values, color_continuous_scale='Greens')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.subheader("Before and After Comparison")
        comparison_df = pd.DataFrame({
            'Class': ['Negative', 'Neutral', 'Positive'],
            'Before SMOTE': [class_dist.get('Negative', 0), class_dist.get('Neutral', 0), class_dist.get('Positive', 0)],
            'After SMOTE': [resampled_dist.get('Negative', 0), resampled_dist.get('Neutral', 0), resampled_dist.get('Positive', 0)]
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Before SMOTE', x=comparison_df['Class'], y=comparison_df['Before SMOTE']))
        fig.add_trace(go.Bar(name='After SMOTE', x=comparison_df['Class'], y=comparison_df['After SMOTE']))
        fig.update_layout(barmode='group', height=400, yaxis_title="Sample Count", xaxis_title="Sentiment Class")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comparison_df, use_container_width=True)

elif page == "Sentiment Predictor":
    st.title("Real-Time Sentiment Predictor")
    st.markdown("---")
    st.markdown("### Try the Sentiment Analyzer! Enter any review text below to get instant sentiment analysis using TextBlob.")
    user_input = st.text_area("Enter your review text:", height=150, placeholder="Type or paste a product review here...")
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if user_input.strip():
            sentiment, polarity, subjectivity = predict_sentiment(user_input)
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_colors = {'Positive': '🟢','Neutral': '🟡','Negative': '🔴'}
                st.metric("Sentiment", f"{sentiment_colors[sentiment]} {sentiment}")
            with col2:
                st.metric("Polarity", f"{polarity:.3f}")
                st.caption("Range: -1 (negative) to +1 (positive)")
            with col3:
                st.metric("Subjectivity", f"{subjectivity:.3f}")
                st.caption("Range: 0 (objective) to 1 (subjective)")
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=polarity, domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Polarity Score"},
                    gauge={'axis': {'range': [-1, 1]}, 'bar': {'color': "darkblue"},
                           'steps': [{'range': [-1, -0.1], 'color': "#EF553B"},
                                     {'range': [-0.1, 0.1], 'color': "#FFA15A"},
                                     {'range': [0.1, 1], 'color': "#00CC96"}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': polarity}}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=subjectivity, domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Subjectivity Score"},
                    gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "purple"},
                           'steps': [{'range': [0, 0.5], 'color': "lightblue"},
                                     {'range': [0.5, 1], 'color': "lightcoral"}]}
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            st.success("✅ Analysis complete!")
        else:
            st.warning("Please enter some text to analyze.")
    st.markdown("---")
    st.subheader("📝 Example Reviews")
    examples = [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "It's okay, nothing special. Does what it's supposed to do.",
        "Terrible quality. Broke after one use. Very disappointed.",
        "Great value for money. Would definitely recommend to friends.",
    ]
    for i, example in enumerate(examples):
        if st.button(f"Try Example {i+1}", key=f"ex_{i}"):
            sentiment, polarity, subjectivity = predict_sentiment(example)
            st.info(f"**Text:** {example}")
            st.write(f"**Result:** {sentiment} (Polarity: {polarity:.3f}, Subjectivity: {subjectivity:.3f})")

elif page == "Product Analysis":
    st.title("Detailed Product Analysis")
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🏆 Top Products", "📉 Bottom Products", "🔍 Product Details"])
    product_stats = df.groupby('ProductId').agg({
        'Score': ['mean', 'count', 'std'],
        'HelpfulnessNumerator': 'sum',
        'Text': lambda x: x.str.len().mean()
    }).reset_index()
    product_stats.columns = ['ProductId', 'AvgScore', 'ReviewCount', 'ScoreStd', 'TotalHelpfulness', 'AvgReviewLength']
    product_stats = product_stats[product_stats['ReviewCount'] >= 5]
    with tab1:
        st.subheader("🏆 Top Performing Products")
        min_reviews_top = st.slider("Minimum reviews for top products", 5, 100, 10, key="top_slider")
        top_products_filtered = product_stats[product_stats['ReviewCount'] >= min_reviews_top]
        top_products = top_products_filtered.nlargest(20, 'AvgScore')
        fig = px.scatter(top_products, x='ReviewCount', y='AvgScore', size='TotalHelpfulness',
                         hover_data=['ProductId', 'ScoreStd'], color='AvgScore',
                         color_continuous_scale='Greens',
                         labels={'AvgScore': 'Average Score', 'ReviewCount': 'Number of Reviews'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_products.sort_values('AvgScore', ascending=False), use_container_width=True)
        csv = top_products.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Top Products (CSV)", data=csv, file_name="top_products.csv", mime="text/csv")
    with tab2:
        st.subheader("📉 Bottom Performing Products")
        min_reviews_bottom = st.slider("Minimum reviews for bottom products", 5, 100, 10, key="bottom_slider")
        bottom_products_filtered = product_stats[product_stats['ReviewCount'] >= min_reviews_bottom]
        bottom_products = bottom_products_filtered.nsmallest(20, 'AvgScore')
        fig = px.scatter(bottom_products, x='ReviewCount', y='AvgScore', size='TotalHelpfulness',
                         hover_data=['ProductId', 'ScoreStd'], color='AvgScore',
                         color_continuous_scale='Reds',
                         labels={'AvgScore': 'Average Score', 'ReviewCount': 'Number of Reviews'})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(bottom_products.sort_values('AvgScore'), use_container_width=True)
        csv = bottom_products.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Bottom Products (CSV)", data=csv, file_name="bottom_products.csv", mime="text/csv")
    with tab3:
        st.subheader("🔍 Individual Product Deep Dive")
        product_id = st.selectbox("Select Product ID",
                                  options=product_stats.nlargest(100, 'ReviewCount')['ProductId'].tolist())
        if product_id:
            product_reviews = df[df['ProductId'] == product_id]
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Average Score", f"{product_reviews['Score'].mean():.2f}")
            with col2: st.metric("Total Reviews", f"{len(product_reviews)}")
            with col3: st.metric("Score Std Dev", f"{product_reviews['Score'].std():.2f}")
            with col4: st.metric("Total Helpfulness", f"{product_reviews['HelpfulnessNumerator'].sum()}")
            col1, col2 = st.columns(2)
            with col1:
                score_dist = product_reviews['Score'].value_counts().sort_index()
                fig = px.bar(x=score_dist.index, y=score_dist.values, labels={'x': 'Score', 'y': 'Count'},
                             title="Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                reviews_over_time = product_reviews.groupby('YearMonth').size().reset_index(name='Count')
                fig = px.line(reviews_over_time, x='YearMonth', y='Count', title="Reviews Over Time", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            st.subheader("Sample Reviews for this Product")
            sample_product_reviews = product_reviews.sample(n=min(5, len(product_reviews)))
            for _, row in sample_product_reviews.iterrows():
                with st.expander(f"⭐ {row['Score']} - {row['Summary'][:80]}..."):
                    st.write(f"**Date:** {row['Date'].strftime('%Y-%m-%d')}")
                    st.write(f"**Review:** {row['Text']}")
                    st.write(f"**Helpful:** {row['HelpfulnessNumerator']}/{row['HelpfulnessDenominator']}")

elif page == "Period Comparison":
    st.title("Time Period Comparison")
    st.markdown("---")
    st.markdown("Compare review patterns, sentiment, and ratings across different time periods.")
    years = sorted(df['Year'].unique())
    col1, col2 = st.columns(2)
    with col1:
        period1 = st.multiselect("Select Period 1 (Years)", years, default=[years[0]] if len(years) > 0 else [])
    with col2:
        period2 = st.multiselect("Select Period 2 (Years)", years, default=[years[-1]] if len(years) > 0 else [])
    if period1 and period2:
        df_period1 = df[df['Year'].isin(period1)]
        df_period2 = df[df['Year'].isin(period2)]
        tab1, tab2, tab3 = st.tabs(["📊 Overview Comparison", "💭 Sentiment Comparison", "📈 Detailed Metrics"])
        with tab1:
            st.subheader("Key Metrics Comparison")
            metrics_comparison = pd.DataFrame({
                'Metric': ['Total Reviews', 'Average Score', 'Unique Products', 'Avg Helpfulness'],
                'Period 1': [len(df_period1), df_period1['Score'].mean(), df_period1['ProductId'].nunique(), df_period1['HelpfulnessNumerator'].mean()],
                'Period 2': [len(df_period2), df_period2['Score'].mean(), df_period2['ProductId'].nunique(), df_period2['HelpfulnessNumerator'].mean()]
            })
            st.dataframe(metrics_comparison.style.format({'Period 1': '{:.2f}', 'Period 2': '{:.2f}'}), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                score_dist1 = df_period1['Score'].value_counts().sort_index()
                fig = px.bar(x=score_dist1.index, y=score_dist1.values, labels={'x': 'Score', 'y': 'Count'},
                             title=f"Period 1: Score Distribution ({', '.join(map(str, period1))})")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                score_dist2 = df_period2['Score'].value_counts().sort_index()
                fig = px.bar(x=score_dist2.index, y=score_dist2.values, labels={'x': 'Score', 'y': 'Count'},
                             title=f"Period 2: Score Distribution ({', '.join(map(str, period2))})")
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("Sentiment Comparison")
            df_p1_sent = df_with_sentiment[df_with_sentiment['Year'].isin(period1)]
            df_p2_sent = df_with_sentiment[df_with_sentiment['Year'].isin(period2)]
            sent1 = df_p1_sent['Sentiment'].value_counts()
            sent2 = df_p2_sent['Sentiment'].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(values=sent1.values, names=sent1.index,
                             title=f"Period 1 Sentiment ({', '.join(map(str, period1))})",
                             color=sent1.index,
                             color_discrete_map={'Positive': '#00CC96','Neutral': '#FFA15A','Negative': '#EF553B'})
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(values=sent2.values, names=sent2.index,
                             title=f"Period 2 Sentiment ({', '.join(map(str, period2))})",
                             color=sent2.index,
                             color_discrete_map={'Positive': '#00CC96','Neutral': '#FFA15A','Negative': '#EF553B'})
                st.plotly_chart(fig, use_container_width=True)
            sentiment_comparison = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Period 1 %': [
                    sent1.get('Positive', 0) / len(df_p1_sent) * 100 if len(df_p1_sent) else 0,
                    sent1.get('Neutral', 0) / len(df_p1_sent) * 100 if len(df_p1_sent) else 0,
                    sent1.get('Negative', 0) / len(df_p1_sent) * 100 if len(df_p1_sent) else 0
                ],
                'Period 2 %': [
                    sent2.get('Positive', 0) / len(df_p2_sent) * 100 if len(df_p2_sent) else 0,
                    sent2.get('Neutral', 0) / len(df_p2_sent) * 100 if len(df_p2_sent) else 0,
                    sent2.get('Negative', 0) / len(df_p2_sent) * 100 if len(df_p2_sent) else 0
                ]
            })
            st.dataframe(sentiment_comparison.style.format({'Period 1 %': '{:.2f}', 'Period 2 %': '{:.2f}'}), use_container_width=True)
        with tab3:
            st.subheader("Detailed Statistical Comparison")
            comparison_df = pd.DataFrame({
                'Metric': ['Mean Score','Median Score','Std Dev Score','Mean Polarity','Mean Subjectivity','Mean Helpfulness Numerator','Mean Review Length (chars)'],
                'Period 1': [df_period1['Score'].mean(), df_period1['Score'].median(), df_period1['Score'].std(),
                             df_p1_sent['Polarity'].mean(), df_p1_sent['Subjectivity'].mean(),
                             df_period1['HelpfulnessNumerator'].mean(), df_period1['Text'].str.len().mean()],
                'Period 2': [df_period2['Score'].mean(), df_period2['Score'].median(), df_period2['Score'].std(),
                             df_p2_sent['Polarity'].mean(), df_p2_sent['Subjectivity'].mean(),
                             df_period2['HelpfulnessNumerator'].mean(), df_period2['Text'].str.len().mean()]
            })
            comparison_df['Difference'] = comparison_df['Period 2'] - comparison_df['Period 1']
            comparison_df['% Change'] = (comparison_df['Difference'] / comparison_df['Period 1'] * 100)
            st.dataframe(comparison_df.style.format({'Period 1': '{:.3f}','Period 2': '{:.3f}','Difference': '{:.3f}','% Change': '{:.2f}%'}), use_container_width=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(name=f'Period 1 ({", ".join(map(str, period1))})',
                                 x=comparison_df['Metric'][:5], y=comparison_df['Period 1'][:5]))
            fig.add_trace(go.Bar(name=f'Period 2 ({", ".join(map(str, period2))})',
                                 x=comparison_df['Metric'][:5], y=comparison_df['Period 2'][:5]))
            fig.update_layout(barmode='group', height=400, title="Key Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select years for both periods to see the comparison.")

# ---------- Final fallback (should never hit if labels match) ----------
else:
    st.warning("Please choose a page from the sidebar.")

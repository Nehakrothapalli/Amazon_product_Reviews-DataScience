# Amazon Food Reviews Sentiment Analysis

## Overview

This project is a comprehensive sentiment analysis application for Amazon food reviews built with Streamlit. It provides interactive data visualization, machine learning performance metrics, natural language processing capabilities, and advanced analytics tools to analyze customer sentiment from review data. The application loads review data from a CSV file, performs sentiment analysis using TextBlob, demonstrates ML model performance, and presents insights through interactive charts and visualizations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for rapid web application development
- **Navigation**: Multi-page application with 10 distinct sections
- **Visualization Libraries**: 
  - Plotly Express and Plotly Graph Objects for interactive charts
  - Seaborn and Matplotlib for statistical visualizations
  - WordCloud for text visualization
- **Layout**: Wide layout with expandable sidebar for controls and filters
- **Caching Strategy**: Uses `@st.cache_data` decorator to cache data loading and sentiment analysis operations, improving performance for repeated operations

### Application Pages
1. **Overview**: Dashboard with key metrics, score distribution, and data export
2. **Data Exploration**: Statistical analysis, distributions, and correlations
3. **Sentiment Analysis**: Sentiment distribution, word clouds, and polarity analysis
4. **Trends & Patterns**: Time series analysis, product insights, and user behavior
5. **Sample Reviews**: Advanced search and filtering with text search capability
6. **ML Performance**: Classification metrics, confusion matrix, and performance analysis
7. **Data Processing**: SMOTE demonstration and before/after class balancing
8. **Sentiment Predictor**: Real-time sentiment prediction for user-input text
9. **Product Analysis**: Top/bottom performing products with detailed deep dive
10. **Period Comparison**: Interactive time period comparison tool

### Data Processing Pipeline
- **Data Loading**: CSV-based data source with configurable sampling (default 50,000 records)
- **Data Cleaning**: 
  - Fills missing text fields with empty strings
  - Fills missing scores with median values
  - Removes duplicates based on ProductId, Time, and Text
- **Feature Engineering**:
  - Converts Unix timestamps to datetime objects
  - Extracts Year, Month, and YearMonth for temporal analysis
  - Generates sentiment polarity and subjectivity scores using TextBlob
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced datasets

### Sentiment Analysis Engine
- **NLP Library**: TextBlob for sentiment analysis
- **Metrics Calculated**:
  - Polarity: Measures positive/negative sentiment (-1 to +1)
  - Subjectivity: Measures opinion vs. fact (0 to 1)
  - Sentiment Classification: Positive (>0.1), Neutral (-0.1 to 0.1), Negative (<-0.1)
- **Real-Time Prediction**: Interactive tool for analyzing user-input text with visual gauges
- **Error Handling**: Try-except blocks to handle text processing failures gracefully

### Machine Learning Features
- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Visual representation of model performance
- **Performance Analysis**: Class-wise metrics comparison
- **SMOTE Balancing**: Visual demonstration of class distribution before and after balancing

### Data Export Capabilities
- **Raw Dataset Export**: Download original data as CSV
- **Sentiment-Enhanced Export**: Download data with sentiment analysis results
- **Filtered Reviews Export**: Download filtered search results
- **Product Analysis Export**: Download top/bottom performing products

### Data Model
- **Primary Data Source**: Reviews.csv located in attached_assets directory
- **Key Fields**:
  - ProductId: Product identifier
  - Score: Numerical rating (1-5)
  - Text: Full review text
  - Summary: Review summary
  - Time: Unix timestamp
  - UserId: User identifier
  - HelpfulnessNumerator: Number of helpful votes
  - HelpfulnessDenominator: Total votes
- **Derived Fields**:
  - Date, Year, Month, YearMonth: Temporal features
  - Polarity, Subjectivity, Sentiment: Sentiment analysis results
  - TextLength: Character count of review text
  - HelpfulnessRatio: Calculated helpfulness metric

## Features

### Core Analytics Features
- Interactive dashboard with key statistics (total reviews, average rating, time span, unique products)
- Score distribution visualization with interactive charts
- Time-series analysis of review trends
- Helpfulness analysis and correlation with ratings
- User behavior analysis and activity patterns
- Product performance metrics and rankings

### Sentiment Analysis Features
- Sentiment distribution with pie charts and bar graphs
- Word cloud generation for positive, neutral, and negative reviews
- Polarity and subjectivity analysis with histograms and scatter plots
- Sentiment by rating score correlation
- Real-time sentiment prediction tool with:
  - Text input for custom reviews
  - Polarity and subjectivity gauges
  - Visual indicators for sentiment classification
  - Example reviews for testing

### Advanced Features
- **Advanced Search**: Text-based search across all reviews with case-insensitive matching
- **Multi-Filter System**: 
  - Filter by score (1-5)
  - Filter by sentiment (Positive, Neutral, Negative)
  - Filter by minimum helpfulness votes
- **Product Deep Dive**:
  - Top 20 and bottom 20 products analysis
  - Product-specific metrics (average score, review count, helpfulness)
  - Individual product timeline and score distribution
  - Sample reviews for each product
- **Period Comparison**:
  - Year-to-year comparison of metrics
  - Sentiment distribution comparison
  - Statistical analysis of changes over time
  - Visual side-by-side comparisons

### Machine Learning Visualization
- Classification report with precision, recall, and F1-scores
- Confusion matrix heatmap
- Performance metrics by class (Positive, Neutral, Negative)
- SMOTE class balancing demonstration
- Before/after comparison of class distributions

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework (>=1.51.0)
- **pandas**: Data manipulation and analysis (>=2.3.3)
- **numpy**: Numerical computing (>=2.3.5)
- **plotly**: Interactive visualization (>=6.4.0)
- **seaborn**: Statistical data visualization (>=0.13.2)
- **matplotlib**: Plotting library (>=3.10.7)
- **textblob**: Natural language processing and sentiment analysis (>=0.19.0)
- **wordcloud**: Text visualization for word frequency (>=1.9.4)
- **scikit-learn**: Machine learning metrics and preprocessing (>=1.7.2)
- **imbalanced-learn**: SMOTE and class balancing (>=0.14.0)

### Data Dependencies
- **Reviews.csv**: Amazon food reviews dataset stored in `attached_assets/` directory
- **Dataset Size**: 568,454 reviews
- **Time Range**: 2000-2012
- Expected schema includes: Id, ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text

### Configuration
- **Page Config**: Custom page title, icon (🍔), wide layout, expanded sidebar
- **Warning Suppression**: Configured to ignore Python warnings for cleaner output
- **Random State**: Fixed at 42 for reproducible sampling
- **Default Sample Size**: 50,000 records for efficient processing
- **Sentiment Analysis Sample**: 10,000 records for TextBlob processing

## Recent Updates

### November 2025
- Added ML Performance page with classification metrics and confusion matrix
- Implemented Data Processing page with SMOTE demonstration
- Created Sentiment Predictor tool for real-time text analysis
- Built Product Analysis page with top/bottom product rankings
- Added Period Comparison tool for temporal analysis
- Implemented advanced text search across all reviews
- Added CSV export functionality on multiple pages
- Enhanced filtering capabilities with multi-select options
- Improved visualizations with Plotly interactive charts
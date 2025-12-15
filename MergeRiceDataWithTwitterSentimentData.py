import pandas as pd


# Load the dataset
df = pd.read_excel("processed_rice_tweets_with_sentiments2.xlsx")

# Ensure the Date column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Add a new column for the year-month (YYYY-MM) format
df['YearMonth'] = df['date'].dt.to_period('M').astype(str)

# Create separate columns for positive, neutral, and negative sentiments
df['positive_sentiment'] = df['sentiment'].apply(lambda x: 1 if x > 0 else 0)
df['neutral_sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 0 else 0)
df['negative_sentiment'] = df['sentiment'].apply(lambda x: 1 if x < 0 else 0)



# Aggregate the data by month
aggregated_df = df.groupby(df['date'].dt.to_period('M')).agg({
    'sentiment': 'mean',  # Average sentiment for each month
    'retweets': 'sum',    # Total retweets for each month
    'likes': 'sum',       # Total likes for each month
    'engagement': 'sum',  # Total engagement for each month
    'log_scaled_engagement': 'sum',  # Total log-scaled engagement for each month
    'weighted_sentiment': 'mean',  # Average weighted sentiment for each month
    'positive_sentiment': 'sum',  # Total positive sentiment count for each month
    'neutral_sentiment': 'sum',   # Total neutral sentiment count for each month
    'negative_sentiment': 'sum'   # Total negative sentiment count for each month
}).reset_index()


# Rename columns for easier handling
aggregated_df = aggregated_df.rename(columns={
    'sentiment': 'average_sentiment',
    'retweets': 'sum_of_retweets',
    'likes': 'sum_of_likes',
    'engagement': 'sum_of_engagements',
    'log_scaled_engagement': 'sum_of_log_scaled_engagement',
    'weighted_sentiment': 'average_weighted_sentiment',
    'positive_sentiment': 'sum_of_positive_sentiment',
    'neutral_sentiment': 'sum_of_neutral_sentiment',
    'negative_sentiment': 'sum_of_negative_sentiment'

})



# Ensure the date column is in datetime format
#aggregated_df['date'] = pd.to_datetime(aggregated_df['date'])


# Convert period to datetime for merging
aggregated_df['date'] = aggregated_df['date'].dt.to_timestamp()

# Load climate data
climate_data = pd.read_csv('processed_climate_rice_new.csv')

# Ensure the date column is in datetime format
climate_data['Date'] = pd.to_datetime(climate_data['Date'])

# Merge the datasets on the date column
merged_data = pd.merge(climate_data, aggregated_df, how='left', left_on='Date', right_on='date')

# Handle missing months in sentiment data by filling with zeros
merged_data['average_sentiment'].fillna(0, inplace=True)
merged_data['sum_of_retweets'].fillna(0, inplace=True)
merged_data['sum_of_likes'].fillna(0, inplace=True)
merged_data['sum_of_engagements'].fillna(0, inplace=True)
merged_data['sum_of_log_scaled_engagement'].fillna(0, inplace=True)
merged_data['average_weighted_sentiment'].fillna(0, inplace=True)
merged_data['sum_of_negative_sentiment'].fillna(0, inplace=True)
merged_data['sum_of_neutral_sentiment'].fillna(0, inplace=True)
merged_data['sum_of_positive_sentiment'].fillna(0, inplace=True)


# Drop the redundant 'date' column from the merged dataset
merged_data.drop(columns=['date'], inplace=True)

# Save the merged dataset
merged_data.to_csv('merged_climate_sentiment_rice1.csv', index=False)

print("Merged dataset saved as 'merged_climate_sentiment_rice2.csv'")

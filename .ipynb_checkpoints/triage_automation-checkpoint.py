#!/usr/bin/env python3
"""
IT Support Ticket Triage Automation for Digital Banking Platform

This script automates the initial triage (L1) of IT support tickets by:
1. Processing ticket data from a CSV file
2. Categorizing issues using NLP
3. Assessing criticality
4. Grouping by root cause
5. Escalating tickets to L2/L3 as needed
6. Generating customer feedback
7. Visualizing key metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import re
import random
import os
from datetime import datetime

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Constants
INPUT_FILE = "support_tickets.csv"
OUTPUT_DIR = "output"
L2_L3_TICKETS_FILE = f"{OUTPUT_DIR}/tickets_l2_l3.txt"
CUSTOMER_FEEDBACK_FILE = f"{OUTPUT_DIR}/customer_feedback.txt"
VISUALIZATIONS_DIR = f"{OUTPUT_DIR}/visualizations"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Define issue categories and keywords
ISSUE_CATEGORIES = {
    "Login Failure": ["login", "password", "authentication", "credentials", "biometric", "fingerprint", "face", "recognition", "2fa", "two-factor", "verification", "code", "otp"],
    "Transaction Error": ["transaction", "payment", "transfer", "deposit", "withdraw", "money", "fund", "debit", "credit", "duplicate", "processing", "failed", "pending"],
    "Account Access": ["access", "locked", "suspended", "reset", "timeout", "session", "expired", "blocked", "restricted", "view", "balance"],
    "App Performance": ["slow", "crash", "freeze", "loading", "performance", "lag", "hang", "unresponsive", "error", "bug", "glitch"],
    "Feature Request": ["feature", "request", "add", "implement", "suggestion", "improve", "enhancement", "functionality", "option", "setting", "preference"],
    "Data Discrepancy": ["incorrect", "wrong", "mismatch", "discrepancy", "difference", "inaccurate", "missing", "data", "information", "balance", "statement"],
    "Security Concern": ["security", "fraud", "suspicious", "unauthorized", "phishing", "scam", "hack", "breach", "compromise", "alert", "warning"]
}

# Root causes mapping
ROOT_CAUSES = {
    "System Error": ["Login Failure", "App Performance", "Transaction Error"],
    "User Error": ["Login Failure", "Account Access"],
    "Data Issue": ["Data Discrepancy", "Transaction Error"],
    "Security Incident": ["Security Concern"],
    "Enhancement Request": ["Feature Request"],
    "Integration Problem": ["Transaction Error", "Data Discrepancy"]
}

# L2/L3 response templates
L2_L3_RESPONSES = {
    "Login Failure": [
        "Reset user credentials and send new temporary password",
        "Clear browser cache and cookies on user device",
        "Reset biometric settings and re-enroll user's biometric data",
        "Escalate to security team for potential account compromise",
        "Update authentication service and restart"
    ],
    "Transaction Error": [
        "Verify transaction in backend system and reconcile discrepancy",
        "Escalate to payment processing team for investigation",
        "Initiate transaction reversal and notify customer",
        "Check for duplicate transaction IDs in database",
        "Verify integration with external payment gateway"
    ],
    "Account Access": [
        "Manually unlock user account and reset security questions",
        "Verify identity through secondary channels and restore access",
        "Increase session timeout parameters for user account",
        "Reset account access permissions in IAM system",
        "Escalate to fraud prevention team for verification"
    ],
    "App Performance": [
        "Clear app cache and reinstall latest version",
        "Escalate to dev team for memory leak investigation",
        "Check for conflicting background processes on user device",
        "Optimize database queries for user's account",
        "Deploy hotfix for identified performance bottleneck"
    ],
    "Feature Request": [
        "Add to product backlog for next sprint planning",
        "Conduct user research to validate feature request",
        "Develop prototype for requested feature",
        "Schedule feature implementation in upcoming release",
        "Evaluate technical feasibility with development team"
    ],
    "Data Discrepancy": [
        "Run data reconciliation process for affected accounts",
        "Verify transaction logs against core banking system",
        "Escalate to data integrity team for investigation",
        "Restore data from last known good backup",
        "Implement data validation checks for affected module"
    ],
    "Security Concern": [
        "Escalate to security operations center for immediate investigation",
        "Enable additional security monitoring for affected account",
        "Reset all user access credentials and security questions",
        "Block suspicious IP addresses and devices",
        "Initiate security incident response protocol"
    ]
}

# Customer feedback templates
CUSTOMER_FEEDBACK = {
    "Low": [
        "Thank you for reporting this issue. We've logged your ticket and will address it in our regular maintenance cycle. You should see a resolution within 3-5 business days.",
        "We appreciate your feedback. Your request has been added to our backlog and will be considered for future updates.",
        "Thanks for bringing this to our attention. This is a known issue that we're working on. We expect to have a fix in our next release."
    ],
    "Medium": [
        "We're actively working on your issue and expect to have it resolved within 24-48 hours. We'll notify you once it's fixed.",
        "Your ticket has been assigned to our technical team. They'll be in touch with you shortly with more information or a solution.",
        "We understand this issue is affecting your banking experience. Our team is investigating and will provide an update by end of day."
    ],
    "High": [
        "We're treating this as a high priority issue. A specialist has been assigned and is working on it immediately. We'll contact you within the next 2 hours with an update.",
        "This issue has been escalated to our senior technical team. They're working on it urgently and will provide a resolution as soon as possible.",
        "We understand the critical nature of this issue. Our emergency response team has been notified and is taking immediate action. We'll call you shortly with more information."
    ]
}

def preprocess_text(text):
    """Preprocess text for NLP analysis."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def train_text_classifier(df):
    """Train a text classifier on labeled data."""
    print("Training text classifier for ticket categorization...")
    
    # If we have 'true_category' from mock data, use it for training
    if 'true_category' in df.columns:
        X = df['preprocessed_description']
        y = df['true_category']
        
        # Split into training and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create a pipeline with TF-IDF and Naive Bayes
        text_clf = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000)),
            ('clf', MultinomialNB())
        ])
        
        # Train the classifier
        text_clf.fit(X_train, y_train)
        
        # Evaluate accuracy
        accuracy = text_clf.score(X_test, y_test)
        print(f"Classifier accuracy: {accuracy:.2f}")
        
        return text_clf
    else:
        print("No labeled data available for training. Using keyword-based categorization.")
        return None

def categorize_issue(description, classifier=None):
    """Categorize the issue based on keywords or using a trained classifier."""
    # If we have a trained classifier, use it
    if classifier is not None:
        # Preprocess the description
        preprocessed = preprocess_text(description)
        # Predict the category
        return classifier.predict([preprocessed])[0]
    
    # Otherwise, fall back to keyword-based categorization
    description_lower = description.lower()
    
    # Count matches for each category
    category_scores = {}
    for category, keywords in ISSUE_CATEGORIES.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        category_scores[category] = score
    
    # Return the category with the highest score, or "Other" if no matches
    max_score = max(category_scores.values())
    if max_score == 0:
        return "Other"
    
    # If there's a tie, return the first category with the max score
    for category, score in category_scores.items():
        if score == max_score:
            return category

def determine_root_cause(category):
    """Map issue category to root cause."""
    for cause, categories in ROOT_CAUSES.items():
        if category in categories:
            return cause
    return "Unknown"

def assess_criticality(description):
    """Assess the criticality of the issue based on keywords and context."""
    description_lower = description.lower()
    
    # High criticality keywords
    high_keywords = [
        "urgent", "immediately", "fraud", "stolen", "unauthorized", 
        "security breach", "hacked", "emergency", "critical", "locked out",
        "money missing", "wrong amount", "failed", "error", "lost money"
    ]
    
    # Medium criticality keywords
    medium_keywords = [
        "not working", "issue", "problem", "slow", "delay", "bug", 
        "glitch", "incorrect", "difficulty", "trouble", "inconsistent"
    ]
    
    # Check for high criticality keywords
    for keyword in high_keywords:
        if keyword in description_lower:
            return "High"
    
    # Check for medium criticality keywords
    for keyword in medium_keywords:
        if keyword in description_lower:
            return "Medium"
    
    # Default to low criticality
    return "Low"

def generate_l2_l3_response(category):
    """Generate a mock L2/L3 response based on the issue category."""
    if category in L2_L3_RESPONSES:
        return random.choice(L2_L3_RESPONSES[category])
    return "Escalated to appropriate technical team for further investigation."

def generate_customer_feedback(criticality):
    """Generate customer feedback based on criticality."""
    return random.choice(CUSTOMER_FEEDBACK[criticality])

def cluster_similar_issues(df):
    """Cluster similar issues using TF-IDF and K-means."""
    # Prepare the preprocessed descriptions
    preprocessed_descriptions = df['preprocessed_description'].tolist()
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_descriptions)
    
    # Determine optimal number of clusters (simplified)
    num_clusters = min(7, len(df))  # Maximum 7 clusters or number of tickets
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    return df

def visualize_top_issues(df, output_file):
    """Create a bar chart of top 5 issue categories."""
    plt.figure(figsize=(10, 6))
    
    # Count issues by category
    category_counts = df['category'].value_counts().head(5)
    
    # Create bar chart
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Top 5 Issue Categories', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Tickets', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    plt.close()

def visualize_criticality_distribution(df, output_file):
    """Create a pie chart of criticality distribution."""
    plt.figure(figsize=(8, 8))
    
    # Count tickets by criticality
    criticality_counts = df['criticality'].value_counts()
    
    # Create pie chart
    plt.pie(
        criticality_counts.values, 
        labels=criticality_counts.index, 
        autopct='%1.1f%%',
        colors=sns.color_palette('Set2'),
        startangle=90,
        explode=[0.05] * len(criticality_counts)
    )
    plt.title('Ticket Criticality Distribution', fontsize=16)
    plt.axis('equal')
    
    # Save the figure
    plt.savefig(output_file)
    plt.close()

def visualize_ticket_trends(df, output_file):
    """Create a line chart of ticket volume over time."""
    plt.figure(figsize=(12, 6))
    
    # Convert date to datetime and extract date only
    df['date'] = pd.to_datetime(df['date'])
    df['date_only'] = df['date'].dt.date
    
    # Count tickets by date
    date_counts = df.groupby('date_only').size()
    
    # Create line chart
    plt.plot(date_counts.index, date_counts.values, marker='o', linestyle='-', linewidth=2)
    plt.title('Ticket Volume Trend', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Tickets', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    plt.close()

def visualize_root_causes(df, output_file):
    """Create a horizontal bar chart of root causes."""
    plt.figure(figsize=(10, 6))
    
    # Count tickets by root cause
    root_cause_counts = df['root_cause'].value_counts()
    
    # Create horizontal bar chart
    bars = plt.barh(root_cause_counts.index, root_cause_counts.values, color=sns.color_palette('viridis', len(root_cause_counts)))
    
    # Add count labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width}', ha='left', va='center')
    
    plt.title('Tickets by Root Cause', fontsize=16)
    plt.xlabel('Number of Tickets', fontsize=12)
    plt.ylabel('Root Cause', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    plt.close()

def process_tickets(input_file):
    """Process the support tickets CSV file."""
    print(f"Processing tickets from {input_file}...")
    
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_columns = ['ticket_id', 'date', 'customer_id', 'product', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")
        
        print(f"Loaded {len(df)} tickets.")
        
        # Preprocess descriptions for NLP
        df['preprocessed_description'] = df['description'].apply(preprocess_text)
        
        # Train a text classifier if possible
        classifier = train_text_classifier(df)
        
        # Categorize issues using the classifier if available, otherwise use keyword-based categorization
        df['category'] = df['description'].apply(lambda x: categorize_issue(x, classifier))
        
        # Determine root cause
        df['root_cause'] = df['category'].apply(determine_root_cause)
        
        # Assess criticality
        df['criticality'] = df['description'].apply(assess_criticality)
        
        # Cluster similar issues
        df = cluster_similar_issues(df)
        
        # Generate L2/L3 responses for medium and high criticality tickets
        l2_l3_tickets = df[(df['criticality'] == 'Medium') | (df['criticality'] == 'High')]
        l2_l3_tickets['l2_l3_response'] = l2_l3_tickets['category'].apply(generate_l2_l3_response)
        
        # Generate customer feedback
        df['customer_feedback'] = df['criticality'].apply(generate_customer_feedback)
        
        # Save L2/L3 tickets to file
        with open(L2_L3_TICKETS_FILE, 'w') as f:
            f.write(f"L2/L3 Tickets - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for _, ticket in l2_l3_tickets.iterrows():
                f.write(f"Ticket ID: {ticket['ticket_id']}\n")
                f.write(f"Date: {ticket['date']}\n")
                f.write(f"Customer ID: {ticket['customer_id']}\n")
                f.write(f"Product: {ticket['product']}\n")
                f.write(f"Description: {ticket['description']}\n")
                f.write(f"Category: {ticket['category']}\n")
                f.write(f"Root Cause: {ticket['root_cause']}\n")
                f.write(f"Criticality: {ticket['criticality']}\n")
                f.write(f"L2/L3 Response: {ticket['l2_l3_response']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Saved {len(l2_l3_tickets)} L2/L3 tickets to {L2_L3_TICKETS_FILE}")
        
        # Save customer feedback to file
        with open(CUSTOMER_FEEDBACK_FILE, 'w') as f:
            f.write(f"Customer Feedback - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for _, ticket in df.iterrows():
                f.write(f"Ticket ID: {ticket['ticket_id']}\n")
                f.write(f"Customer ID: {ticket['customer_id']}\n")
                f.write(f"Feedback: {ticket['customer_feedback']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Saved customer feedback for {len(df)} tickets to {CUSTOMER_FEEDBACK_FILE}")
        
        # Create visualizations
        visualize_top_issues(df, f"{VISUALIZATIONS_DIR}/top_issues.png")
        visualize_criticality_distribution(df, f"{VISUALIZATIONS_DIR}/criticality_distribution.png")
        visualize_ticket_trends(df, f"{VISUALIZATIONS_DIR}/ticket_trends.png")
        visualize_root_causes(df, f"{VISUALIZATIONS_DIR}/root_causes.png")
        
        print(f"Generated visualizations in {VISUALIZATIONS_DIR}")
        
        # Save processed data
        df.to_csv(f"{OUTPUT_DIR}/processed_tickets.csv", index=False)
        print(f"Saved processed ticket data to {OUTPUT_DIR}/processed_tickets.csv")
        
        return df
        
    except Exception as e:
        print(f"Error processing tickets: {str(e)}")
        raise

def main():
    """Main function to run the ticket triage automation."""
    print("Starting IT Support Ticket Triage Automation...")
    
    # Process the tickets
    processed_df = process_tickets(INPUT_FILE)
    
    # Print summary statistics
    print("\nTriage Summary:")
    print(f"Total tickets: {len(processed_df)}")
    print(f"Tickets by category:\n{processed_df['category'].value_counts()}")
    print(f"Tickets by criticality:\n{processed_df['criticality'].value_counts()}")
    print(f"Tickets by root cause:\n{processed_df['root_cause'].value_counts()}")
    
    print("\nTriage automation completed successfully!")

if __name__ == "__main__":
    main()

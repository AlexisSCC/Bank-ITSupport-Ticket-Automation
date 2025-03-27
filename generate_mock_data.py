#!/usr/bin/env python3
"""
Generate mock IT support ticket data for a digital banking platform.
This script creates a CSV file with realistic sample data for testing the triage automation system.
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import csv

# Initialize Faker
fake = Faker()

# Define constants
NUM_TICKETS = 1000
OUTPUT_FILE = "support_tickets.csv"

# Define possible products
PRODUCTS = [
    "Mobile Banking App", 
    "Web Portal", 
    "ATM Services", 
    "Credit Card Portal", 
    "Loan Application System",
    "Investment Dashboard",
    "Bill Payment Service",
    "Account Management System"
]

# Define possible issues with varying criticality
ISSUES = {
    "Login Failure": [
        "Unable to log in to {product} with correct credentials",
        "Getting 'Invalid username or password' despite using correct details",
        "Authentication fails every time I try to access my account",
        "Two-factor authentication not sending verification code",
        "Biometric login not working on my device"
    ],
    "Transaction Error": [
        "Payment failed but money was deducted from my account",
        "Duplicate transaction showing in my account history",
        "Transfer to external account stuck in 'processing' for 2 days",
        "Scheduled payment not executed on time",
        "International transfer showing incorrect exchange rate"
    ],
    "Account Access": [
        "Account locked after multiple login attempts",
        "Cannot reset password through the forgot password option",
        "Getting 'Your account has been temporarily suspended' message",
        "Unable to view my account balance and transactions",
        "Session keeps timing out too quickly while using the service"
    ],
    "App Performance": [
        "App crashes when trying to view transaction history",
        "Extremely slow loading times for all screens",
        "App freezes during fund transfers",
        "Cannot upload documents for loan application",
        "Charts and graphs not displaying correctly in investment dashboard"
    ],
    "Feature Request": [
        "Would like an option to categorize transactions automatically",
        "Need dark mode for the mobile app",
        "Request for biometric authentication for web portal",
        "Suggestion to add spending analytics feature",
        "Would like to receive push notifications for large transactions"
    ],
    "Data Discrepancy": [
        "Balance shown in app doesn't match my actual account balance",
        "Recent transactions not showing up in activity feed",
        "Incorrect interest calculation on my savings account",
        "Statement shows transactions I didn't make",
        "Credit score displayed is different from what other services show"
    ],
    "Security Concern": [
        "Received suspicious email claiming to be from the bank",
        "Noticed login attempt from unknown location",
        "Not receiving security alerts for login attempts",
        "Concerned about the security of my banking data",
        "Need to report potential phishing attempt targeting bank customers"
    ]
}

# Keywords that indicate criticality
HIGH_CRITICALITY_KEYWORDS = [
    "urgent", "immediately", "fraud", "stolen", "unauthorized", 
    "security breach", "hacked", "emergency", "critical", "locked out",
    "money missing", "wrong amount", "failed", "error", "lost money"
]

MEDIUM_CRITICALITY_KEYWORDS = [
    "not working", "issue", "problem", "slow", "delay", "bug", 
    "glitch", "incorrect", "difficulty", "trouble", "inconsistent",
    "intermittent", "occasionally", "sometimes"
]

def generate_ticket_data():
    """Generate a single ticket with realistic data."""
    customer_id = fake.uuid4()
    product = random.choice(PRODUCTS)
    
    # Select a random issue category and description
    issue_category = random.choice(list(ISSUES.keys()))
    description_template = random.choice(ISSUES[issue_category])
    description = description_template.format(product=product.lower())
    
    # Add some randomness to descriptions
    if random.random() < 0.3:
        description = f"{description}. {fake.sentence()}"
    
    # Determine criticality based on keywords and randomness
    if any(keyword in description.lower() for keyword in HIGH_CRITICALITY_KEYWORDS) or random.random() < 0.2:
        criticality = "High"
    elif any(keyword in description.lower() for keyword in MEDIUM_CRITICALITY_KEYWORDS) or random.random() < 0.5:
        criticality = "Medium"
    else:
        criticality = "Low"
    
    # Generate a realistic timestamp within the last 30 days
    days_ago = random.randint(0, 30)
    hours_ago = random.randint(0, 23)
    minutes_ago = random.randint(0, 59)
    timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
    date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate a status (mostly new tickets, some in progress or resolved)
    status_options = ["New", "In Progress", "Resolved"]
    status_weights = [0.7, 0.2, 0.1]
    status = random.choices(status_options, weights=status_weights)[0]
    
    return {
        "ticket_id": fake.unique.random_number(digits=6),
        "date": date_str,
        "customer_id": customer_id,
        "product": product,
        "description": description,
        "status": status,
        # The actual criticality will be determined by the triage system,
        # but we include a "true" value for validation
        "true_criticality": criticality,
        "true_category": issue_category
    }

def main():
    """Generate the mock data and save to CSV."""
    print(f"Generating {NUM_TICKETS} mock support tickets...")
    
    tickets = [generate_ticket_data() for _ in range(NUM_TICKETS)]
    df = pd.DataFrame(tickets)
    
    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Mock data saved to {OUTPUT_FILE}")
    
    # Print sample statistics
    print("\nSample Statistics:")
    print(f"Product distribution:\n{df['product'].value_counts()}")
    print(f"\nCriticality distribution:\n{df['true_criticality'].value_counts()}")
    print(f"\nCategory distribution:\n{df['true_category'].value_counts()}")
    
if __name__ == "__main__":
    main()

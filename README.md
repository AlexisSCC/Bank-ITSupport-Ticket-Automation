# IT Support Ticket Triage Automation

An automated system for triaging IT support tickets in a digital banking platform. This system processes support ticket data, categorizes issues using NLP, assesses criticality, groups by root cause, escalates tickets to L2/L3 support, generates customer feedback, and visualizes key metrics.

## Project Overview

This system automates the initial triage (Level 1 - L1) of IT support tickets by:

1. Processing ticket data from a CSV file
2. Using NLP to categorize issues and identify patterns
3. Assessing ticket criticality based on content analysis
4. Grouping similar issues by root cause
5. Automating ticket escalation to L2/L3 support teams
6. Generating appropriate customer feedback
7. Visualizing key metrics for analysis

## Features

- **Input Handling**: Reads and validates CSV files containing ticket data
- **NLP Categorization**: Uses natural language processing to classify problem types
- **Criticality Assessment**: Assigns Low/Medium/High criticality levels based on content analysis
- **Root Cause Analysis**: Groups tickets by underlying root causes
- **Ticket Escalation**: Automatically identifies tickets requiring L2/L3 support
- **Customer Feedback**: Generates appropriate responses based on ticket status
- **Data Visualization**: Creates insightful graphs showing ticket distribution and trends
- **Mock Data Generation**: Includes a script to generate realistic sample data for testing

## Project Structure

```
BankTicketTriage/
├── triage_automation.py     # Main script for ticket processing
├── generate_mock_data.py    # Script to generate sample data
├── requirements.txt         # Project dependencies
├── support_tickets.csv      # Generated sample data (after running generate_mock_data.py)
└── output/                  # Generated output files
    ├── processed_tickets.csv    # Processed ticket data with categories and criticality
    ├── tickets_l2_l3.txt        # Tickets escalated to L2/L3 support
    ├── customer_feedback.txt    # Generated customer feedback
    └── visualizations/          # Generated charts and graphs
        ├── top_issues.png           # Bar chart of top issue categories
        ├── criticality_distribution.png  # Pie chart of criticality levels
        ├── ticket_trends.png        # Line chart of ticket volume over time
        └── root_causes.png          # Bar chart of root causes
```

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. Clone this repository or download the files to your local machine

2. Navigate to the project directory:
   ```
   cd BankTicketTriage
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Generate sample data (if needed):
   ```
   python generate_mock_data.py
   ```
   This will create a `support_tickets.csv` file with realistic mock data.

2. Run the triage automation script:
   ```
   python triage_automation.py
   ```
   This will process the tickets, categorize them, generate feedback, and create visualizations.

3. View the results in the `output` directory:
   - Check `tickets_l2_l3.txt` for tickets escalated to L2/L3 support
   - Check `customer_feedback.txt` for generated customer responses
   - View the visualizations in the `visualizations` directory

## Customization

You can customize the system by modifying the following in `triage_automation.py`:

- `ISSUE_CATEGORIES`: Keywords used to categorize tickets
- `ROOT_CAUSES`: Mapping of issue categories to root causes
- `L2_L3_RESPONSES`: Response templates for escalated tickets
- `CUSTOMER_FEEDBACK`: Feedback templates for different criticality levels

## Data Format

The input CSV file should contain the following columns:
- `ticket_id`: Unique identifier for the ticket
- `date`: Timestamp when the ticket was created
- `customer_id`: Identifier for the customer
- `product`: The banking product or service related to the issue
- `description`: Detailed description of the problem
- `status` (optional): Current status of the ticket

## Visualizations

The system generates the following visualizations:

1. **Top Issues**: Bar chart showing the most common issue categories
2. **Criticality Distribution**: Pie chart showing the distribution of ticket criticality
3. **Ticket Trends**: Line chart showing ticket volume over time
4. **Root Causes**: Bar chart showing the distribution of underlying root causes

## Future Enhancements

Potential improvements for future versions:

- Integration with real ticketing systems (e.g., ServiceNow, Jira)
- Machine learning model for more accurate issue categorization
- Sentiment analysis for better criticality assessment
- Real-time monitoring dashboard
- Automated ticket assignment to specific support teams
- Integration with knowledge base for solution recommendations

import google.generativeai as genai
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import json
import os
import matplotlib.pyplot as plt
import math 
from collections import defaultdict

# --- 1. Configuration ---
try:
    # Assuming API Key is set up elsewhere in the real execution environment
    genai.configure(api_key="AIzaSyB2ROPgddv5SzQtSZXDlt4igvXyS0rR5GU") 
except Exception:
    pass

model = genai.GenerativeModel('gemini-flash-latest')
DATA_FILE = "data.json"
CURRENCY_SYMBOL = "₹"

# --- 2. Persistent Data Storage ---
def load_data():
    # Helper to calculate the start date for historical context
    ref_date = datetime.now()
    
    if not os.path.exists(DATA_FILE) or os.path.getsize(DATA_FILE) == 0:
        print("No data file found. Creating a new default dataset.")
        
        # Determine last full month dates
        last_month_end = ref_date.replace(day=1) - timedelta(days=1)
        last_month_start = last_month_end.replace(day=1)
        
        default_data = {
            "transactions": [
                {"date": (ref_date - timedelta(days=180)).isoformat(), "description": "Salary June", "amount": 65000.0, "category": "Income"},
                {"date": (ref_date - timedelta(days=150)).isoformat(), "description": "Shopping June", "amount": -2000.0, "category": "Shopping"},
                {"date": (ref_date - timedelta(days=120)).isoformat(), "description": "Salary July", "amount": 65000.0, "category": "Income"},
                {"date": (ref_date - timedelta(days=90)).isoformat(), "description": "Salary Aug", "amount": 65000.0, "category": "Income"},
                
                # Transactions for Last Full Month (September 2025)
                {"date": last_month_start.isoformat(), "description": "Salary Sept", "amount": 65000.0, "category": "Income"},
                {"date": (last_month_start + timedelta(days=5)).isoformat(), "description": "Grocery Mart (Last Month)", "amount": -1200.0, "category": "Groceries"},
                {"date": (last_month_start + timedelta(days=8)).isoformat(), "description": "Electricity Bill (Spike)", "amount": -4000.0, "category": "Bills & Utilities"}, # ANOMALY IN LAST FULL MONTH
                {"date": (last_month_start + timedelta(days=15)).isoformat(), "description": "Flipkart Purchase", "amount": -8500.0, "category": "Shopping"},
                {"date": (last_month_start + timedelta(days=20)).isoformat(), "description": "Swiggy Dinner", "amount": -550.0, "category": "Food & Drink"},
                {"date": (last_month_start + timedelta(days=25)).isoformat(), "description": "Movie Tickets", "amount": -1500.0, "category": "Entertainment"},
                
                # Transactions for Current Incomplete Month (October 2025)
                {"date": (ref_date - timedelta(days=2)).isoformat(), "description": "Salary Oct", "amount": 65000.0, "category": "Income"},
                {"date": (ref_date - timedelta(days=1)).isoformat(), "description": "Groceries BIG", "amount": -3000.0, "category": "Groceries"}, 
                
                # Historical average data points (for base calculation)
                {"date": (ref_date - timedelta(days=90)).isoformat(), "description": "Normal Bills Aug", "amount": -1700.0, "category": "Bills & Utilities"},
                {"date": (ref_date - timedelta(days=60)).isoformat(), "description": "Normal Bills Sept", "amount": -1500.0, "category": "Bills & Utilities"},
            ],
            "budgets": {"Groceries": 8000.0, "Shopping": 10000.0, "Food & Drink": 5000.0, "Transport": 3000.0, "Bills & Utilities": 15000.0},
            "goals": [{"name": "Rainy Day Fund", "target_amount": 50000.0, "saved_amount": 15000.0}]
        }
        save_data(default_data)
        return default_data
    with open(DATA_FILE, 'r') as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- 3. Core AI Function (The "Orchestrator") ---
def get_tool_call(user_query, data):
    prompt = f"""
    You are an AI orchestrator. Your job is to analyze the user's request and decide which of the available tools to call.
    You must respond with a single, clean JSON object specifying the 'tool_name' and its 'arguments'.

    *AVAILABLE TOOLS:*
    - get_summary(time_period: str = None): Provides a financial summary including total spending, total income, and top 5 spending categories for a given period.
    - get_financial_total(type: str, time_period: str = None): Calculates total income, total spending, or net savings for a period. Use 'SPENDING', 'INCOME', or 'NET SAVINGS' for the 'type' argument.
    - get_top_spending_category(time_period: str = None): Find the category with the highest TOTAL spending.
    - find_peak_spending_day_for_category(category: str, time_period: str = None): Finds the SINGLE BIGGEST transaction/day within a category.
    - visualize_spending(time_period: str = None): Creates a pie chart of spending.
    - find_transaction_date(description: str): Find the date of a specific past transaction.
    - add_transaction(description: str, amount: float, category: str, date: str = None): Add a new transaction (auto-negates amount for expenses).
    - get_financial_advice(): Provide advice on where to save money (based on past overspending).
    - check_budgets(): Get the status of all budgets.
    - add_budget(category: str, amount: float): Add or update a budget for a specific category.
    - get_balance(): Get the current total account balance.
    - contribute_to_goal(name: str, amount: float): Add savings to a goal.
    - check_goals(name: str = None): Check progress on one or all savings goals.
    - calculate_savings_plan(target_amount: float): Calculates a realistic monthly savings amount, suggests where to cut expenses (including 6-month discretionary spending and recent anomalies), and provides the time needed to reach the target.
    - add_savings_goal(name: str, target_amount: float, months: int): Creates and saves a new savings goal with the required target amount and timeframe, AND provides a recommendation.
    - identify_unnecessary_spending(time_period: str = None): Finds discretionary categories with high spending and identifies spending anomalies (unusual spikes) in specific categories compared to historical data.

    *CRITICAL RULES:*
    - **CRITICAL RULE 0:** If the User Query is a simple greeting (e.g., "hello", "hi", "good morning") or purely conversational, you **MUST** respond with the special `tool_name`: **`greeting_response`** and use the arguments: `{{ "response": "Hi there! How can I help you with your finances today? I can help you with saving goals, budget checks, and spending analysis." }}`.
    - If the user asks for a **"summary"** or **"overview"**, you **MUST** call **get_summary**.
    - If user asks for **total spending**, **total income**, or **how much they saved** (net savings), you MUST call **get_financial_total** and use the appropriate 'type' argument.
    - If the user asks about **"unnecessary"** or **"unwanted"** spending, you **MUST** call **identify_unnecessary_spending**.
    - If the user explicitly states a goal with a target amount AND a time limit, you MUST call **add_savings_goal**.
    - If user asks for a forward-looking question about a *target amount* and *saving*, such as "how much do I need to save" or "I want to buy X, how can I save for it," you MUST call **calculate_savings_plan**.


    *User Query:* "{user_query}"
    *JSON Output:*
    """
    try:
        response = model.generate_content(prompt)
        json_string = response.text.strip().replace('`', '').replace('json', '')
        return json.loads(json_string)
    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        print(f"Error decoding tool call JSON: {e}")
        return {"tool_name": "unknown", "arguments": {}}

# --- 4. The "Tool Belt" (Backend Functions) ---

def get_df(data):
    if not data.get('transactions'): return pd.DataFrame()
    df = pd.DataFrame(data['transactions'])
    df['date'] = pd.to_datetime(df['date'])
    return df

def parse_time_period(time_period_str):
    today = datetime.now()
    if not time_period_str: return None, None
    time_period_str = time_period_str.lower()

    # Check for a specific year like "year 2023" or "2023"
    import re
    match = re.search(r'\b(20\d{2})\b', time_period_str)
    if match:
        year = int(match.group(1))
        start_date = datetime(year, 1, 1, 0, 0, 0)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        return start_date, end_date

    if 'this month' in time_period_str:
        start_date, end_date = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0), today
    elif 'last month' in time_period_str:
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start_date, end_date = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0), last_month_end.replace(hour=23, minute=59, second=59)
    elif 'last two months' in time_period_str:
        end_date = (today.replace(day=1) - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        start_date = (end_date.replace(day=1) - pd.DateOffset(months=1)).replace(day=1, hour=0, minute=0, second=0)
    elif 'last three months' in time_period_str:
        end_date = (today.replace(day=1) - timedelta(days=1)).replace(hour=23, minute=59, second=59)
        start_date = (end_date.replace(day=1) - pd.DateOffset(months=2)).replace(day=1, hour=0, minute=0, second=0)
    elif 'last four months' in time_period_str:
        d = today.replace(day=1)
        for _ in range(4):
            d = d - timedelta(days=1)
            d = d.replace(day=1)
        start_date, end_date = d, today
    elif 'last year' in time_period_str:
        last_year = today.year - 1
        start_date = datetime(last_year, 1, 1, 0, 0, 0)
        end_date = datetime(last_year, 12, 31, 23, 59, 59)
    else: return None, None
    return start_date, end_date

# --- HELPER FUNCTIONS ---

def get_spending_analysis(data):
    df = get_df(data)
    # Analyze the last 6 full months for a more stable savings capacity calculation
    end_date = datetime.now().replace(day=1) - timedelta(days=1)
    # Use pandas DateOffset for robust month calculations
    start_date = end_date.replace(day=1) - pd.DateOffset(months=5) # 6 months total
    
    non_discretionary = ['Bills & Utilities', 'Income', 'Savings', 'Groceries', 'Transport'] 
    
    if df.empty:
        return {'avg_discretionary_monthly': 0, 'top_categories_6_months': [], 'category_spending_6_months': {}}

    recent_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    expenses = recent_df[recent_df['amount'] < 0]
    
    discretionary = expenses[~expenses['category'].isin(non_discretionary)].copy()
    
    if discretionary.empty:
        return {'avg_discretionary_monthly': 0, 'top_categories_6_months': [], 'category_spending_6_months': {}}

    # Calculate 6-month total and average monthly
    discretionary_total_6_months = abs(discretionary['amount'].sum())
    avg_discretionary_monthly = discretionary_total_6_months / 6
    
    discretionary['amount'] = abs(discretionary['amount'])
    category_totals = discretionary.groupby('category')['amount'].sum().sort_values(ascending=False)
    top_categories = category_totals.head(2).index.tolist()
    
    return {
        'avg_discretionary_monthly': avg_discretionary_monthly, 
        'top_categories_6_months': top_categories,
        'category_spending_6_months': category_totals.to_dict()
    }

# NEW HELPER FUNCTION: Checks for anomalies in a target year vs the previous year
def get_spending_anomalies_yearly(df, target_year):
    target_year_df = df[df['date'].dt.year == target_year]
    previous_year_df = df[df['date'].dt.year == target_year - 1]

    if previous_year_df.empty:
        return [] # No previous year data to compare against

    target_year_spending = target_year_df[target_year_df['amount'] < 0].groupby('category')['amount'].sum().abs()
    previous_year_spending = previous_year_df[previous_year_df['amount'] < 0].groupby('category')['amount'].sum().abs()

    anomalies = []
    
    # Compare spending for each category in the target year against the previous year
    for category, target_spend in target_year_spending.items():
        previous_spend = previous_year_spending.get(category, 0)
        
        # If spending in the target year is greater than the previous year, it's a spike
        if target_spend > previous_spend:
            anomalies.append({
                'category': category,
                'previous_year_spend': previous_spend,
                'target_year_spend': target_spend
            })
            
    return anomalies

# MODIFIED HELPER FUNCTION (v1.9): Checks for anomalies in a given period
def get_spending_anomalies(df, start_date, end_date, num_months):
    
    # 1. Calculate Historical Average (from before the check period)
    historical_df = df[df['date'] < start_date].copy()
    
    if historical_df.empty:
        return [] # No historical data to compare against

    historical_df['year_month'] = historical_df['date'].dt.to_period('M')
    # Calculate average by category, only for periods where spending occurred
    monthly_spending = historical_df.groupby(['year_month', 'category'])['amount'].sum().abs().reset_index()
    historical_average = monthly_spending.groupby('category')['amount'].mean()
    
    anomalies = []
    
    # 2. Analyze the Anomaly Check Period
    comparison_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    comparison_df['amount'] = comparison_df['amount'].abs()
    
    period_spending = comparison_df.groupby('category')['amount'].sum()
    
    CORE_EXPENSES = [c for c in df['category'].unique() if c not in ['Income', 'Savings']]
    ANOMALY_THRESHOLD = 2.0 # 200% (2x) the historical average

    for category in CORE_EXPENSES:
        avg_monthly = historical_average.get(category, 0)
        current_period_spend = period_spending.get(category, 0)
        
        # Compare the spending in the period against the expected historical spending for the same duration
        if avg_monthly > 0 and current_period_spend > (avg_monthly * num_months * ANOMALY_THRESHOLD):
            anomalies.append({
                'category': category,
                'avg_spend': avg_monthly,
                'current_spend': current_period_spend,
                'num_months': num_months
            })

    return anomalies

# --- TOOL FUNCTIONS ---

# New handler for the conversational greeting response
def handle_greeting(response: str):
    return response

def get_financial_total(data, type: str, time_period: str = None):
    """Calculates total income, total spending, or net savings for a period."""
    df = get_df(data)
    
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
    if df.empty:
        return f"No transactions found for the specified period."

    total_income = df[df['amount'] > 0]['amount'].sum()
    total_spending = df[df['amount'] < 0]['amount'].sum() 

    type = type.upper()
    
    if type == 'SPENDING':
        return f"Your total spending {('in ' + time_period) if time_period else ''} was **{CURRENCY_SYMBOL}{abs(total_spending):,.2f}**."
    elif type == 'INCOME':
        return f"Your total income {('in ' + time_period) if time_period else ''} was **{CURRENCY_SYMBOL}{total_income:,.2f}**."
    elif type == 'NET SAVINGS':
        net_savings = total_income + total_spending
        if net_savings >= 0:
            return (f"Your net savings {('in ' + time_period) if time_period else ''} (Income - Spending) was "
                    f"**{CURRENCY_SYMBOL}{net_savings:,.2f}**.")
        else:
             return (f"Your net savings {('in ' + time_period) if time_period else ''} (Income - Spending) was a deficit of "
                    f"**{CURRENCY_SYMBOL}{abs(net_savings):,.2f}**.")
    else:
        return f"Invalid type '{type}'. Please use 'SPENDING', 'INCOME', or 'NET SAVINGS'."


def get_summary(data, time_period: str = None):
    """Provides a financial summary including total spending, total income, and top 5 spending categories."""
    df = get_df(data)
    period_text = "all time"
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: 
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            period_text = time_period
    
    if df.empty:
        return f"No transactions found for the specified period: {period_text}"

    total_income = df[df['amount'] > 0]['amount'].sum()
    total_spending = abs(df[df['amount'] < 0]['amount'].sum())

    expenses = df[df['amount'] < 0].copy()
    
    top_categories_report = "Top 5 Spending Categories:\n"
    if expenses.empty:
        top_categories_report += "No spending in this period."
    else:
        expenses['amount'] = abs(expenses['amount'])
        category_totals = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
        top_5_categories = category_totals.head(5)
        
        if top_5_categories.empty:
             top_categories_report += "No spending in this period."
        else:
            for category, amount in top_5_categories.items():
                top_categories_report += f"- {category}: {CURRENCY_SYMBOL}{amount:,.2f}\n"

    report = (f"Financial Summary for {period_text.title()}:\n"
              f"----------------------------------------\n"
              f"Total Income:   {CURRENCY_SYMBOL}{total_income:,.2f}\n"
              f"Total Spending: {CURRENCY_SYMBOL}{total_spending:,.2f}\n"
              f"----------------------------------------\n"
              f"{top_categories_report.strip()}")
              
    return report.strip()

def get_top_spending_category(data, time_period: str = None):
    df = get_df(data)
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    expenses = df[df['amount'] < 0].copy()
    if expenses.empty: return "No expenses found for this period."
    expenses['amount'] = abs(expenses['amount'])
    category_totals = expenses.groupby('category')['amount'].sum()
    top_category, top_amount = category_totals.idxmax(), category_totals.max()
    return f"Your top spending category was *'{top_category}'*, with a total of {CURRENCY_SYMBOL}{top_amount:,.2f}."

def find_peak_spending_day_for_category(data, category: str, time_period: str = None):
    df = get_df(data)
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: 
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    category_expenses = df[(df['category'].str.lower() == category.lower()) & (df['amount'] < 0)]
    if category_expenses.empty:
        return f"No spending found for '{category}' in this period."
    peak_transaction = category_expenses.loc[category_expenses['amount'].idxmin()]
    return (f"Your highest spending on '{category}' was for '{peak_transaction['description']}' "
            f"on {peak_transaction['date'].strftime('%Y-%m-%d')} for {CURRENCY_SYMBOL}{abs(peak_transaction['amount']):,.2f}.")

def find_transaction_date(data, description: str):
    df = get_df(data)
    matches = df[df['description'].str.contains(description, case=False)]
    if matches.empty: return f"I couldn't find any transaction matching '{description}'."
    latest = matches.sort_values(by='date', ascending=False).iloc[0]
    return f"The last transaction for '{description}' was on *{latest['date'].strftime('%Y-%m-%d')}* for {CURRENCY_SYMBOL}{abs(latest['amount']):,.2f}."

def visualize_spending(data, time_period: str = None):
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend for Flask/threaded environments
    import matplotlib.pyplot as plt
    df = get_df(data)
    period_text = "All Time"
    if time_period:
        start_date, end_date = parse_time_period(time_period)
        if start_date and end_date: 
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            period_text = time_period.title()
    expenses = df[df['amount'] < 0].copy()
    if expenses.empty: return "No spending data found for this period to visualize."
    expenses['amount'] = abs(expenses['amount'])
    category_totals = expenses.groupby('category')['amount'].sum()
    plt.style.use('seaborn-v0_8-deep')
    plt.figure(figsize=(10, 8))
    plt.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
    plt.title(f'Spending Breakdown for {period_text}', fontsize=16)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    file_name = "spending_chart.png"
    plt.savefig(file_name)
    plt.close()
    return f"I've generated a pie chart of your spending and saved it as *{file_name}*."

def get_financial_advice(data):
    df = get_df(data)
    budgets = data.get('budgets', {})
    if not budgets: return "Set some budgets first for me to give advice."
    last_month_str = (datetime.now().replace(day=1) - timedelta(days=1)).strftime("%B")
    start_date, end_date = parse_time_period('last month')
    if not start_date: return "Not enough time has passed to analyze last month's data."
    last_month_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    if last_month_df.empty: return f"I don't have enough data from {last_month_str} to give advice."
    overspent_categories = [(cat, abs(last_month_df[last_month_df['category'] == cat]['amount'].sum()) - budget) for cat, budget in budgets.items() if abs(last_month_df[last_month_df['category'] == cat]['amount'].sum()) > budget]
    if not overspent_categories:
        expenses = last_month_df[last_month_df['amount'] < 0]
        discretionary = expenses[~expenses['category'].isin(['Bills & Utilities', 'Income', 'Savings'])]
        if discretionary.empty: return "You did a great job staying within budget last month!"
        top_category = discretionary.groupby('category')['amount'].sum().abs().idxmax()
        return f"You stayed within all budgets last month! Your highest discretionary spending was in *'{top_category}'*, which could be an area to watch."
    top_offender = max(overspent_categories, key=lambda item: item[1])
    return f"Based on last month, the best place to save money might be in *'{top_offender[0]}'*. You went over your budget by {CURRENCY_SYMBOL}{top_offender[1]:,.2f}."

def add_transaction(data, description: str, amount: float, category: str, date: str = None):
    if not all([description, amount, category]): return "I need a description, amount, and category."
    GAIN_CATEGORIES = ['Income', 'Savings'] 
    final_amount = amount
    if category not in GAIN_CATEGORIES and final_amount > 0:
        final_amount = -amount
    try:
        transaction_date = parse_date(date) if date else datetime.now()
    except (ValueError, TypeError):
        transaction_date = datetime.now()
    new_trans = { "date": transaction_date.isoformat(), "description": description, "amount": final_amount, "category": category }
    data['transactions'].append(new_trans)
    save_data(data)
    return f"Added: '{description}' ({CURRENCY_SYMBOL}{abs(final_amount):,.2f}) to '{category}' for {transaction_date.strftime('%Y-%m-%d')}."

def get_balance(data):
    df = get_df(data)
    total = df['amount'].sum()
    return f"Your current total balance is {CURRENCY_SYMBOL}{total:,.2f}."

def check_budgets(data):
    df = get_df(data)
    this_month_df = df[df['date'].dt.month == datetime.now().month]
    report = "Budget Status (This Month):\n"
    for category, budget_amount in data.get('budgets', {}).items():
        spent = abs(this_month_df[this_month_df['category'] == category]['amount'].sum())
        remaining = budget_amount - spent
        report += f"- {category}: Spent {CURRENCY_SYMBOL}{spent:,.2f} of {CURRENCY_SYMBOL}{budget_amount:,.2f} ({CURRENCY_SYMBOL}{abs(remaining):,.2f} {'left' if remaining >=0 else 'over'}).\n"
    return report.strip()

def add_budget(data, category: str, amount: float):
    """Adds or updates a budget for a specific category."""
    if not category or amount is None:
        return "I need both a category and an amount to set a budget."
    if amount < 0:
        return "Budget amount must be a positive number."
    
    data['budgets'][category] = amount
    save_data(data)
    return f"Budget for '{category}' has been set to {CURRENCY_SYMBOL}{amount:,.2f}."

def check_goals(data, name: str = None):
    goals = data.get('goals', [])
    if not goals: return "You haven't set any savings goals yet."
    if name:
        for goal in goals:
            if goal.get('name', '').lower() == name.lower():
                saved, target = goal.get('saved_amount', 0), goal.get('target_amount', 0)
                progress = (saved / target) * 100 if target > 0 else 0
                monthly_req = goal.get('required_monthly_saving')
                monthly_text = f" (Requires {CURRENCY_SYMBOL}{monthly_req:,.2f}/month)" if monthly_req else ""
                return (f"For your '{name}' goal, you have saved {CURRENCY_SYMBOL}{saved:,.2f} of "
                        f"{CURRENCY_SYMBOL}{target:,.2f} ({progress:.0f}% complete).{monthly_text}")
        return f"Sorry, I couldn't find a goal named '{name}'."
    else:
        report = "Here are your current savings goals:\n"
        for goal in goals:
            g_name, saved, target = goal.get('name'), goal.get('saved_amount', 0), goal.get('target_amount', 0)
            monthly_req = goal.get('required_monthly_saving')
            monthly_text = f", Monthly Target: {CURRENCY_SYMBOL}{monthly_req:,.2f}" if monthly_req else ""
            report += f"- {g_name}: {CURRENCY_SYMBOL}{saved:,.2f} / {CURRENCY_SYMBOL}{target:,.2f} saved{monthly_text}.\n"
        return report.strip()

def contribute_to_goal(data, name: str, amount: float):
    goals = data.get('goals', [])
    for goal in goals:
        if goal.get('name', '').lower() == name.lower():
            goal['saved_amount'] += amount
            add_transaction(data, f"Contribution to {name}", amount, "Savings") 
            return f"Great! I've added {CURRENCY_SYMBOL}{amount:,.2f} to your '{name}' goal."
    return f"Sorry, I couldn't find a goal named '{name}'."

def calculate_savings_plan(data, target_amount: float):
    """Calculates a realistic monthly saving needed based on available discretionary spending and suggests truly non-essential categories for cutting."""
    
    if target_amount <= 0:
        return "Please specify a positive amount for your target purchase."

    df = get_df(data)
    
    # 1. Analyze 6-Month Spending for Capacity
    analysis = get_spending_analysis(data)
    avg_discretionary_monthly = analysis['avg_discretionary_monthly']
    top_categories = analysis['top_categories_6_months']
    category_spending = analysis['category_spending_6_months']
    
    # 2. Check for recent Anomalies (Now checks LAST FULL MONTH)
    anomaly_start_date, anomaly_end_date = parse_time_period('last month')
    anomalies = get_spending_anomalies(df, anomaly_start_date, anomaly_end_date, 1)
    
    # Determine Realistic Saving Capacity (30% of average discretionary spending)
    if avg_discretionary_monthly == 0:
        realistic_monthly_saving = 500.0
    else:
        realistic_monthly_saving = avg_discretionary_monthly * 0.30 
        if realistic_monthly_saving < 200:
             realistic_monthly_saving = 200.0
             
    # Calculate time to reach the target
    months_needed_raw = target_amount / realistic_monthly_saving
    months_needed_rounded = int(math.ceil(months_needed_raw)) 
    
    # --- Generate Detailed Recommendation Output ---
    
    # A. Savings Target & Time
    report = (f"Based on your spending habits over the last 6 months, a realistic monthly saving amount is "
            f"{CURRENCY_SYMBOL}{realistic_monthly_saving:,.2f} per month. This means you will reach your goal "
            f"of {CURRENCY_SYMBOL}{target_amount:,.2f} in approximately *{months_needed_rounded} months*.")
    
    report += "\n\n" + "-" * 40 + "\n"
    
    # B. Discretionary Spending Breakdown (The core area for cuts)
    report += "## 💰 Where to Cut (Discretionary Spending: Last 6 Months)\n"
    if top_categories:
        report += f"To achieve this, focus on cutting spending in your highest non-essential categories (over the last 6 months):\n"
        for category in top_categories:
            monthly_avg = category_spending.get(category, 0) / 6 if 6 > 0 else 0
            report += f"- *{category}*: You spent {CURRENCY_SYMBOL}{category_spending.get(category, 0):,.2f} (Average of {CURRENCY_SYMBOL}{monthly_avg:,.0f}/month).\n"
    else:
        report += "You have minimal discretionary spending. You may need to look for ways to boost income.\n"
        
    report += "\n" + "-" * 40 + "\n"

    # C. Spending Anomalies (Immediate/Short-term warning)
    report += "## ⚠ Spending Anomalies (Spikes in the Last Full Month)\n"
    if anomalies:
        report += "We detected recent unusual spikes that could derail your plan. Address these immediately:\n"
        for anomaly in anomalies:
            report += (f"- *{anomaly['category']}: You typically spend **{CURRENCY_SYMBOL}{anomaly['avg_spend']:,.0f}** per month, but you spent "
                       f"**{CURRENCY_SYMBOL}{anomaly['current_spend']:,.0f}** last month. This is *{anomaly['current_spend']/anomaly['avg_spend']:.1f}x* your usual rate!\n")
    else:
        report += "No major spending anomalies (unusual spikes) were detected last month.\n"

    return report.strip()


def add_savings_goal(data, name: str, target_amount: float, months: int):
    """Creates and saves a new savings goal with the required target amount and timeframe, and provides a recommendation."""
    
    if target_amount <= 0 or months <= 0:
        return "Please provide positive values for the target amount and months."
    
    required_monthly_saving = target_amount / months
    
    new_goal = {
        "name": name, 
        "target_amount": target_amount, 
        "saved_amount": 0.0, 
        "required_monthly_saving": required_monthly_saving 
    }
    data['goals'].append(new_goal)
    save_data(data)
    
    analysis = get_spending_analysis(data)
    avg_discretionary_monthly = analysis['avg_discretionary_monthly']
    top_categories = analysis['top_categories_6_months']
    
    base_response = (f"Goal '{name}' (Target: {CURRENCY_SYMBOL}{target_amount:,.2f}) has been successfully added! "
                     f"To reach this goal in **{months} months**, you need to save "
                     f"*{CURRENCY_SYMBOL}{required_monthly_saving:,.2f} per month*.")
    
    if avg_discretionary_monthly == 0:
        recommendation = "You're already very frugal. To meet this goal, you may need to look for ways to boost income."
    elif required_monthly_saving > avg_discretionary_monthly:
        shortfall = required_monthly_saving - avg_discretionary_monthly
        recommendation = (f"**⚠️ ALERT:** This monthly target is higher than your average monthly discretionary spending over the last 6 months "
                          f"({CURRENCY_SYMBOL}{avg_discretionary_monthly:,.2f}). "
                          f"You will need to cut all non-essential spending **AND** find a way to save an extra {CURRENCY_SYMBOL}{shortfall:,.2f} per month.")
    else: # required_monthly_saving <= avg_discretionary_monthly
        if required_monthly_saving <= (avg_discretionary_monthly * 0.30):
             recommendation = "This goal is highly achievable with your current spending patterns! Start by cutting minor expenses."
        else:
            if len(top_categories) > 1:
                cut_suggestion = f"The best areas for savings cuts are: **'{top_categories[0]}'** and **'{top_categories[1]}'** (based on last 6 months)."
            elif len(top_categories) == 1:
                cut_suggestion = f"The best area for savings cuts is **'{top_categories[0]}'** (based on last 6 months)."
            else:
                cut_suggestion = "Focus on reducing your non-essential spending."
            recommendation = f"This goal is achievable! {cut_suggestion}"

    return f"{base_response}\n\n**Recommendation:** {recommendation}"


def identify_unnecessary_spending(data, time_period: str = None):
    df = get_df(data)
    if df.empty:
        return "I need more transaction data to analyze your spending."
    
    # Create a copy with absolute amounts for the first part of the analysis
    df_abs = df.copy()
    df_abs['amount'] = df_abs['amount'].abs() 
    DISCRETIONARY_CATEGORIES = ['Shopping', 'Entertainment', 'Food & Drink']
    
    # Define time period for Discretionary Spending Analysis (Part 1)
    start_date, end_date = parse_time_period(time_period)
    time_frame_text = time_period.title() if time_period else "Last 3 Months"

    if start_date and end_date:
        current_df = df_abs[(df_abs['date'] >= start_date) & (df_abs['date'] <= end_date)]
    else:
        start_date = datetime.now() - timedelta(days=90)
        current_df = df_abs[df_abs['date'] >= start_date]
        time_frame_text = "Last 3 Months"

    discretionary_spending = current_df[current_df['category'].isin(DISCRETIONARY_CATEGORIES)].groupby('category')['amount'].sum().sort_values(ascending=False)
    
    report = f"## 💸 Unnecessary Spending Analysis for {time_frame_text}\n"
    report += "-" * 40 + "\n"
    
    # 1. Highest Discretionary Categories (Uses the period requested by the user)
    report += "### 1. Highest Discretionary Spending Categories\n"
    if discretionary_spending.empty:
        report += "You had minimal spending in typical discretionary categories like Shopping, Entertainment, and Food & Drink in this period.\n"
    else:
        report += "The most significant discretionary spending came from these categories:\n"
        for category, total in discretionary_spending.items():
            report += f"- **{category}**: {CURRENCY_SYMBOL}{total:,.2f}\n"

    report += "\n" + "-" * 40 + "\n"

    # 2. Spending Anomalies (Spikes) - Logic now depends on the time period
    is_yearly_analysis = time_period and 'year' in time_period.lower()
    is_monthly_analysis = time_period and ('last month' in time_period.lower() or 'last two months' in time_period.lower() or 'last three months' in time_period.lower())

    if is_yearly_analysis:
        target_year = start_date.year
        report += f"### 2. Spending Spikes (Compared to Previous Year)\n"
        # Pass the original df, not the one with absolute values
        anomalies = get_spending_anomalies_yearly(df, target_year)
        
        if anomalies:
            for anomaly in anomalies:
                report += (f"**⚠️ Spike in {anomaly['category']}:** Spending increased from "
                           f"{CURRENCY_SYMBOL}{anomaly['previous_year_spend']:,.0f} in {target_year - 1} to "
                           f"**{CURRENCY_SYMBOL}{anomaly['target_year_spend']:,.0f}** in {target_year}.\n")
        else:
            report += f"No significant spending increases found in {target_year} compared to the previous year.\n"
    else: # Default to monthly anomaly check
        # Determine the period for monthly anomaly check
        num_months = 1
        period_text = "Last Full Month"
        if is_monthly_analysis:
            if 'two' in time_period:
                num_months = 2
                period_text = "Last 2 Full Months"
            elif 'three' in time_period:
                num_months = 3
                period_text = "Last 3 Full Months"
        
        # For default (no time_period), we check the last full month
        if not time_period:
            start_date, end_date = parse_time_period('last month')

        report += f"### 2. Spending Anomalies (Monthly Spikes)\n"
        anomalies = get_spending_anomalies(df, start_date, end_date, num_months)
        
        if anomalies:
            for anomaly in anomalies:
                expected_spend = anomaly['avg_spend'] * anomaly['num_months']
                report += (f"**⚠️ Spike in {anomaly['category']} ({period_text}):** You typically spend around "
                           f"{CURRENCY_SYMBOL}{expected_spend:,.0f} in a {anomaly['num_months']}-month period on **{anomaly['category']}**, but you spent "
                           f"**{CURRENCY_SYMBOL}{anomaly['current_spend']:,.0f}**. This is **{anomaly['current_spend']/expected_spend:.1f}x** your usual rate!\n")
        else:
            report += f"No significant spending spikes (anomalies) were detected in {period_text.lower()}.\n"
        
    return report.strip()

def generate_proactive_insight(data):
    return ""
# --- End of Tool Definitions ---


# --- 5. The Main Executor Loop ---
def main():
    data = load_data()
    proactive_insight = generate_proactive_insight(data)
    
    print("🤖 AI Financial Assistant (God Level v2.1 - Critical Fix)")
    if proactive_insight:
        print(f"   {proactive_insight}")
    print("   Now handles greetings correctly and without a Python error! How can I help you today? 🎯")
    print("   Type 'exit' to quit.")

    tool_belt = {
        "greeting_response": handle_greeting, # Added new handler
        "get_summary": get_summary,
        "get_financial_total": get_financial_total,
        "get_top_spending_category": get_top_spending_category,
        "find_peak_spending_day_for_category": find_peak_spending_day_for_category,
        "visualize_spending": visualize_spending,
        "find_transaction_date": find_transaction_date,
        "get_financial_advice": get_financial_advice,
        "add_transaction": add_transaction,
        "get_balance": get_balance,
        "check_budgets": check_budgets,
        "add_budget": add_budget,
        "check_goals": check_goals,
        "contribute_to_goal": contribute_to_goal,
        "calculate_savings_plan": calculate_savings_plan,
        "add_savings_goal": add_savings_goal,
        "identify_unnecessary_spending": identify_unnecessary_spending,
    }

    while True:
        current_data = load_data()
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("🤖 Goodbye!") 
            break

        tool_call = get_tool_call(user_query, current_data)
        tool_name = tool_call.get("tool_name")
        arguments = tool_call.get("arguments", {})

        print(f"🧠 (Debug) AI decided to call tool: '{tool_name}' with arguments: {arguments}")

        if tool_name == "greeting_response":
            # Special case handler for conversational greetings
            response = arguments.get("response", "Hi! How can I help?")
        elif tool_name in tool_belt:
            try:
                # All other tool calls require 'data' argument which is passed from the main loop
                # The way arguments are handled here relies on the tool definition
                if tool_name in ["get_financial_advice", "check_budgets", "get_balance", "get_top_spending_category", "visualize_spending", "generate_proactive_insight", "check_goals"]:
                    # These only require data
                    response = tool_belt[tool_name](data=current_data)
                elif tool_name in ["add_transaction", "contribute_to_goal", "calculate_savings_plan", "add_savings_goal", "add_budget"]:
                    # These require data plus named arguments
                    response = tool_belt[tool_name](data=current_data, **arguments)
                else:
                    # Generic handler for tools requiring data + optional args or others
                    response = tool_belt[tool_name](data=current_data, **arguments)
            except TypeError as e:
                # Catching cases where arguments are missing or incorrect
                print(f"Error: Missing or incorrect arguments for tool '{tool_name}'. Details: {e}")
                response = f"I'm missing some information to run the tool '{tool_name}'. Please try providing all necessary details."
            except Exception as e:
                response = f"An error occurred while running the tool: {e}"
        else:
            response = "I'm not sure how to do that. Please try rephrasing."
            
        print(f"🤖 Assistant: {response}")

if __name__ == "__main__":
    main()
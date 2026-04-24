import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os
import io
from datetime import datetime, timedelta
import pdfplumber
from PIL import Image
import base64
import re

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💰 Expense Detector AI Agent",
    page_icon="💰",
    layout="wide"
)

# ─── Data File ─────────────────────────────────────────────────────────────────
DATA_FILE = "expenses.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ─── Parse Expenses from Text ──────────────────────────────────────────────────
def parse_expenses_from_text(text):
    expenses = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Try to find amount pattern like: item 500 or 500 item or item: 500
        patterns = [
            r'(.+?)[:\-]\s*Rs\.?\s*(\d+(?:\.\d+)?)',
            r'(.+?)\s+(\d+(?:\.\d+)?)\s*(?:Rs|PKR|rs)?$',
            r'(\d+(?:\.\d+)?)\s*(?:Rs|PKR|rs)?\s+(.+)',
            r'(.+?)\s+(\d+(?:\.\d+)?)',
        ]
        matched = False
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()
                try:
                    # Try second group as amount
                    amount = float(groups[1])
                    name = groups[0].strip()
                    expenses.append({"name": name, "amount": amount})
                    matched = True
                    break
                except:
                    try:
                        amount = float(groups[0])
                        name = groups[1].strip()
                        expenses.append({"name": name, "amount": amount})
                        matched = True
                        break
                    except:
                        pass
        if not matched and line:
            # Just store the line as unknown expense
            expenses.append({"name": line, "amount": 0.0})
    return expenses

# ─── AI Advice ─────────────────────────────────────────────────────────────────
def get_ai_advice(df):
    if df.empty:
        return "No data available for advice."
    
    total = df['amount'].sum()
    top_expense = df.groupby('category')['amount'].sum().idxmax() if 'category' in df.columns else df.nlargest(1, 'amount')['name'].values[0]
    
    advice = f"""
### 🤖 AI Expense Advisor - Future Planning

**📊 Current Spending Analysis:**
- Total Expenses: Rs. {total:,.0f}
- Highest Spending Area: {top_expense}

**📈 Future Price Predictions (Pakistan Market Trends):**
- 🛒 **Groceries & Food**: Prices likely to increase 8-12% next month due to seasonal changes
- ⚡ **Electricity & Utilities**: Expected 5-10% increase in summer months
- 🚗 **Transport/Fuel**: Petrol prices fluctuate - save Rs. 2,000-3,000 buffer
- 🏥 **Healthcare**: Medical costs rising 10-15% annually - consider health savings
- 📱 **Technology**: Prices stable but dollar rate affects imports

**💡 Money Saving Recommendations:**
1. 🛍️ Buy groceries in bulk to save 15-20%
2. ⚡ Reduce electricity usage between 6-10 PM (peak hours)
3. 🚌 Use public transport 2-3 days/week
4. 🍳 Cook at home instead of dining out - save Rs. 5,000-8,000/month
5. 📋 Create a monthly budget and stick to 80% of income for expenses

**🎯 Next Month Budget Plan:**
- Increase your savings target by 10%
- Set aside emergency fund: 20% of income
- Review subscriptions and cancel unused ones

**⚠️ Alert:** If your expenses exceed 70% of income, consider reviewing luxury spending!
    """
    return advice

# ─── Charts ────────────────────────────────────────────────────────────────────
def create_pie_chart(df):
    if df.empty:
        return None
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    category_data = df.groupby('name')['amount'].sum()
    colors = plt.cm.Set3(range(len(category_data)))
    wedges, texts, autotexts = ax.pie(
        category_data.values,
        labels=category_data.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight('bold')
    ax.set_title('Expense Distribution (%)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_bar_chart(df, title="Monthly Expenses"):
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    category_data = df.groupby('name')['amount'].sum().sort_values(ascending=False)
    bars = ax.bar(range(len(category_data)), category_data.values, 
                  color=plt.cm.viridis([i/len(category_data) for i in range(len(category_data))]))
    ax.set_xticks(range(len(category_data)))
    ax.set_xticklabels(category_data.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Amount (Rs.)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    for bar, val in zip(bars, category_data.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                f'Rs.{val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

def create_weekly_chart(df):
    if df.empty or 'date' not in df.columns:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.isocalendar().week
    weekly = df.groupby('week')['amount'].sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weekly.index, weekly.values, marker='o', linewidth=2.5, 
            markersize=8, color='#2196F3', markerfacecolor='#FF5722')
    ax.fill_between(weekly.index, weekly.values, alpha=0.2, color='#2196F3')
    ax.set_xlabel('Week Number', fontsize=12)
    ax.set_ylabel('Total Expenses (Rs.)', fontsize=12)
    ax.set_title('Weekly Expense Trend', fontsize=14, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ─── Main App ──────────────────────────────────────────────────────────────────
def main():
    st.title("💰 Expense Detector AI Agent")
    st.markdown("### Your Smart Personal Finance Manager")
    st.markdown("---")

    # Load existing data
    all_expenses = load_data()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio("Go to", [
        "➕ Add Expenses",
        "📊 Dashboard",
        "📈 Charts & Reports",
        "🤖 AI Advice",
        "📅 Budget Planner"
    ])

    # ── ADD EXPENSES PAGE ─────────────────────────────────────────────────────
    if page == "➕ Add Expenses":
        st.header("➕ Add Your Expenses")
        
        tab1, tab2, tab3, tab4 = st.tabs(["✍️ Text Input", "📄 CSV File", "📋 PDF File", "🖼️ Image"])

        # Text Input Tab
        with tab1:
            st.subheader("Enter Expenses as Text")
            st.info("Format: Item Name: Amount  (one per line)\nExample:\nGroceries: 3000\nElectricity: 2500\nTransport: 1500")
            
            col1, col2 = st.columns(2)
            with col1:
                expense_date = st.date_input("📅 Date", datetime.today())
                period = st.selectbox("📆 Period", ["Daily", "Weekly", "Monthly"])
            with col2:
                month_name = st.text_input("📅 Month/Period Label", 
                                           value=datetime.today().strftime("%B %Y"))
            
            text_input = st.text_area("Enter your expenses (one per line):", 
                                       height=200,
                                       placeholder="Groceries: 3000\nElectricity: 2500\nRent: 15000\nTransport: 2000\nFood: 4000")
            
            if st.button("💾 Save Expenses", type="primary"):
                if text_input.strip():
                    parsed = parse_expenses_from_text(text_input)
                    for exp in parsed:
                        exp['date'] = str(expense_date)
                        exp['period'] = period
                        exp['month'] = month_name
                    all_expenses.extend(parsed)
                    save_data(all_expenses)
                    st.success(f"✅ {len(parsed)} expenses saved successfully!")
                    st.balloons()
                else:
                    st.warning("Please enter some expenses first!")

        # CSV File Tab
        with tab2:
            st.subheader("Upload CSV File")
            st.info("CSV should have columns: name, amount (and optionally: date, category)")
            
            col1, col2 = st.columns(2)
            with col1:
                csv_date = st.date_input("📅 Date", datetime.today(), key="csv_date")
                csv_period = st.selectbox("📆 Period", ["Daily", "Weekly", "Monthly"], key="csv_period")
            with col2:
                csv_month = st.text_input("📅 Month Label", 
                                          value=datetime.today().strftime("%B %Y"), 
                                          key="csv_month")
            
            uploaded_csv = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded_csv:
                try:
                    df = pd.read_csv(uploaded_csv)
                    st.write("Preview:", df.head())
                    if st.button("💾 Import CSV Data", type="primary"):
                        for _, row in df.iterrows():
                            exp = {
                                'name': str(row.get('name', row.get('item', row.iloc[0]))),
                                'amount': float(row.get('amount', row.get('cost', row.iloc[1] if len(row) > 1 else 0))),
                                'date': str(csv_date),
                                'period': csv_period,
                                'month': csv_month
                            }
                            all_expenses.append(exp)
                        save_data(all_expenses)
                        st.success(f"✅ {len(df)} expenses imported!")
                        st.balloons()
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        # PDF Tab
        with tab3:
            st.subheader("Upload PDF File")
            st.info("Upload a PDF bill or statement - text will be extracted automatically")
            
            col1, col2 = st.columns(2)
            with col1:
                pdf_date = st.date_input("📅 Date", datetime.today(), key="pdf_date")
                pdf_period = st.selectbox("📆 Period", ["Daily", "Weekly", "Monthly"], key="pdf_period")
            with col2:
                pdf_month = st.text_input("📅 Month Label",
                                          value=datetime.today().strftime("%B %Y"),
                                          key="pdf_month")
            
            uploaded_pdf = st.file_uploader("Upload PDF", type=['pdf'])
            if uploaded_pdf:
                try:
                    with pdfplumber.open(uploaded_pdf) as pdf:
                        text = ""
                        for page in pdf.pages:
                            text += page.extract_text() or ""
                    st.text_area("Extracted Text (Edit if needed):", text, height=200, key="pdf_text")
                    if st.button("💾 Parse & Save PDF Data", type="primary"):
                        parsed = parse_expenses_from_text(text)
                        for exp in parsed:
                            exp['date'] = str(pdf_date)
                            exp['period'] = pdf_period
                            exp['month'] = pdf_month
                        all_expenses.extend(parsed)
                        save_data(all_expenses)
                        st.success(f"✅ {len(parsed)} expenses extracted from PDF!")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")

        # Image Tab
        with tab4:
            st.subheader("Upload Image (Receipt/Bill)")
            st.info("Upload a photo of your receipt or bill")
            
            col1, col2 = st.columns(2)
            with col1:
                img_date = st.date_input("📅 Date", datetime.today(), key="img_date")
                img_period = st.selectbox("📆 Period", ["Daily", "Weekly", "Monthly"], key="img_period")
            with col2:
                img_month = st.text_input("📅 Month Label",
                                          value=datetime.today().strftime("%B %Y"),
                                          key="img_month")
            
            uploaded_img = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
            if uploaded_img:
                image = Image.open(uploaded_img)
                st.image(image, caption="Uploaded Receipt", use_column_width=True)
                st.info("💡 For image receipts, please manually enter the amounts below after viewing the image:")
                manual_text = st.text_area("Enter expenses from the image:", height=150,
                                           placeholder="Groceries: 2500\nMilk: 200\nBread: 150")
                if st.button("💾 Save Image Expenses", type="primary"):
                    if manual_text.strip():
                        parsed = parse_expenses_from_text(manual_text)
                        for exp in parsed:
                            exp['date'] = str(img_date)
                            exp['period'] = img_period
                            exp['month'] = img_month
                        all_expenses.extend(parsed)
                        save_data(all_expenses)
                        st.success(f"✅ {len(parsed)} expenses saved!")
                        st.balloons()

    # ── DASHBOARD PAGE ────────────────────────────────────────────────────────
    elif page == "📊 Dashboard":
        st.header("📊 Expense Dashboard")
        
        if not all_expenses:
            st.warning("No expenses recorded yet! Go to 'Add Expenses' to add some.")
            return
        
        df = pd.DataFrame(all_expenses)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        # Summary Cards
        total = df['amount'].sum()
        avg_daily = total / 30
        avg_weekly = total / 4
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Expenses", f"Rs. {total:,.0f}")
        with col2:
            st.metric("📅 Daily Average", f"Rs. {avg_daily:,.0f}")
        with col3:
            st.metric("📆 Weekly Average", f"Rs. {avg_weekly:,.0f}")
        with col4:
            st.metric("📋 Total Entries", len(df))
        
        st.markdown("---")
        
        # Expense Table
        st.subheader("📋 All Expenses")
        st.dataframe(df, use_container_width=True)
        
        # Delete option
        st.subheader("🗑️ Manage Data")
        if st.button("🗑️ Clear All Data", type="secondary"):
            save_data([])
            st.success("All data cleared!")
            st.rerun()
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button("📥 Download as CSV", csv, "expenses.csv", "text/csv")

    # ── CHARTS PAGE ───────────────────────────────────────────────────────────
    elif page == "📈 Charts & Reports":
        st.header("📈 Charts & Visual Reports")
        
        if not all_expenses:
            st.warning("No expenses recorded yet!")
            return
        
        df = pd.DataFrame(all_expenses)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df = df[df['amount'] > 0]
        
        # Filter by period
        if 'month' in df.columns:
            months = ['All'] + list(df['month'].unique())
            selected = st.selectbox("Filter by Month/Period:", months)
            if selected != 'All':
                df = df[df['month'] == selected]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥧 Expense Distribution")
            fig_pie = create_pie_chart(df)
            if fig_pie:
                st.pyplot(fig_pie)
                plt.close()
        
        with col2:
            st.subheader("📊 Expense by Category")
            fig_bar = create_bar_chart(df, "Expenses Breakdown")
            if fig_bar:
                st.pyplot(fig_bar)
                plt.close()
        
        st.subheader("📈 Weekly Trend")
        if 'date' in df.columns:
            fig_weekly = create_weekly_chart(df)
            if fig_weekly:
                st.pyplot(fig_weekly)
                plt.close()
        
        # Percentage breakdown
        st.subheader("📊 Percentage Breakdown")
        total = df['amount'].sum()
        pct_df = df.groupby('name')['amount'].sum().reset_index()
        pct_df['percentage'] = (pct_df['amount'] / total * 100).round(1)
        pct_df['amount_formatted'] = pct_df['amount'].apply(lambda x: f"Rs. {x:,.0f}")
        pct_df['percentage_str'] = pct_df['percentage'].apply(lambda x: f"{x}%")
        pct_df.columns = ['Expense Item', 'Amount (Rs)', 'Percentage', 'Amount', '% Share']
        st.dataframe(pct_df[['Expense Item', 'Amount', '% Share']], use_container_width=True)

    # ── AI ADVICE PAGE ────────────────────────────────────────────────────────
    elif page == "🤖 AI Advice":
        st.header("🤖 AI Financial Advisor")
        
        if not all_expenses:
            st.warning("Add some expenses first to get AI advice!")
            return
        
        df = pd.DataFrame(all_expenses)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        
        advice = get_ai_advice(df)
        st.markdown(advice)
        
        # Savings Calculator
        st.markdown("---")
        st.subheader("💡 Savings Calculator")
        col1, col2 = st.columns(2)
        with col1:
            income = st.number_input("Your Monthly Income (Rs):", min_value=0, value=50000, step=1000)
        with col2:
            total_exp = df['amount'].sum()
            st.metric("Total Expenses", f"Rs. {total_exp:,.0f}")
        
        savings = income - total_exp
        savings_pct = (savings / income * 100) if income > 0 else 0
        
        if savings >= 0:
            st.success(f"✅ You are saving Rs. {savings:,.0f} ({savings_pct:.1f}% of income)")
        else:
            st.error(f"⚠️ You are overspending by Rs. {abs(savings):,.0f}! Time to cut expenses.")

    # ── BUDGET PLANNER PAGE ───────────────────────────────────────────────────
    elif page == "📅 Budget Planner":
        st.header("📅 Budget Planner")
        
        st.subheader("Set Your Monthly Budget")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            monthly_income = st.number_input("Monthly Income (Rs):", min_value=0, value=50000, step=1000)
        with col2:
            savings_target = st.slider("Savings Target (%)", 0, 50, 20)
        with col3:
            budget = monthly_income * (1 - savings_target/100)
            st.metric("Available Budget", f"Rs. {budget:,.0f}")
        
        st.markdown("---")
        
        # Budget Breakdown (50/30/20 rule)
        st.subheader("💡 Recommended Budget Split (50/30/20 Rule)")
        needs = monthly_income * 0.50
        wants = monthly_income * 0.30
        savings_amt = monthly_income * 0.20
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Needs (50%)**\nRs. {needs:,.0f}\n\n(Rent, Food, Bills)")
        with col2:
            st.warning(f"**Wants (30%)**\nRs. {wants:,.0f}\n\n(Entertainment, Shopping)")
        with col3:
            st.success(f"**Savings (20%)**\nRs. {savings_amt:,.0f}\n\n(Emergency Fund, Investment)")
        
        # Compare with actual
        if all_expenses:
            df = pd.DataFrame(all_expenses)
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            actual = df['amount'].sum()
            
            st.markdown("---")
            st.subheader("📊 Budget vs Actual")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Recommended Budget', 'Actual Spending']
            values = [budget, actual]
            colors = ['#4CAF50', '#F44336' if actual > budget else '#2196F3']
            bars = ax.bar(categories, values, color=colors, width=0.4)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                        f'Rs. {val:,.0f}', ha='center', fontweight='bold')
            ax.set_ylabel("Amount (Rs.)")
            ax.set_title("Budget vs Actual Spending")
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rs.{x:,.0f}'))
            st.pyplot(fig)  
            plt.close()

# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

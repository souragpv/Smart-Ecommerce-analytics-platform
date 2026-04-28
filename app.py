import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Smart E-Commerce Analytics Platform")
st.markdown("---")


# ════════════════════════════════════════════════════════
# DATA LOADERS
# All use @st.cache_resource to avoid pickling MemoryError
# ════════════════════════════════════════════════════════

@st.cache_resource
def load_data():
    df = pd.read_csv('data/cleaned_retail.csv', parse_dates=['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['Month']      = df['InvoiceDate'].dt.to_period('M').astype(str)
    df['DayOfWeek']  = df['InvoiceDate'].dt.day_name()
    return df

@st.cache_resource
def load_rfm():
    return pd.read_csv('data/rfm_segments.csv')

@st.cache_resource
def load_weekly():
    return pd.read_csv('data/weekly_sales.csv')

@st.cache_resource
def load_popular():
    return pd.read_csv('data/popular_products.csv')

@st.cache_resource
def load_lstm():
    """
    Fix: compile=False avoids the keras.metrics.mse
    deserialization error on load.
    """
    from tensorflow.keras.models import load_model
    try:
        # Try .keras format first (TF 2.12+)
        model = load_model('models/lstm_model.keras', compile=False)
    except Exception:
        # Fall back to .h5 format
        model = load_model('models/lstm_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mse')
    scaler = pickle.load(open('models/lstm_scaler.pkl', 'rb'))
    return model, scaler

@st.cache_resource
def build_matrix():
    """
    Fix: Use @st.cache_resource (not @st.cache_data) to avoid
    pickling MemoryError on large matrices.
    Also reduces to top 500 customers x top 500 products
    to keep memory usage low.
    """
    df_tmp = pd.read_csv('data/cleaned_retail.csv')
    df_tmp['CustomerID'] = df_tmp['CustomerID'].astype(int)

    # Keep top 500 most active customers
    top_custs = (
        df_tmp.groupby('CustomerID')['Invoice']
        .nunique()
        .sort_values(ascending=False)
        .head(500)
        .index
    )

    # Keep top 500 best-selling products
    top_prods = (
        df_tmp.groupby('StockCode')['Quantity']
        .sum()
        .sort_values(ascending=False)
        .head(500)
        .index
    )

    df_tmp = df_tmp[
        df_tmp['CustomerID'].isin(top_custs) &
        df_tmp['StockCode'].isin(top_prods)
    ]

    # Build Customer x Product matrix
    cp = (
        df_tmp.groupby(['CustomerID', 'StockCode'])['Quantity']
        .sum()
        .unstack(fill_value=0)
    )
    cp.index.name   = None
    cp.columns.name = None

    ids = cp.index.tolist()
    sim = cosine_similarity(cp.values)

    return cp, sim, ids


# ── Load everything ──
df               = load_data()
rfm              = load_rfm()
weekly_sales     = load_weekly()
popular_products = load_popular()

lstm_model, lstm_scaler = load_lstm()
customer_product, sim_matrix, customer_ids = build_matrix()

product_map = (
    df[['StockCode', 'Description']]
    .drop_duplicates(subset=['StockCode'])
    .set_index('StockCode')['Description']
)


# ════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════

st.sidebar.title("📌 Navigation")
section = st.sidebar.radio("Go to", [
    "📊 Business Overview",
    "👥 Customer Segments",
    "📈 Sales Forecast",
    "🎯 Recommendations"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** UCI Online Retail II")
st.sidebar.markdown("**Churn Model:** Random Forest")
st.sidebar.markdown("**Forecast:** LSTM (Weekly)")
st.sidebar.markdown("**Recommender:** Cosine Similarity")


# ════════════════════════════════════════════════════════
# SECTION 1 — BUSINESS OVERVIEW
# ════════════════════════════════════════════════════════

if section == "📊 Business Overview":

    st.header("📊 Business Overview")

    # ── KPI Cards ──
    total_revenue   = df['TotalAmount'].sum()
    total_orders    = df['Invoice'].nunique()
    total_customers = df['CustomerID'].nunique()
    total_products  = df['StockCode'].nunique()
    avg_order_value = df.groupby('Invoice')['TotalAmount'].sum().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Revenue",   f"£{total_revenue:,.0f}")
    c2.metric("🧾 Total Orders",    f"{total_orders:,}")
    c3.metric("👥 Customers",       f"{total_customers:,}")
    c4.metric("📦 Products",        f"{total_products:,}")
    c5.metric("🛒 Avg Order Value", f"£{avg_order_value:,.2f}")

    st.markdown("---")

    # ── Monthly Revenue ──
    st.subheader("Monthly Revenue Trend")
    monthly = df.groupby('Month')['TotalAmount'].sum()

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(monthly.index, monthly.values,
            marker='o', color='steelblue', linewidth=2, markersize=4)
    ax.fill_between(monthly.index, monthly.values,
                    alpha=0.1, color='steelblue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue (GBP)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}')
    )
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Top Products & Countries ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Products by Revenue")
        top_prod = (
            df.groupby('Description')['TotalAmount']
            .sum().sort_values(ascending=False).head(10)
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top_prod.index[::-1], top_prod.values[::-1],
                color='coral', edgecolor='white')
        ax.set_xlabel('Revenue (GBP)')
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}')
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Top 10 Countries by Revenue")
        top_country = (
            df.groupby('Country')['TotalAmount']
            .sum().sort_values(ascending=False).head(10)
        )
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top_country.index[::-1], top_country.values[::-1],
                color='mediumpurple', edgecolor='white')
        ax.set_xlabel('Revenue (GBP)')
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}')
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Sales by Day ──
    st.subheader("Revenue by Day of Week")
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    sales_day = df.groupby('DayOfWeek')['TotalAmount'].sum().reindex(day_order)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sales_day.index, sales_day.values,
           color='mediumseagreen', edgecolor='white')
    ax.set_ylabel('Revenue (GBP)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'£{x/1e6:.1f}M')
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════
# SECTION 2 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════

elif section == "👥 Customer Segments":

    st.header("👥 Customer Segmentation & Churn")

    # ── KPI Cards ──
    total_customers = len(rfm)
    churned         = rfm['is_churned'].sum()
    churn_rate      = churned / total_customers * 100
    champions       = (rfm['Segment'] == 'Champions').sum()
    at_risk         = (rfm['Segment'] == 'At Risk').sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers", f"{total_customers:,}")
    c2.metric("✅ Active",          f"{total_customers - churned:,}")
    c3.metric("🚨 Churned",         f"{churned:,}")
    c4.metric("📉 Churn Rate",      f"{churn_rate:.1f}%")
    c5.metric("🏆 Champions",       f"{champions:,}")

    st.markdown("---")

    # ── Segment Bar & Churn Pie ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Segments")
        seg_counts = rfm['Segment'].value_counts()
        colors     = ['#6CB8D4','#E07B6B','#79C99E','#F5C166']

        fig, ax = plt.subplots(figsize=(6, 5))
        bars = ax.bar(seg_counts.index, seg_counts.values,
                      color=colors, edgecolor='white')
        ax.set_ylabel('Number of Customers')
        for bar, v in zip(bars, seg_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 5, str(v), ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn Distribution")
        churn_counts = rfm['is_churned'].value_counts()

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(churn_counts.values,
               labels=['Active','Churned'],
               autopct='%1.1f%%',
               colors=['#79C99E','#E07B6B'],
               startangle=90,
               wedgeprops={'edgecolor':'white','linewidth':2})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Segment Summary Table ──
    st.subheader("Segment Summary Table")
    summary = rfm.groupby('Segment').agg(
        Customers     = ('CustomerID',  'count'),
        Avg_Recency   = ('Recency',    'mean'),
        Avg_Frequency = ('Frequency',  'mean'),
        Avg_Monetary  = ('Monetary',   'mean'),
        Churn_Rate    = ('is_churned', 'mean')
    ).round(2)
    summary['Churn_Rate'] = (
        (summary['Churn_Rate'] * 100).round(1).astype(str) + '%'
    )
    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    # ── Scatter Plot ──
    st.subheader("Recency vs Monetary by Segment")
    palette = {
        'Champions'      : '#6CB8D4',
        'At Risk'        : '#E07B6B',
        'New Customers'  : '#79C99E',
        'Loyal Customers': '#F5C166'
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    for seg, grp in rfm.groupby('Segment'):
        ax.scatter(grp['Recency'], grp['Monetary'],
                   label=seg, alpha=0.5, s=15,
                   color=palette.get(seg, 'gray'))
    ax.set_xlabel('Recency (days since last purchase)')
    ax.set_ylabel('Monetary Value (GBP)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Customer Lookup ──
    st.subheader("🔍 Customer Lookup")
    cid = st.number_input("Enter Customer ID", min_value=0, step=1)
    if st.button("Search"):
        result = rfm[rfm['CustomerID'] == int(cid)]
        if result.empty:
            st.warning(f"Customer {int(cid)} not found.")
        else:
            r = result.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Segment",   r['Segment'])
            c2.metric("Recency",   f"{int(r['Recency'])} days")
            c3.metric("Frequency", f"{int(r['Frequency'])} orders")
            c4.metric("Monetary",  f"£{r['Monetary']:,.2f}")
            status = '🚨 Churned' if r['is_churned'] == 1 else '✅ Active'
            st.info(f"Churn Status: {status}")


# ════════════════════════════════════════════════════════
# SECTION 3 — SALES FORECAST
# ════════════════════════════════════════════════════════

elif section == "📈 Sales Forecast":

    st.header("📈 Sales Forecasting — LSTM Model")

    # ── KPI Cards ──
    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Total Weeks",         f"{len(weekly_sales)}")
    c2.metric("💰 Avg Weekly Revenue",  f"£{weekly_sales['Revenue'].mean():,.0f}")
    c3.metric("🔝 Peak Weekly Revenue", f"£{weekly_sales['Revenue'].max():,.0f}")

    st.markdown("---")

    # ── Historical Chart ──
    st.subheader("Historical Weekly Revenue")
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(weekly_sales['Week'], weekly_sales['Revenue'],
            color='steelblue', linewidth=1.5, marker='o', markersize=3)
    ax.fill_between(weekly_sales.index, weekly_sales['Revenue'],
                    alpha=0.1, color='steelblue')
    ax.set_xlabel('Week')
    ax.set_ylabel('Revenue (GBP)')
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}')
    )
    step = max(1, len(weekly_sales) // 12)
    plt.xticks(
        ticks=weekly_sales.index[::step],
        labels=weekly_sales['Week'].iloc[::step],
        rotation=45, ha='right'
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Forecast Controls ──
    st.subheader("🔮 Forecast Future Revenue")
    n_weeks = st.slider("Weeks to forecast", min_value=1, max_value=12, value=4)

    if st.button("Generate Forecast"):
        with st.spinner("Running LSTM model..."):
            try:
                window_size    = 4
                revenue_values = weekly_sales[['Revenue']].values
                revenue_scaled = lstm_scaler.transform(revenue_values)
                last_seq       = revenue_scaled[-window_size:].reshape(1, window_size, 1)

                future_preds = []
                for _ in range(n_weeks):
                    val = lstm_model.predict(last_seq, verbose=0)[0][0]
                    future_preds.append(float(val))
                    last_seq = np.append(
                        last_seq[:, 1:, :], [[[val]]], axis=1
                    )

                future_actual = lstm_scaler.inverse_transform(
                    np.array(future_preds).reshape(-1, 1)
                ).flatten()

                # ── Metric Cards ──
                st.success("Forecast complete!")
                cols = st.columns(min(n_weeks, 4))
                for i, val in enumerate(future_actual):
                    cols[i % 4].metric(f"Week +{i+1}", f"£{val:,.0f}")

                # ── Forecast Chart ──
                last_n = weekly_sales['Revenue'].values[-8:]
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(range(len(last_n)), last_n,
                        label='Actual (last 8 weeks)',
                        color='steelblue', linewidth=2)
                ax.plot(
                    range(len(last_n), len(last_n) + n_weeks),
                    future_actual,
                    label=f'Forecast (next {n_weeks} weeks)',
                    color='coral', linewidth=2,
                    linestyle='--', marker='o'
                )
                ax.axvline(x=len(last_n) - 1, color='gray',
                           linestyle=':', label='Forecast start')
                ax.set_xlabel('Week')
                ax.set_ylabel('Revenue (GBP)')
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}')
                )
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── Forecast Table ──
                st.subheader("Forecast Table")
                forecast_df = pd.DataFrame({
                    'Week'                   : [f'Week +{i+1}' for i in range(n_weeks)],
                    'Forecasted Revenue (£)' : [f'£{v:,.2f}' for v in future_actual]
                })
                st.dataframe(forecast_df, use_container_width=True)

            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.info("Make sure lstm_model.h5 and lstm_scaler.pkl are in the models/ folder.")


# ════════════════════════════════════════════════════════
# SECTION 4 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════

elif section == "🎯 Recommendations":

    st.header("🎯 Product Recommendation System")

    # ── Recommendation Function ──
    def recommend_products(customer_id, n=5):
        customer_id = int(customer_id)

        # Fallback to popular products if customer not in matrix
        if customer_id not in customer_ids:
            return None

        idx        = customer_ids.index(customer_id)
        sim_scores = pd.Series(sim_matrix[idx], index=customer_ids)
        sim_scores = sim_scores.drop(index=customer_id, errors='ignore')

        top_similar = (
            sim_scores.sort_values(ascending=False).head(5).index.tolist()
        )

        already_bought = set(
            customer_product.iloc[idx][
                customer_product.iloc[idx] > 0
            ].index.tolist()
        )

        scores = customer_product.loc[top_similar].sum()
        scores = scores.drop(index=list(already_bought), errors='ignore')
        scores = scores.sort_values(ascending=False).head(n)

        result                = scores.reset_index()
        result.columns        = ['StockCode', 'Score']
        result['Description'] = result['StockCode'].map(product_map)
        return result

    # ── Popular Products Chart ──
    st.subheader("🔥 Most Popular Products (Overall)")
    top10 = popular_products.head(10)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(top10['Description'][::-1],
            top10['PurchaseCount'][::-1],
            color='mediumpurple', edgecolor='white')
    ax.set_xlabel('Number of Orders')
    ax.set_title('Top 10 Popular Products')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Personalised Recommendations ──
    st.subheader("🎯 Personalised Recommendations")
    st.caption(f"Recommendations available for {len(customer_ids):,} top customers.")

    col1, col2 = st.columns([2, 1])
    with col1:
        cid    = st.number_input("Enter Customer ID", min_value=0, step=1)
    with col2:
        n_recs = st.selectbox("Number of recommendations", [3, 5, 10], index=1)

    if st.button("Get Recommendations"):
        with st.spinner("Finding recommendations..."):
            recs = recommend_products(int(cid), n=n_recs)

        if recs is None:
            st.warning(
                f"Customer {int(cid)} not in top customers list. "
                "Showing most popular products instead."
            )
            recs = popular_products[['StockCode','Description','PurchaseCount']].head(n_recs).copy()
            recs.columns = ['StockCode','Description','Score']

        if recs.empty:
            st.info("No new recommendations found for this customer.")
        else:
            st.success(f"Top {len(recs)} recommendations for Customer {int(cid)}:")

            cols = st.columns(min(len(recs), 3))
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style='
                        background:#f0f4f8;
                        border-radius:10px;
                        padding:14px;
                        margin-bottom:10px;
                        border-left:4px solid #6CB8D4;
                    '>
                        <b>#{i+1} {row['Description']}</b><br>
                        <small style='color:gray'>StockCode: {row['StockCode']}</small>
                    </div>
                    """, unsafe_allow_html=True)

            st.dataframe(recs.reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # ── Purchase History ──
    st.subheader("📋 Customer Purchase History")
    hid = st.number_input("Customer ID for history", min_value=0, step=1, key="hist")

    if st.button("Show History"):
        history = df[df['CustomerID'] == int(hid)]
        if history.empty:
            st.warning(f"No purchase history found for Customer {int(hid)}.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Orders",    history['Invoice'].nunique())
            c2.metric("Total Spent",     f"£{history['TotalAmount'].sum():,.2f}")
            c3.metric("Unique Products", history['StockCode'].nunique())

            st.markdown("**Last 10 Purchases:**")
            recent = (
                history[['InvoiceDate','Description','Quantity','TotalAmount']]
                .sort_values('InvoiceDate', ascending=False)
                .head(10)
                .reset_index(drop=True)
            )
            st.dataframe(recent, use_container_width=True)

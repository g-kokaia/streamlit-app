import streamlit as st
import pandas as pd
import plotly.express as px

# st.set_page_config(page_title="Betting Dashboard", layout="wide")

# Cache DataFrame loading
@st.cache_data
def load_data():
    # df = pd.read_csv('C:\\General_workspace/data/user_profiling/scores_df.csv')
    # df = pd.read_parquet('C:\\General_workspace/data/user_profiling/scores_df.parquet')
    df = pd.read_parquet('scores_df.parquet')
    return df

# Load data
df = load_data()

st.caption(f'Updated: :red[**{df.RunDate.max().date()}**]')

brand_list = sorted(df['BrandName'].unique().tolist())
selected_brand = st.selectbox("Select Brand", brand_list)

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['BrandName'] == selected_brand]


st.caption(f'Number of Unique Clients: :red[**{filtered_df.index.nunique():,}**]')

tab1, tab2= st.tabs(['**Filters**', '**Clients**'])

with tab1:

    st.subheader("Score")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        selected_1 = st.select_slider("Amount Score", options=range(11), value=(0, 10))
    with col2:
        selected_2 = st.select_slider("Frequency Score", options=range(11), value=(0, 10))
    with col3:
        selected_3 = st.select_slider("Progit Score", options=range(11), value=(0, 10))
    with col4:
        selected_4 = st.select_slider("Diversity Score", options=range(11), value=(0, 10))
    with col5:
        selected_5 = st.select_slider("Loyal Score", options=range(11), value=(0, 10))

    # Apply filtering based on min and max values
    filtered_df = filtered_df[
        (filtered_df['AMOUNT_SCORE'] >= selected_1[0]) & (filtered_df['AMOUNT_SCORE'] <= selected_1[1]) &
        (filtered_df['FREQ_SCORE'] >= selected_2[0]) & (filtered_df['FREQ_SCORE'] <= selected_2[1]) &
        (filtered_df['PROFIT_SCORE'] >= selected_3[0]) & (filtered_df['PROFIT_SCORE'] <= selected_3[1]) &
        (filtered_df['DIVERSITY_SCORE'] >= selected_4[0]) & (filtered_df['DIVERSITY_SCORE'] <= selected_4[1]) &
        (filtered_df['LOYAL_SCORE'] >= selected_5[0]) & (filtered_df['LOYAL_SCORE'] <= selected_5[1])
    ]


    # Info cards
    col1, col2, col3 = st.columns(3)

    with col1:
        unique_clients = filtered_df.index.nunique()
        st.metric(":red[Unique Clients]", f"{unique_clients:,}", border=True)
    with col2:
        total_bets = filtered_df['BetAmount_System'].sum()
        st.metric(":red[Total Bet Amount]", f"{total_bets:,.0f}", border=True)
    with col3:
        total_ggr = filtered_df['GGR'].sum()
        st.metric(":red[Total GGR]", f"{total_ggr:,.0f}", border=True)

    st.dataframe(filtered_df.drop(['RunDate', 'BrandName'], axis=1), height=200)

    # Histogram section
    st.subheader("Score Distributions")

    score_columns = ['AMOUNT_SCORE', 'FREQ_SCORE', 'PROFIT_SCORE', 'DIVERSITY_SCORE', 'LOYAL_SCORE']

    # Buttons for histogram selection
    selected_score = st.radio(
        "Select Score for Histogram",
        score_columns,
        horizontal=False,
    )

    # Create histogram
    fig = px.histogram(
        filtered_df,
        x=selected_score,
        nbins=50,
        # title=f"Distribution of {selected_score}",
        labels={selected_score: selected_score.replace('_', ' ').title()},
        template="plotly_white",
        color_discrete_sequence=['grey']
    )

    fig.update_layout(
        xaxis_title=selected_score.replace('_', ' ').title(),
        yaxis_title="Count",
        bargap=0.1
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:

    st.subheader('Client Profile')

    selected_user = st.text_input("", placeholder="Insert client ID", value='0000')

    if selected_user:
        user_df = filtered_df[filtered_df.index == int(selected_user)]
    else:
        st.write("Please enter Client ID")


    if not user_df.empty:

        score_columns = ['AMOUNT_SCORE', 'FREQ_SCORE', 'PROFIT_SCORE', 'DIVERSITY_SCORE', 'LOYAL_SCORE']
        
        cols = st.columns(5)
        for idx, col_name in enumerate(score_columns):
            score = user_df[col_name].iloc[0]
            green = min(int(255 * (score / 10)), 255)
            red = min(int(255 * (1 - score / 10)), 255)
            color = f"rgb({red},{green},60)"
            
            with cols[idx]:
                st.markdown(
                    f'<div class="score-box" style="background-color:{color}"><h4>{col_name.replace("_", " ").title()}</h4><h2>{score:.2f}</h2></div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("No data available for the selected User Profile ID.")

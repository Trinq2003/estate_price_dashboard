import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
from plotly.subplots import make_subplots

# Load Data
@st.cache
def load_data():
    df = pd.read_csv('./data/cleaned_data.csv')
    df = df.drop_duplicates()
    return df

data = load_data()

# Configure Sidebar
st.sidebar.title("Navigation")
pages = ["Page 1: House Layout Analysis", "Page 2: House Direction Analysis", "Page 3: Legits Analysis", "Page 4: Geographical Analysis", "Page 5: Temporal Analysis"]
selected_page = st.sidebar.radio("Go to:", pages)

if selected_page == "Page 1: House Layout Analysis":
    st.title("House Layout Analysis")

    with st.expander("1.1: Distribution Plots"):
        # Distribution of Bedrooms
        st.header("Distribution of Bedrooms")
        q_low_bedrooms = data['Bedrooms'].quantile(0.01)
        q_high_bedrooms = data['Bedrooms'].quantile(0.99)

        fig1 = px.histogram(data, x='Bedrooms', histnorm='density', title="Distribution of Bedrooms",
                            category_orders={'Bedrooms': sorted(data['Bedrooms'].unique())},
                            color_discrete_sequence=['steelblue'])
        fig1.update_layout(
            xaxis_title="Number of Bedrooms", 
            yaxis_title="Density",
            xaxis_range=[q_low_bedrooms, q_high_bedrooms],
            legend_title="Bedrooms"
        )
        st.plotly_chart(fig1)

        # Distribution of Toilets
        st.header("Distribution of Toilets")
        q_low_toilets = data['Toilets'].quantile(0.01)
        q_high_toilets = data['Toilets'].quantile(0.99)

        fig2 = px.histogram(data, x='Toilets', histnorm='density', title="Distribution of Toilets",
                            category_orders={'Toilets': sorted(data['Toilets'].unique())},
                            color_discrete_sequence=['darkorange'])
        fig2.update_layout(
            xaxis_title="Number of Toilets", 
            yaxis_title="Density",
            xaxis_range=[q_low_toilets, q_high_toilets],
            legend_title="Toilets"
        )
        st.plotly_chart(fig2)

        # Distribution of Floors
        st.header("Distribution of Floors (Zoomed)")
        q_low_floors = data['Floors'].quantile(0.01)
        q_high_floors = data['Floors'].quantile(0.99)

        fig3 = px.histogram(data, x='Floors', title="Distribution of Floors",
                            color_discrete_sequence=['steelblue'])
        fig3.update_layout(
            xaxis_title="Number of Floors", 
            yaxis_title="Frequency",
            xaxis_range=[q_low_floors, q_high_floors]
        )
        st.plotly_chart(fig3)
        
    with st.expander("1.2: Correlation Plots"):
        # Scatter Plot of Bedrooms vs Toilets
        st.header("Scatter Plot of Bedrooms vs Toilets")
        bedrooms_and_toilets_filtered_df = data[(data['Bedrooms'] <= 150) & (data['Toilets'] <= 150)]

        fig4 = px.scatter(bedrooms_and_toilets_filtered_df, x='Bedrooms', y='Toilets',
                          title="Scatter Plot of Bedrooms vs Toilets",
                          color_discrete_sequence=['#FCA902'])
        st.plotly_chart(fig4)

        # 2D Histogram
        st.header("2D Histogram of Bedrooms vs Toilets")
        fig5 = go.Figure(data=go.Histogram2d(
            x=bedrooms_and_toilets_filtered_df['Bedrooms'],
            y=bedrooms_and_toilets_filtered_df['Toilets'],
            colorscale='Viridis',
            nbinsx=60, nbinsy=60
        ))
        fig5.update_layout(
            title="2D Histogram of Bedrooms vs Toilets",
            xaxis_title="Number of Bedrooms",
            yaxis_title="Number of Toilets"
        )
        st.plotly_chart(fig5)

        # Interactive 3D Scatter Plot
        st.header("3D Scatter Plot of Log-Transformed Bedrooms, Toilets, and Floors")
        bedrooms_and_toilets_filtered_df['Log_Bedrooms'] = np.log1p(bedrooms_and_toilets_filtered_df['Bedrooms'])
        bedrooms_and_toilets_filtered_df['Log_Toilets'] = np.log1p(bedrooms_and_toilets_filtered_df['Toilets'])
        bedrooms_and_toilets_filtered_df['Log_Floors'] = np.log1p(bedrooms_and_toilets_filtered_df['Floors'])

        fig6 = px.scatter_3d(bedrooms_and_toilets_filtered_df, 
                             x='Log_Bedrooms', y='Log_Toilets', z='Log_Floors',
                             color='Log_Floors', title="3D Scatter Plot",
                             color_continuous_scale='Viridis')
        st.plotly_chart(fig6)

        # KDE Plot for Bedrooms/Floor and Toilets/Floor
        st.header("Distribution of Bedrooms/Floor and Toilets/Floor")
        bedrooms_and_toilets_filtered_df['Bedrooms_per_floor'] = bedrooms_and_toilets_filtered_df['Bedrooms'] / bedrooms_and_toilets_filtered_df['Floors']
        bedrooms_and_toilets_filtered_df['Toilets_per_floor'] = bedrooms_and_toilets_filtered_df['Toilets'] / bedrooms_and_toilets_filtered_df['Floors']

        fig7 = px.histogram(bedrooms_and_toilets_filtered_df, 
                            x='Bedrooms_per_floor', 
                            title="Distribution of Bedrooms per Floor", 
                            color_discrete_sequence=['steelblue'],
                            nbins=50)
        fig7.add_vline(x=bedrooms_and_toilets_filtered_df['Bedrooms_per_floor'].mean(), 
                       line_dash="dash", line_color="blue", 
                       annotation_text="Mean")
        st.plotly_chart(fig7)

        fig8 = px.histogram(bedrooms_and_toilets_filtered_df, 
                            x='Toilets_per_floor', 
                            title="Distribution of Toilets per Floor", 
                            color_discrete_sequence=['darkorange'],
                            nbins=50)
        fig8.add_vline(x=bedrooms_and_toilets_filtered_df['Toilets_per_floor'].mean(), 
                       line_dash="dash", line_color="red", 
                       annotation_text="Mean")
        st.plotly_chart(fig8)

if selected_page == "Page 2: House Direction Analysis":
    st.title("House Direction Analysis")
    house_dir_balcony_dir_filtered_df = data[data['House Direction'].notnull() & data['Balcony Direction'].notnull()]

    with st.expander("2.1: Alignment Analysis"):

        direction_angles = {
            'đông': 0,
            'đông - nam': 45,
            'nam': 90,
            'tây - nam': 135,
            'tây': 180,
            'tây - bắc': 225,
            'bắc': 270,
            'đông - bắc': 315
        }

        def angle_difference(house, balcony):
            angle_house = direction_angles.get(house, None)
            angle_balcony = direction_angles.get(balcony, None)
            if angle_house is not None and angle_balcony is not None:
                diff = abs(angle_house - angle_balcony)
                return min(diff, 360 - diff)
            return None

        house_dir_balcony_dir_filtered_df['Angle'] = house_dir_balcony_dir_filtered_df.apply(
            lambda row: angle_difference(row['House Direction'], row['Balcony Direction']), axis=1
        )

        # -------- Interactive Histogram --------
        st.header("Distribution of House Direction and Balcony Direction")
        fig1 = px.histogram(
            house_dir_balcony_dir_filtered_df,
            x="House Direction",
            color_discrete_sequence=["steelblue"],
            nbins=len(direction_angles),
            title="Distribution of House Direction"
        )
        fig2 = px.histogram(
            house_dir_balcony_dir_filtered_df,
            x="Balcony Direction",
            color_discrete_sequence=["darkorange"],
            nbins=len(direction_angles),
            title="Distribution of Balcony Direction"
        )
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)

        # -------- Interactive Radar Plot --------
        st.header("Relationship Between House Direction and Balcony Direction (Radar Plot)")
        angle_labels = ['0° (Same)', '45°', '90° (Perpendicular)', '135°', '180° (Opposite)']
        angle_values = [0, 45, 90, 135, 180]
        counts = [house_dir_balcony_dir_filtered_df['Angle'].value_counts().get(val, 0) for val in angle_values]

        radar_fig = go.Figure()
        radar_fig.add_trace(go.Scatterpolar(
            r=counts + [counts[0]],
            theta=angle_labels + [angle_labels[0]],
            fill='toself',
            name='Angle Differences',
            line=dict(color='blue')
        ))

        radar_fig.update_layout(
            polar=dict(
                angularaxis=dict(tickmode='array', tickvals=angle_values, ticktext=angle_labels),
                radialaxis=dict(visible=True)
            ),
            title="Radar Plot of Angle Differences"
        )

        st.plotly_chart(radar_fig)

    with st.expander("2.2: Direction Analysis"):
        # Filter for Same and Opposite Directions
        same_direction = house_dir_balcony_dir_filtered_df[house_dir_balcony_dir_filtered_df['Angle'] == 0].copy()
        opposite_direction = house_dir_balcony_dir_filtered_df[house_dir_balcony_dir_filtered_df['Angle'] == 180].copy()

        same_direction['Category'] = 'Same Direction'
        opposite_direction['Category'] = 'Opposite Direction'
        combined_df = pd.concat([same_direction, opposite_direction])

        # -------- Interactive Count Plot --------
        st.header("Distribution of House Directions for Same and Opposite Categories")
        fig3 = px.histogram(
            combined_df,
            x="House Direction",
            color="Category",
            barmode="group",
            title="Distribution of House Directions",
            color_discrete_map={"Same Direction": "steelblue", "Opposite Direction": "darkorange"}
        )
        fig3.update_layout(
            xaxis_title="House Direction",
            yaxis_title="Count",
            legend_title="Direction Relationship",
            xaxis=dict(tickmode="array", tickvals=list(direction_angles.keys()), ticktext=list(direction_angles.keys()))
        )
        st.plotly_chart(fig3)
        
    with st.expander("2.3: Price and Direction Correlation"):
        # Filter the data for relevant angles
        angles_of_interest = [0, 45, 90, 135, 180]
        filtered_data = house_dir_balcony_dir_filtered_df[house_dir_balcony_dir_filtered_df['Angle'].isin(angles_of_interest)].copy()

        # Apply quantile filtering to focus on the important part
        q_low = filtered_data['Price'].quantile(0.01)  # 1st percentile
        q_high = filtered_data['Price'].quantile(0.99)  # 99th percentile
        filtered_data = filtered_data[(filtered_data['Price'] >= q_low) & (filtered_data['Price'] <= q_high)]

        # Scale 'Price' by dividing by 1000
        filtered_data['Scaled_Price'] = filtered_data['Price'] / 1000

        # -------- Interactive Violin and Box Plot --------
        st.header("Price Distribution by Direction Angles")
        fig4 = px.violin(
            filtered_data,
            x='Angle',
            y='Scaled_Price',
            box=True,
            points='all',
            color='Angle',
            title="Violin Plot of Scaled Price with Box Appearance",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # Customize data point size and box attributes
        fig4.update_traces(
            marker=dict(size=2),  # Adjust marker size
            box=dict(line_width=1, line_color='white', fillcolor="rgba(255, 255, 255, 0.7)", width=0.2)  # Customize box appearance
        )

        fig4.update_layout(
            xaxis_title="Angle Between House and Balcony Direction (Degrees)",
            yaxis_title="Price (in Thousands)",
            legend_title="Angles"
        )
        st.plotly_chart(fig4)

if selected_page == "Page 3: Legits Analysis":
    st.title("Legits Analysis")
    legits_area_filtered_df = data.dropna(subset=['Legits', 'Area'])  # Drop rows with missing values in relevant columns
    
    with st.expander("3.1: Legits Distribution"):
        # Filter the data for Legits and Area
        legits_area_filtered_df['Legits'] = legits_area_filtered_df['Legits'].astype(str)  # Ensure Legits is a string

        # -------- Interactive Histogram --------
        st.header("Distribution of Legits")
        fig5 = px.histogram(
            legits_area_filtered_df,
            x='Legits',
            title="Distribution of Legits",
            color_discrete_sequence=['steelblue']
        )

        # Add annotations for bar heights
        fig5.update_traces(
            texttemplate='%{y}',
            textposition='outside'
        )

        fig5.update_layout(
            xaxis_title="Number of Legits",
            yaxis_title="Frequency",
            legend_title="",
            xaxis=dict(tickangle=90),
            bargap=0.1  # Adjust bar gap
        )

        st.plotly_chart(fig5)

    with st.expander("3.2: Correlation Plot"):
        # Apply quantile filtering for Area and Price
        q_low_area = legits_area_filtered_df['Area'].quantile(0.01)
        q_high_area = legits_area_filtered_df['Area'].quantile(0.99)

        q_low_price = legits_area_filtered_df['Price'].quantile(0.1)
        q_high_price = legits_area_filtered_df['Price'].quantile(0.9)

        area_filtered_df = legits_area_filtered_df[(legits_area_filtered_df['Area'] >= q_low_area) & 
                                                   (legits_area_filtered_df['Area'] <= q_high_area)]
        price_filtered_df = legits_area_filtered_df[(legits_area_filtered_df['Price'] >= q_low_price) & 
                                                    (legits_area_filtered_df['Price'] <= q_high_price)]

        price_filtered_df['Price/1000'] = price_filtered_df['Price'] / 1000  # Scale Price by dividing by 1000

        # -------- First Violin and Box Plot: Area --------
        st.header("Distribution of Area for Each Legits Category")
        fig6 = px.violin(
            area_filtered_df,
            x='Legits',
            y='Area',
            box=True,
            points='all',
            color='Legits',
            title="Violin Plot of Area with Box Appearance",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # Customize box attributes for Area plot
        fig6.update_traces(
            marker=dict(size=1),  # Adjust marker size
            box=dict(line_width=1, line_color='white', fillcolor="rgba(255, 255, 255, 0.7)", width=0.2)  # Customize box appearance
        )

        fig6.update_layout(
            yaxis_title="Area (m²)",
            xaxis_title="Legits Categories",
            legend_title="Legits"
        )
        st.plotly_chart(fig6)

        # -------- Second Violin and Box Plot: Price --------
        st.header("Distribution of Price (in billion VND) for Each Legits Category")
        fig7 = px.violin(
            price_filtered_df,
            x='Legits',
            y='Price/1000',
            box=True,
            points='all',
            color='Legits',
            title="Violin Plot of Price with Box Appearance",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # Customize box attributes for Price plot
        fig7.update_traces(
            marker=dict(size=1),  # Adjust marker size
            box=dict(line_width=1, line_color='white', fillcolor="rgba(255, 255, 255, 0.7)", width=0.2)  # Customize box appearance
        )

        fig7.update_layout(
            yaxis_title="Price (in Billion VND)",
            xaxis_title="Legits Categories",
            legend_title="Legits"
        )
        st.plotly_chart(fig7)

    with st.expander("3.3: Multi-Correlation Plots"):
        # Log-transform Price and Area
        legits_area_filtered_df['Log_Price'] = np.log1p(legits_area_filtered_df['Price'])
        legits_area_filtered_df['Log_Area'] = np.log1p(legits_area_filtered_df['Area'])

        # Apply quantile filtering for Log_Area and Log_Price
        q_low_log_area = legits_area_filtered_df['Log_Area'].quantile(0.01)
        q_high_log_area = legits_area_filtered_df['Log_Area'].quantile(0.99)
        q_low_log_price = legits_area_filtered_df['Log_Price'].quantile(0.01)
        q_high_log_price = legits_area_filtered_df['Log_Price'].quantile(0.99)

        filtered_df = legits_area_filtered_df[
            (legits_area_filtered_df['Log_Area'] >= q_low_log_area) &
            (legits_area_filtered_df['Log_Area'] <= q_high_log_area) &
            (legits_area_filtered_df['Log_Price'] >= q_low_log_price) &
            (legits_area_filtered_df['Log_Price'] <= q_high_log_price)
        ]

        subset_red_pink = filtered_df[filtered_df['Legits'].isin(['+đỏ+hồng', '+đỏ', '+hồng'])]
        subset_none = filtered_df[filtered_df['Legits'] == 'không có']

        palette = {
            '+đỏ': 'orange',
            '+hồng': 'green',
            'không có': 'red',
            '+đỏ+hồng': 'blue',
            '+hđmb': 'purple'
        }

        # -------- Scatter Plot for All Legits Categories --------
        st.header("Scatter Plot of Log-Transformed Price vs Area by Legits Categories")
        fig8 = px.scatter(
            filtered_df,
            x="Log_Price",
            y="Log_Area",
            color="Legits",
            color_discrete_map=palette,
            title="Scatter Plot of Log(Price + 1) vs Log(Area + 1) by Legits Categories",
            opacity=0.5
        )

        fig8.update_traces(marker=dict(size=2, line=dict(width=0.5, color="white")))
        fig8.update_layout(
            xaxis_title="Log(Price + 1)",
            yaxis_title="Log(Area + 1)",
            legend_title="Legits Categories"
        )
        st.plotly_chart(fig8)

        # -------- Scatter Plot for '+đỏ+hồng' Category --------
        st.header("Scatter Plot for '+đỏ+hồng'")
        fig9 = px.scatter(
            subset_red_pink,
            x="Log_Price",
            y="Log_Area",
            title="Scatter Plot for '+đỏ+hồng'",
            color_discrete_sequence=["blue"],
            opacity=0.5
        )

        fig9.update_traces(marker=dict(size=2, line=dict(width=0.5, color="white")))
        fig9.update_layout(
            xaxis_title="Log(Price + 1)",
            yaxis_title="Log(Area + 1)"
        )
        st.plotly_chart(fig9)

        # -------- Scatter Plot for 'không có' Category --------
        st.header("Scatter Plot for 'không có'")
        fig10 = px.scatter(
            subset_none,
            x="Log_Price",
            y="Log_Area",
            title="Scatter Plot for 'không có'",
            color_discrete_sequence=["red"],
            opacity=0.5
        )

        fig10.update_traces(marker=dict(size=2, line=dict(width=0.5, color="white")))
        fig10.update_layout(
            xaxis_title="Log(Price + 1)",
            yaxis_title="Log(Area + 1)"
        )
        st.plotly_chart(fig10)
        
if selected_page == "Page 4: Geographical Analysis":    
    st.title("Geographical Analysis")
    geo_df = data.copy()
    
    with st.expander("4.1: District House Distribution"):
        # Distribution of Districts
        st.header("Distribution of Districts")
        fig11 = px.histogram(
            geo_df,
            x='District',
            title="Distribution of District",
            color_discrete_sequence=['steelblue']
        )

        # Customize histogram appearance
        fig11.update_traces(
            marker=dict(line=dict(width=0.1, color="black"))
        )

        fig11.update_layout(
            xaxis_title="District",
            yaxis_title="Frequency",
            xaxis=dict(tickangle=90),
            bargap=0.1  # Adjust bar gap
        )

        st.plotly_chart(fig11)

    with st.expander("4.2: Price Distribution per District"):
        # Filter the data for Price and District
        price_district_filtered_df = geo_df.dropna(subset=['Price', 'District'])
        q_low_price = price_district_filtered_df['Price'].quantile(0.01)
        q_high_price = price_district_filtered_df['Price'].quantile(0.99)

        price_district_filtered_df = price_district_filtered_df[
            (price_district_filtered_df['Price'] >= q_low_price) &
            (price_district_filtered_df['Price'] <= q_high_price)
        ]

        district_counts = price_district_filtered_df['District'].value_counts()
        valid_districts = district_counts[district_counts > 1000].index

        price_district_filtered_df = price_district_filtered_df[price_district_filtered_df['District'].isin(valid_districts)]
        price_district_filtered_df['District'] = pd.Categorical(price_district_filtered_df['District'], categories=valid_districts, ordered=True)
        price_district_filtered_df['Price/1000'] = price_district_filtered_df['Price'] / 1000

        # -------- Interactive Violin and Box Plot --------
        st.header("Price Distribution per District (Ordered by Record Count)")
        fig12 = px.violin(
            price_district_filtered_df,
            x='District',
            y='Price/1000',
            box=True,
            points='all',
            color='District',
            title="Violin Plot of Price Distribution per District",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        # Customize box and violin attributes
        fig12.update_traces(
            box=dict(line_width=1, line_color='white', fillcolor="rgba(255, 255, 255, 0.7)", width=0.2),
            marker=dict(size=2, line=dict(width=0.1, color="black"))
        )

        # Add annotations for district counts
        for i, district in enumerate(valid_districts):
            count = district_counts[district]
            fig12.add_annotation(
                x=i,
                y=price_district_filtered_df['Price/1000'].max() * 1.1,
                text=f"{count}",
                showarrow=False,
                font=dict(size=10, color="black", family="Arial")
            )

        fig12.update_layout(
            xaxis_title="District (Ordered by Count)",
            yaxis_title="Price (Billion VND)",
            xaxis=dict(tickangle=90),
            legend_title="Districts"
        )

        st.plotly_chart(fig12)

        # Area Distribution
        area_district_filtered_df = geo_df.dropna(subset=['Area', 'District'])
        q_low_area = area_district_filtered_df['Area'].quantile(0.01)
        q_high_area = area_district_filtered_df['Area'].quantile(0.99)

        area_district_filtered_df = area_district_filtered_df[
            (area_district_filtered_df['Area'] >= q_low_area) &
            (area_district_filtered_df['Area'] <= q_high_area)
        ]

        district_counts = area_district_filtered_df['District'].value_counts()
        valid_districts = district_counts[district_counts > 1000].index

        area_district_filtered_df = area_district_filtered_df[area_district_filtered_df['District'].isin(valid_districts)]
        area_district_filtered_df['District'] = pd.Categorical(area_district_filtered_df['District'], categories=valid_districts, ordered=True)

        st.header("Area Distribution per District (Ordered by Record Count)")
        fig13 = px.violin(
            area_district_filtered_df,
            x='District',
            y='Area',
            box=True,
            points='all',
            color='District',
            title="Violin Plot of Area Distribution per District",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig13.update_traces(
            box=dict(line_width=1, line_color='white', fillcolor="rgba(255, 255, 255, 0.7)", width=0.2),
            marker=dict(size=2, line=dict(width=0.1, color="black"))
        )

        for i, district in enumerate(valid_districts):
            count = district_counts[district]
            fig13.add_annotation(
                x=i,
                y=area_district_filtered_df['Area'].max() * 1.1,
                text=f"{count}",
                showarrow=False,
                font=dict(size=10, color="black", family="Arial")
            )

        fig13.update_layout(
            xaxis_title="District (Ordered by Count)",
            yaxis_title="Area (m²)",
            xaxis=dict(tickangle=90),
            legend_title="Districts"
        )

        st.plotly_chart(fig13)

    # with st.expander("4.3: Map"):
    #     # Filter the data for X, Y, and Price
    #     xy_price_filtered_df = geo_df.dropna(subset=['X', 'Y', 'Price'])
    #     q_low_price = xy_price_filtered_df['Price'].quantile(0.01)
    #     q_high_price = xy_price_filtered_df['Price'].quantile(0.99)
    #     xy_price_filtered_df = xy_price_filtered_df[(xy_price_filtered_df['Price'] >= q_low_price) & (xy_price_filtered_df['Price'] <= q_high_price)]

    #     xy_price_filtered_df['Price/1000'] = xy_price_filtered_df['Price'] / 1000

    #     # Define the color map for the feature values
    #     colormap = cm.LinearColormap(
    #         colors=['green', 'yellow', 'red'],  # Gradient from green to red
    #         vmin=min(xy_price_filtered_df['Price/1000']), 
    #         vmax=max(xy_price_filtered_df['Price/1000'])
    #     )

    #     # Initialize the map centered around the mean of X and Y
    #     center = [xy_price_filtered_df['X'].mean(), xy_price_filtered_df['Y'].mean()]
    #     m = folium.Map(location=center, zoom_start=14)

    #     # Plot each point in the DataFrame onto the map
    #     for i in range(len(xy_price_filtered_df)):
    #         folium.Circle(
    #             location=[xy_price_filtered_df.iloc[i]['X'], xy_price_filtered_df.iloc[i]['Y']],  # Coordinates
    #             radius=20,  # Radius size
    #             fill=True,
    #             color=colormap(xy_price_filtered_df.iloc[i]['Price/1000']),  # Color based on Price
    #             fill_opacity=0.5,  # Transparency level
    #             popup=f"Price: {xy_price_filtered_df.iloc[i]['Price/1000']:,}"  # Popup with price information
    #         ).add_to(m)

    #     # Add the color map legend to the map
    #     m.add_child(colormap)

    #     # Display the map
    #     st.header("Geographical Distribution of Price")
    #     st_folium(m, width=700, height=500)

if selected_page == "Page 5: Temporal Analysis":    
    st.title("Temporal Analysis")
    temporal_df = data.copy()
    temporal_df.dropna(subset=['Published Date', 'Expired Date'], inplace=True)
    temporal_df['Published Date'] = pd.to_datetime(temporal_df['Published Date'], format="%d/%m/%Y")

    with st.expander("5.1: Daily Plot"):
        q_low = temporal_df['Published Date'].quantile(0.05)
        q_high = temporal_df['Published Date'].quantile(0.95)
        temporal_df = temporal_df[(temporal_df['Published Date'] >= q_low) & (temporal_df['Published Date'] <= q_high)]

        aggregated_df = temporal_df.groupby('Published Date').agg(
            Min_Price=('Price', 'min'),
            Max_Price=('Price', 'max'),
            Avg_Price=('Price', 'mean')
        ).reset_index()

        aggregated_df['Log_Min_Price'] = np.log1p(aggregated_df['Min_Price'])
        aggregated_df['Log_Max_Price'] = np.log1p(aggregated_df['Max_Price'])
        aggregated_df['Log_Avg_Price'] = np.log1p(aggregated_df['Avg_Price'])
        aggregated_df['Smoothed_Price'] = aggregated_df['Log_Avg_Price'].rolling(window=10).mean()

        # -------- Interactive Daily Plot --------
        st.header("Daily Log_Price Range (Min-Max) and Average Log_Price Trend")
        fig14 = px.line(
            aggregated_df,
            x='Published Date',
            y=['Log_Min_Price', 'Log_Max_Price', 'Log_Avg_Price', 'Smoothed_Price'],
            labels={
                'Published Date': 'Published Date',
                'value': 'Log(Price + 1)',
                'variable': 'Metric'
            },
            title="Daily Log_Price Range and Trends"
        )

        fig14.update_traces(mode='lines+markers', line=dict(width=2), marker=dict(size=3))
        fig14.update_layout(
            legend_title="Metrics",
            xaxis_title="Published Date",
            yaxis_title="Log(Price + 1)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig14)

    with st.expander("5.2: District Trend Plot"):
        # Prepare data for district trends
        selected_districts = ['cầu giấy', 'đống đa', 'hà đông', 'thanh xuân', 'long biên', 'hoàng mai']
        district_temporal_df = temporal_df[temporal_df['District'].isin(selected_districts)]
        district_temporal_df['Log_Price'] = np.log1p(district_temporal_df['Price'])

        # -------- Interactive District Trends --------
        st.header("Daily Log_Price Trends with Min-Max Area at Selected Districts")
        fig15 = make_subplots(rows=2, cols=3, shared_yaxes=True, shared_xaxes=True, subplot_titles=[district.capitalize() for district in selected_districts])

        for i, district in enumerate(selected_districts):
            district_df = district_temporal_df[district_temporal_df['District'] == district]
            aggregated_df = district_df.groupby('Published Date').agg(
                Min_Log_Price=('Log_Price', 'min'),
                Max_Log_Price=('Log_Price', 'max'),
                Avg_Log_Price=('Log_Price', 'mean')
            ).reset_index()

            row, col = divmod(i, 3)
            row += 1
            col += 1

            fig15.add_traces([
                go.Scatter(
                    x=aggregated_df['Published Date'],
                    y=aggregated_df['Avg_Log_Price'],
                    mode='lines+markers',
                    name=f"{district.capitalize()} Avg Price",
                    line=dict(color="steelblue"),
                    marker=dict(size=3),
                    showlegend=False
                ),
                go.Scatter(
                    x=aggregated_df['Published Date'],
                    y=aggregated_df['Min_Log_Price'],
                    mode='lines',
                    fill='tonexty',
                    name=f"{district.capitalize()} Min-Max Range",
                    line=dict(color="skyblue"),
                    marker=dict(size=3),
                    fillcolor="rgba(135,206,235,0.4)",
                    showlegend=False
                )
            ], rows=row, cols=col)

        fig15.update_layout(
            title_text="Daily Log_Price Trends at Selected Districts",
            height=800,
            legend_title="Metrics",
            xaxis_title="Published Date",
            yaxis_title="Log(Price + 1)",
            showlegend=True
        )

        st.plotly_chart(fig15)
















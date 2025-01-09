import pandas as pd
import numpy as np
import altair as alt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA

alt.data_transformers.disable_max_rows()
df = pd.read_csv('data/accidents_preprocessed.csv')


##################################### Selections #####################################

# Interactive dropdown options
options_month = ['June', 'July', 'August', 'September']
input_dropdown_month = alt.binding_select(
    options=[None] + options_month, labels=['All'] + options_month, name='Month: '
)
selection_month = alt.param(name='SelectMonth', value=None, bind=input_dropdown_month)

selection_vehicle = alt.selection_point(fields=['VEHICLE TYPE'], name="SelectVehicle", empty="all")
selection_borough = alt.selection_point(fields=['BOROUGH'], name="SelectBorough", empty="all")
selection_weather = alt.selection_point(fields=['WEATHER'], name="SelectWeather", empty="all")
selection_point = alt.selection_point(fields=['HOUR'], name='SelectPoint', empty='all')



##################################### PCA #####################################

# Variables
selected_columns = ['BOROUGH', 'ZIP CODE', 'LATITUDE', 'LONGITUDE', 
                    'CONTRIBUTING FACTOR', 'VEHICLE TYPE', 'MONTH', 
                    'HOUR', 'WEEK_DAY', 'DAY', 'WEATHER', 
                    'TOTAL_INJURIES', 'TOTAL_DEATHS']

categorical_cols = ['BOROUGH', 'CONTRIBUTING FACTOR', 'VEHICLE TYPE', 'MONTH', 'WEEK_DAY', 'WEATHER']

df_pca = df[selected_columns].dropna().copy()

df_pca['OriginalIndex'] = df_pca.index

for c in categorical_cols:
    df_pca[c] = df_pca[c].astype(str)

# Ordinal encoding for categorical variables
encoder = OrdinalEncoder()
df_pca[categorical_cols] = encoder.fit_transform(df_pca[categorical_cols])

# Variables scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca.drop(columns=['OriginalIndex']))

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

df_pca['PC1'] = X_pca[:, 0]
df_pca['PC2'] = X_pca[:, 1]

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df_pca['cluster'] = kmeans.fit_predict(X_pca[:, :2])

# Restore original dataset for visualization
df_pca_coords = df_pca[['OriginalIndex', 'PC1', 'PC2', 'cluster']]
df = df.merge(
    df_pca_coords,
    how='left',
    left_index=True,
    right_on='OriginalIndex'
)
df.drop(columns=['OriginalIndex'], inplace=True)

# Principal components loadings
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
feature_names = df_pca.drop(columns=['OriginalIndex','PC1','PC2','cluster']).columns
loading_df = pd.DataFrame(
    loadings,
    index=feature_names,
    columns=[f'PC{i}' for i in range(1, len(pca.components_)+1)]
)

loading_2d = loading_df[['PC1', 'PC2']].reset_index()
loading_2d.columns = ['Variable', 'PC1', 'PC2']
arrow_scale = 2.0
loading_2d['x_end'] = loading_2d['PC1'] * arrow_scale
loading_2d['y_end'] = loading_2d['PC2'] * arrow_scale

line_data = []
for i, row in loading_2d.iterrows():
    line_data.append({
        'Variable': row['Variable'], 'x': 0, 'y': 0
    })
    line_data.append({
        'Variable': row['Variable'], 'x': row['x_end'], 'y': row['y_end']
    })
line_df = pd.DataFrame(line_data)


# scatterplot points for principal components 1 and 2
points = alt.Chart(df).mark_circle(size=40).encode(
    x=alt.X('PC1:Q', title='PC1'),
    y=alt.Y('PC2:Q', title='PC2'),
    color=alt.condition(selection_vehicle & selection_borough & selection_weather & selection_point, 
                        'cluster:N', 
                        alt.value('lightgray')),
).add_params(
    selection_weather, selection_month, selection_point, selection_vehicle, selection_borough
).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)
).properties(
    width=450,
    height=300,
    title='Biplot PCA with Clusters'
)

# Arrows
lines = alt.Chart(line_df).mark_line(color='black').encode(
    x='x:Q',
    y='y:Q',
    detail='Variable:N'
)

arrow_heads = alt.Chart(loading_2d).mark_point(color='black').encode(
    x='x_end:Q',
    y='y_end:Q',
    tooltip=['Variable']
)

text = alt.Chart(loading_2d).mark_text(
    align='left',
    dx=5,
    dy=-5,
    color='black',
    size = 8
).encode(
    x='x_end:Q',
    y='y_end:Q',
    text='Variable'
)

biplot = (points + lines + arrow_heads + text)



##################################### Accidents Map #####################################

df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')

df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])

# Reach through the URL for the GeoJSON stored in GitHub account
raw_geojson_url = 'https://raw.githubusercontent.com/AlbaGrc/Traffic_Accidents_NYC/main/NYC_map.geojson'

ny_city_map = alt.Data(
    url=raw_geojson_url,
    format=alt.DataFormat(property='features')
)

# Base map
nyc_base_map = alt.Chart(ny_city_map).mark_geoshape(
    fill='lightgray', stroke='white', strokeWidth=1.3, opacity=0.4
).encode(tooltip=alt.value(None))

# Map points classified by severity of the accident (red - death, orange - injury, yellow - no damage)
death_points = alt.Chart(df[df['SEVERITY'] == 'Death']).mark_circle(size=9, opacity=0.8).encode(
    longitude='LONGITUDE:Q',
    latitude='LATITUDE:Q',
    color=alt.Color(
        'SEVERITY:N',
        scale=alt.Scale(
            domain=['Death', 'Injury', 'No Damage'],
            range=['red', 'orange', '#FFEB3B']
        ),
        legend=alt.Legend(title='Severity')
    ),
    tooltip=['DATETIME:N','BOROUGH:N','ZIP CODE:N','TOTAL_DEATHS:Q','TOTAL_INJURIES:Q'],
    opacity=alt.condition(
        selection_borough & selection_vehicle & selection_weather & selection_point,
        alt.value(1), alt.value(0.1)
    )
).add_params(
    selection_month
).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)
)

injury_points = alt.Chart(df[df['SEVERITY'] == 'Injury']).mark_circle(size=7, opacity=0.6).encode(
    longitude='LONGITUDE:Q',
    latitude='LATITUDE:Q',
    color=alt.Color(
        'SEVERITY:N',
        scale=alt.Scale(
            domain=['Death', 'Injury', 'No Damage'],
            range=['red', 'orange', '#FFEB3B']
        ),
        legend=alt.Legend(title='Severity')
    ),
    tooltip=['DATETIME:N','BOROUGH:N','ZIP CODE:N','TOTAL_DEATHS:Q','TOTAL_INJURIES:Q'],
    opacity=alt.condition(
        selection_borough & selection_vehicle & selection_weather & selection_point,
        alt.value(1), alt.value(0.1)
    )
).add_params(
    selection_month
).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)
)

no_damage_points = alt.Chart(df[df['SEVERITY'] == 'No Damage']).mark_circle(size=5, opacity=0.6).encode(
    longitude='LONGITUDE:Q',
    latitude='LATITUDE:Q',
    color=alt.Color(
        'SEVERITY:N',
        scale=alt.Scale(
            domain=['Death', 'Injury', 'No Damage'],
            range=['red', 'orange', '#FFEB3B']
        ),
        legend=alt.Legend(title='Severity')
    ),
    tooltip=['DATETIME:N','BOROUGH:N','ZIP CODE:N','TOTAL_DEATHS:Q','TOTAL_INJURIES:Q'],
    opacity=alt.condition(
        selection_borough & selection_vehicle & selection_weather & selection_point,
        alt.value(1), alt.value(0.1)
    )
).add_params(
    selection_month
).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)
)

# Final map
final_map = (
    nyc_base_map + no_damage_points + injury_points + death_points
).properties(
    width=350,
    height=350
).interactive().add_params(
    selection_month,
    selection_borough,
    selection_vehicle,
    selection_weather,
    selection_point
)

# Bar chart with accumulation of incidents per borough
bar_chart = alt.Chart(df).mark_bar(size=35).encode(
    x=alt.X('count():Q', title='Number of Accidents'),
    y=alt.Y('BOROUGH:N', sort='-x', title='Borough'),
    color=alt.condition(
        selection_borough,
        alt.value('steelblue'),
        alt.value('lightgray')
    ),
    opacity=alt.condition(
        selection_borough,
        alt.value(1),
        alt.value(0.3)
    ),
    tooltip=['BOROUGH:N', 'count():Q']
).add_params(
    selection_month,
    selection_borough
).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)
).properties(
    width=300,
    height=300
)



##################################### Hours Line Chart #####################################

# Line chart with hours on axis X and accidents count on axis Y
hours_line_chart = alt.Chart(df).mark_line(point=True).encode(
    x=alt.X('HOUR:O',
            title='Hour of the day',
            axis=alt.Axis(labelAngle=0)),
    y=alt.Y('count():Q',
            title='Accidents count'),
    color=alt.condition(selection_point, alt.value('steelblue'), alt.value('lightgray')),
    tooltip=['HOUR:O', 'count():Q']
).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)
).add_params(
    selection_month, selection_point
).properties(
    width=450,
    height=300
)


##################################### Heatmap Chart #####################################

df['DAY'] = pd.to_numeric(df['DAY'], errors='coerce')
df['MONTH'] = pd.Categorical(df['MONTH'], categories=['June', 'July', 'August', 'September'], ordered=True)

base = alt.Chart(df).encode(
    x=alt.X('DAY:O', title='Day', scale=alt.Scale(domain=np.arange(1, 32))),
    y=alt.Y('MONTH:N', title='Month', scale=alt.Scale(domain=options_month)),
    tooltip=['DAY:O', 'MONTH:N', 'count():Q']  # Tooltip
).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)  # Filter month
).add_params(
    selection_month
)

heatmap = base.mark_rect().encode(
    color=alt.Color('count():Q', scale=alt.Scale(scheme='orangered'), title='Accidents Number', legend=None)
)

# Labels with count for each day
labels = base.mark_text(baseline='middle', fontSize=9).encode(
    text=alt.Text('count():Q'),
    color=alt.value('black')
)

heatmap_chart = (heatmap + labels).properties(
    width=850,  
    height=140    
)



##################################### Weather Bar Chart #####################################

# Map emojis for a faster and easier user understanding
weather_icon_to_emoji = {
    'rain': '🌧️',
    'clear-day': '☀️',
    'partly-cloudy-day': '⛅',
    'cloudy': '☁️'
}

df['WEATHER_EMOJI'] = df['WEATHER'].map(weather_icon_to_emoji)

base = alt.Chart(df).transform_filter(
    (alt.datum.MONTH == selection_month) | (selection_month == None)
).transform_aggregate(
    count='count()',
    groupby=[
        'MONTH', 'WEEK_DAY', 'BOROUGH', 'VEHICLE TYPE', 'HOUR',
        'WEATHER', 'DAY', 'SEVERITY', 'WEATHER_EMOJI'
    ]
).encode(
    x=alt.X('WEATHER:N', title='Weather condition', sort='-y', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('sum(count):Q', title='Accidents count'),
    color=alt.Color(
        'SEVERITY:N',
        scale=alt.Scale(
            domain=['No Damage', 'Injury', 'Death'],
            range=['#FFEB3B', 'orange', 'red']
        ),
        legend=None
    ),
    tooltip=['WEATHER:N', 'SEVERITY:N', 'sum(count):Q']
)

bars = base.mark_bar().encode(
    opacity=alt.condition(selection_weather, alt.value(1), alt.value(0.6))
)

emojis = base.mark_text(
    align='center', baseline='bottom', dy=-40, size=20
).transform_filter(
    alt.datum.SEVERITY == 'No Damage'  # Solo un emoji por barra
).encode(
    text='WEATHER_EMOJI:N'
)

# Combine bar chart with emojis
weather_chart = alt.layer(
    bars, emojis
).add_params(
    selection_month, selection_weather
).properties(
    width=280,
    height=300
)



##################################### Vehicle Type #####################################

# Histogram with X as accidents count and Y as contributing factor
histogram = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        y=alt.Y('CONTRIBUTING FACTOR:N', sort='-x', axis=alt.Axis(title=None)),
        x=alt.X('count()', title='Accidents count'),
        color=alt.Color(
            'VEHICLE TYPE:N', 
            title='Vehicle Type',
            scale=alt.Scale(scheme='tableau10')),
        tooltip=['VEHICLE TYPE', 'count()']
    )
    .transform_filter(
        (alt.datum.MONTH == selection_month) | (selection_month == None)
    )
    .add_params(selection_month)
    .add_params(selection_vehicle)
    .encode(
        opacity=alt.condition(selection_vehicle, alt.value(1), alt.value(0.3))
    )
    .properties(
        title='Contributing factors',
        width=400,
        height=300
    )
)


##################################### Dashboard #####################################

# Interactions
bar_chart = bar_chart.transform_filter(selection_vehicle & selection_borough & selection_weather & selection_point)
histogram = histogram.transform_filter(selection_vehicle & selection_borough & selection_weather & selection_point)
weather_chart = weather_chart.transform_filter(selection_vehicle & selection_borough & selection_weather & selection_point)
hours_line_chart = hours_line_chart.transform_filter(selection_vehicle & selection_borough  & selection_weather & selection_point)
heatmap_chart = heatmap_chart.transform_filter(selection_vehicle & selection_borough & selection_weather & selection_point)

# Combine the charts into the final layout
final_dashboard = alt.vconcat(
    alt.hconcat(final_map, bar_chart, biplot, spacing=10).resolve_scale(color='independent'),
    alt.hconcat(histogram, weather_chart, hours_line_chart, spacing=10).resolve_scale(color='independent'),
    alt.hconcat(heatmap_chart, spacing=10).resolve_scale(color='independent'),
).resolve_scale(
    color='independent',
    opacity='independent'
)
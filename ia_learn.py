Import packages
import os
import glob
import logging
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(dataset_directory):
    """
    Load and combine all CSV files from the specified directory.

    Parameters:
    - dataset_directory (str): The path to the directory containing CSV files.

    Returns:
    - pd.DataFrame: A DataFrame containing the combined data from all CSV files.
    """
    dataframes = []
    for file_path in glob.glob(os.path.join(dataset_directory, '*.csv')):
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespaces from column names
            dataframes.append(df)
        except pd.errors.EmptyDataError as e:
            logging.error(f"Empty file {file_path}: {e}")
        except pd.errors.ParserError as e:
            logging.error(f"Parsing error in {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
    if dataframes:
        df_combined = pd.concat(dataframes, ignore_index=True)
        return df_combined
    else:
        logging.error("No data was loaded.")
        return pd.DataFrame()

def process_data(df):
    """
    Preprocess the data, including type conversions and creation of additional columns.

    Parameters:
    - df (pd.DataFrame): The combined DataFrame to preprocess.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame.
    """
    # Remove leading/trailing whitespaces and standardize to lowercase in text columns
    text_columns = ['School_Name', 'Server_Responsible', 'Option_1', 'Option_2', 'Option_3', 'Option_4', 'City_District']
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str).str.strip().str.lower()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)  # Remove multiple spaces

    # Convert 'Visit_DateTime' to datetime
    df['Visit_DateTime'] = pd.to_datetime(df['Visit_DateTime'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Student_Count'] = pd.to_numeric(df['Student_Count'], errors='coerce')
    df['Min_Age'] = pd.to_numeric(df['Min_Age'], errors='coerce')
    df['Max_Age'] = pd.to_numeric(df['Max_Age'], errors='coerce')

    # Create 'Age_Group' column
    df['Age_Group'] = df['Min_Age'].astype(str) + ' - ' + df['Max_Age'].astype(str)

    # Classify ratings for NPS
    df['NPS_Category'] = pd.cut(df['Rating'], bins=[0, 2, 3, 5], labels=['Detractors', 'Passives', 'Promoters'], right=True)

    # Extract date and hour if 'Visit_DateTime' was successfully converted
    if pd.api.types.is_datetime64_any_dtype(df['Visit_DateTime']):
        df['Date'] = df['Visit_DateTime'].dt.date
        df['Hour'] = df['Visit_DateTime'].dt.hour
    else:
        logging.warning("'Visit_DateTime' could not be converted to datetime.")

    return df

def calculate_nps(df):
    """
    Calculate the overall NPS and NPS by district.

    Parameters:
    - df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
    - float: The overall NPS score.
    - pd.DataFrame: DataFrame containing NPS scores by district.
    """
    # Overall NPS
    nps_counts = df['NPS_Category'].value_counts()
    total_responses = nps_counts.sum()
    promoters = nps_counts.get('Promoters', 0)
    detractors = nps_counts.get('Detractors', 0)

    if total_responses == 0:
        nps_score = 0
    else:
        nps_score = ((promoters - detractors) / total_responses) * 100

    # NPS by district
    df_nps = df.groupby('City_District')['NPS_Category'].value_counts().unstack(fill_value=0)
    df_nps['Total'] = df_nps.sum(axis=1)
    df_nps['Promoters'] = df_nps.get('Promoters', 0)
    df_nps['Detractors'] = df_nps.get('Detractors', 0)
    df_nps['NPS'] = ((df_nps['Promoters'] - df_nps['Detractors']) / df_nps['Total']) * 100

    return nps_score, df_nps.reset_index()

def get_option_counts(df):
    """
    Count the occurrences of options selected by visitors.

    Parameters:
    - df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
    - pd.DataFrame: DataFrame containing counts of options.
    """
    options = df[['Option_1', 'Option_2', 'Option_3', 'Option_4']].apply(lambda x: x.str.strip())
    options = options.values.flatten()
    options = pd.Series(options)
    option_counts = options.value_counts().dropna()
    return option_counts.reset_index().rename(columns={'index': 'Option_Text', 0: 'count'})

def create_figures(df):
    """
    Create figures for visualization in the dashboard.

    Parameters:
    - df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
    - dict: Dictionary containing Plotly figures.
    """
    figures = {}

    # Calculate NPS
    nps_score, df_nps = calculate_nps(df)
    figures['nps_score'] = nps_score
    figures['df_nps'] = df_nps if not df_nps.empty else pd.DataFrame({'City_District': [], 'NPS': []})

    # Count of Selected Options
    df_option_counts = get_option_counts(df)
    if not df_option_counts.empty:
        figures['option_counts'] = px.bar(
            df_option_counts, x='Option_Text', y='count',
            title='Ocorrências das opções selecionadas', labels={'count': 'Count', 'Option_Text': 'Option'}
        )
    else:
        figures['option_counts'] = None

    # Count of City Districts
    df_city_district_counts = df['City_District'].value_counts().reset_index()
    df_city_district_counts.columns = ['City_District', 'count']
    if not df_city_district_counts.empty:
        figures['city_district_counts'] = px.bar(
            df_city_district_counts, x='City_District', y='count',
            title='Ocorrências por Distritos', labels={'count': 'Count', 'City_District': 'District'}
        )
    else:
        figures['city_district_counts'] = None

    # Student Count by School
    df_student_count = df.groupby('School_Name', as_index=False)['Student_Count'].sum()
    if not df_student_count.empty:
        figures['student_count'] = px.bar(
            df_student_count, x='School_Name', y='Student_Count',
            title='Total de alunos por escola', labels={'Student_Count': 'Number of Students', 'School_Name': 'School'}
        )
    else:
        figures['student_count'] = None

    # Visits by District
    df_district_visits = df.groupby('City_District', as_index=False)['Student_Count'].sum()
    if not df_district_visits.empty:
        figures['district_visits'] = px.bar(
            df_district_visits, x='City_District', y='Student_Count',
            title='Total de visitas por Distrito', labels={'Student_Count': 'Number of Visitors', 'City_District': 'District'}
        )
    else:
        figures['district_visits'] = None

    # Average Rating by District
    df_avg_rating = df.groupby('City_District', as_index=False)['Rating'].mean()
    if not df_avg_rating.empty:
        figures['avg_rating'] = px.bar(
            df_avg_rating, x='City_District', y='Rating',
            title='Avaliação média por distrito', labels={'Rating': 'Average Rating', 'City_District': 'District'}
        )
    else:
        figures['avg_rating'] = None

    # Age Distribution
    df_age_distribution = df.groupby('Age_Group', as_index=False)['Student_Count'].sum()
    if not df_age_distribution.empty:
        figures['age_distribution'] = px.bar(
            df_age_distribution, x='Age_Group', y='Student_Count',
            title='Distribuição de visitantes por faixa etária', labels={'Student_Count': 'Number of Visitors', 'Age_Group': 'Age Group'}
        )
    else:
        figures['age_distribution'] = None

    # Visits by Date
    if 'Date' in df.columns:
        df_visits_by_day = df.groupby('Date', as_index=False)['Student_Count'].sum()
        if not df_visits_by_day.empty:
            figures['visits_by_day'] = px.line(
                df_visits_by_day, x='Date', y='Student_Count',
                title='Visitas por data', labels={'Student_Count': 'Number of Visitors', 'Date': 'Date'}
            )
        else:
            figures['visits_by_day'] = None
    else:
        figures['visits_by_day'] = None

    # Visits by Hour
    if 'Hour' in df.columns:
        df_visits_by_hour = df.groupby('Hour', as_index=False)['Student_Count'].sum()
        if not df_visits_by_hour.empty:
            figures['visits_by_hour'] = px.line(
                df_visits_by_hour, x='Hour', y='Student_Count',
                title='Visitas por hora', labels={'Student_Count': 'Number of Visitors', 'Hour': 'Hour of Day'}
            )
        else:
            figures['visits_by_hour'] = None
    else:
        figures['visits_by_hour'] = None

    # NPS Breakdown
    nps_counts = df['NPS_Category'].value_counts().reset_index()
    nps_counts.columns = ['Group', 'count']
    if not nps_counts.empty:
        figures['nps_breakdown'] = px.pie(
            nps_counts, names='Group', values='count',
            title='Análise de Promotores, Passivos e Detratores'
        )
    else:
        figures['nps_breakdown'] = None

    # Correlation between Options and Rating
    df_options_melted = df.melt(
        id_vars=['Rating'],
        value_vars=['Option_1', 'Option_2', 'Option_3', 'Option_4'],
        var_name='Option',
        value_name='Option_Text'
    )
    df_options_melted = df_options_melted.dropna(subset=['Option_Text'])

    if not df_options_melted.empty:
        df_option_rating_corr = df_options_melted.groupby('Option_Text')['Rating'].mean().reset_index()
        if not df_option_rating_corr.empty:
            corr_matrix = df_option_rating_corr.set_index('Option_Text')
            if not corr_matrix.empty:
                figures['heatmap'] = px.imshow(
                    corr_matrix,
                    aspect='auto',
                    color_continuous_scale='Viridis',
                    labels={'color': 'Average Rating'},
                    title=''
                )
            else:
                figures['heatmap'] = None
        else:
            figures['heatmap'] = None
    else:
        figures['heatmap'] = None

    return figures

custom_styles = {
    'primary': '#0288d1',  # Main blue color for headers and text
    'background': '#ffffff',  # White background
    'text': '#333333',  # Dark text color for contrast
    'card_colors': ['#1565c0', '#1e88e5', '#42a5f5'],  # Different shades of blue for the cards
    'font_blue': '#007cba'
}


def build_dashboard(df):
    """
    Build the Dash dashboard layout with interactive filters.

    Parameters:
    - df (pd.DataFrame): The preprocessed DataFrame.

    Returns:
    - Dash: The Dash application instance.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Lists for filters
    school_names = df['School_Name'].dropna().unique().tolist()
    city_districts = df['City_District'].dropna().unique().tolist()
    min_date = df['Visit_DateTime'].min().date() if not df['Visit_DateTime'].isnull().all() else None
    max_date = df['Visit_DateTime'].max().date() if not df['Visit_DateTime'].isnull().all() else None
    ratings = sorted(df['Rating'].dropna().unique())

    # Main layout with filters
    app.layout = dbc.Container([
        html.H1(children='Parque da Ciência - Dashboard', className="text-center my-4", style={'color': custom_styles['font_blue']}),

        # Filters
        dbc.Row([
            dbc.Col([
                html.Label("School:", style={'color': custom_styles['font_blue']}),
                dcc.Dropdown(
                    id='school-filter',
                    options=[{'label': school.title(), 'value': school} for school in school_names],
                    value=school_names,  # Select all schools by default
                    multi=True,
                    placeholder="Select a school"
                ),
            ], md=3),
            dbc.Col([
                html.Label("District:", style={'color': custom_styles['font_blue']}),
                dcc.Dropdown(
                    id='district-filter',
                    options=[{'label': district.title(), 'value': district} for district in city_districts],
                    value=city_districts,  # Select all districts by default
                    multi=True,
                    placeholder="Select a district"
                ),
            ], md=3),
            dbc.Col([
                html.Label("Rating:", style={'color': custom_styles['font_blue']}),
                dcc.Dropdown(
                    id='rating-filter',
                    options=[{'label': str(rating), 'value': rating} for rating in ratings],
                    value=ratings,  # Select all ratings by default
                    multi=True,
                    placeholder="Select a rating"
                ),
            ], md=2),
            dbc.Col([
                html.Label("Date Range:", style={'color': custom_styles['font_blue']}),
                dcc.DatePickerRange(
                    id='date-filter',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
                    display_format='DD/MM/YYYY',
                    start_date_placeholder_text='Start Date',
                    end_date_placeholder_text='End Date',
                ),
            ], md=4),
        ], className="mb-4"),

        # Metrics cards (will be updated via callback)
        html.Div(id='cards-metrics'),

        html.Hr(),

        # Graphs (will be updated via callback)
        html.Div(id='graphs-content'),

        html.Hr(),

        html.H3("Visitor Feedbacks", style={'color': custom_styles['font_blue']}),
        dash_table.DataTable(
            id='table-feedbacks',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        ),

        html.Footer("Developed by Daniel Yudi de Carvalho", className="text-center mt-4 mb-2")
    ], fluid=True)

    # Callbacks to update graphs and metrics
    @app.callback(
        [
            Output('cards-metrics', 'children'),
            Output('graphs-content', 'children'),
            Output('table-feedbacks', 'data'),
        ],
        [
            Input('school-filter', 'value'),
            Input('district-filter', 'value'),
            Input('rating-filter', 'value'),
            Input('date-filter', 'start_date'),
            Input('date-filter', 'end_date'),
        ]
    )
    def update_dashboard(selected_schools, selected_districts, selected_ratings, start_date, end_date):
        """
        Update the dashboard content based on selected filters.

        Parameters:
        - selected_schools (list): List of selected schools.
        - selected_districts (list): List of selected districts.
        - selected_ratings (list): List of selected ratings.
        - start_date (str): Start date string.
        - end_date (str): End date string.

        Returns:
        - cards: HTML content for the metrics cards.
        - graphs: HTML content for the graphs.
        - table_data: Data for the feedbacks table.
        """
        # Ensure inputs are lists
        if not isinstance(selected_schools, list):
            selected_schools = [selected_schools]
        if not isinstance(selected_districts, list):
            selected_districts = [selected_districts]
        if not isinstance(selected_ratings, list):
            selected_ratings = [selected_ratings]

        # Filter the data
        df_filtered = df.copy()

        # If filters are empty, select all data
        if not selected_schools:
            selected_schools = df['School_Name'].unique().tolist()
        if not selected_districts:
            selected_districts = df['City_District'].unique().tolist()
        if not selected_ratings:
            selected_ratings = df['Rating'].unique().tolist()

        # Apply filters
        df_filtered = df_filtered[df_filtered['School_Name'].isin(selected_schools)]
        df_filtered = df_filtered[df_filtered['City_District'].isin(selected_districts)]
        df_filtered = df_filtered[df_filtered['Rating'].isin(selected_ratings)]

        # Filter by date, if available
        if start_date and end_date:
            df_filtered = df_filtered[(df_filtered['Visit_DateTime'] >= start_date) & (df_filtered['Visit_DateTime'] <= end_date)]

        # Log after filtering
        logging.info(f"Number of rows after filtering: {len(df_filtered)}")

        if df_filtered.empty:
            # Return empty components or informative messages
            cards = dbc.Row([
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4(" NPS Score", className="card-title"),
                        html.H2("Sem dados", className="card-text"),
                    ]),
                    color="secondary", inverse=True
                ), width=4),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Total de visitantes", className="card-title"),
                        html.H2("Sem dados", className="card-text"),
                    ]),
                    color="secondary", inverse=True
                ), width=4),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Classificação média", className="card-title"),
                        html.H2("No data", className="card-text"),
                    ]),
                    color="secondary", inverse=True
                ), width=4),
            ], className="mb-4")

            graphs = html.Div("Não há dados disponíveis para os filtros selecionados. Ajuste seus critérios de filtro.", className="text-center")

            table_data = []

            return cards, graphs, table_data
        else:
            # Recalculate metrics and figures with filtered data
            nps_score, df_nps = calculate_nps(df_filtered)
            total_visitors = round(df_filtered['Student_Count'].sum(), 2)
            avg_rating = round(df_filtered['Rating'].mean(), 2)

            # Metrics cards
            cards = dbc.Row([
            dbc.Col(dbc.Card(
                dbc.CardBody([
                    html.H4("Pontuação geral do NPS", className="card-title", style={'color': 'white'}),
                    html.H2(nps_score, className="card-text", style={'color': 'white'}),
                ]),
                color=custom_styles['card_colors'][0], inverse=True
            ), width=4),
            dbc.Col(dbc.Card(
                dbc.CardBody([
                    html.H4("Total de visitantes", className="card-title", style={'color': 'white'}),
                    html.H2(total_visitors, className="card-text", style={'color': 'white'}),
                ]),
                color=custom_styles['card_colors'][1], inverse=True
            ), width=4),
            dbc.Col(dbc.Card(
                dbc.CardBody([
                    html.H4("Avaliação média", className="card-title", style={'color': 'white'}),
                    html.H2(avg_rating, className="card-text", style={'color': 'white'}),
                ]),
                color=custom_styles['card_colors'][2], inverse=True
            ), width=4),
        ], className="mb-4"),


            # Recreate figures with filtered data
            figures = create_figures(df_filtered)

            # Graphs content
            graphs = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H3("Ocorrências das opções selecionadas", style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['option_counts']) if figures['option_counts'] else html.Div("No data to display the options chart.")
                    ], width=6),
                    dbc.Col([
                        html.H3("Ocorrências por  Distritos",style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['city_district_counts']) if figures['city_district_counts'] else html.Div("No data to display the districts chart.")
                    ], width=6),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.H3("Total de alunos por escola", style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['student_count']) if figures['student_count'] else html.Div("No data to display the students chart.")
                    ], width=6),
                    dbc.Col([
                        html.H3("Total de visitas por distrito", style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['district_visits']) if figures['district_visits'] else html.Div("No data to display the visits chart.")
                    ], width=6),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.H3("Avaliação média por distrito", style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['avg_rating']) if figures['avg_rating'] else html.Div("No data to display the average rating chart.")
                    ], width=6),
                    dbc.Col([
                        html.H3("Distribuição de visitantes por faixa etária", style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['age_distribution']) if figures['age_distribution'] else html.Div("No data to display the age group chart.")
                    ], width=6),
                ]),

                html.Hr(),

                html.H3("NPS por distrito"),
                dcc.Graph(
                    figure=px.bar(
                        figures['df_nps'], x='City_District', y='NPS',
                        title='NPS por distrito (Satisfação dos visitantes)', color='NPS', color_continuous_scale='Viridis',
                        labels={'City_District': 'District', 'NPS': 'NPS Score'}
                    )
                ) if not figures['df_nps'].empty else html.Div("No data to display NPS by district."),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.H3("Visitas por data", style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['visits_by_day']) if figures['visits_by_day'] else html.Div('Date data unavailable.')
                    ], width=6),
                    dbc.Col([
                        html.H3("Visitas por hora", style={'color': custom_styles['font_blue']}),
                        dcc.Graph(figure=figures['visits_by_hour']) if figures['visits_by_hour'] else html.Div('Hour data unavailable.')
                    ], width=6),
                ]),

                html.Hr(),

                html.H3("Análise de Promotores, Passivos e Detratores", style={'color': custom_styles['font_blue']}),
                dcc.Graph(figure=figures['nps_breakdown']) if figures['nps_breakdown'] else html.Div("No data to display the NPS breakdown chart."),

                html.Hr(),

                html.H3("Correlação entre Opções e Rating", style={'color': custom_styles['font_blue']}),
                dcc.Graph(figure=figures['heatmap']) if figures['heatmap'] else html.Div("No data to display the heatmap."),
            ])

            # Data for the feedbacks table
            table_data = df_filtered.to_dict('records')

            return cards, graphs, table_data

    return app  # Ensure that the 'app' object is returned at the end of the function

# Main function
def main():
    dataset_directory = os.getenv('DATASET_DIRECTORY', 'data')
    df_combined = load_data(dataset_directory)
    if df_combined.empty:
        logging.error("No data to process. Exiting the program.")
        return

    df_processed = process_data(df_combined)

    app = build_dashboard(df_processed)

    # Run the app
    app.run_server(debug=True, host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
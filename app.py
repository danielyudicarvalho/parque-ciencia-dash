# Import packages
import os
import logging
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dash.dependencies import Input, Output

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_data(dataset_directory):
    """
    Carrega e combina todos os arquivos CSV do diretório especificado.
    """
    dataframes = []
    for file in os.listdir(dataset_directory):
        if file.endswith('.csv'):
            file_path = os.path.join(dataset_directory, file)
            try:
                df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()  # Remove espaços em branco nos nomes das colunas
                dataframes.append(df)
            except Exception as e:
                logging.error(f"Erro ao ler {file_path}: {e}")
    if dataframes:
        df_combined = pd.concat(dataframes, ignore_index=True)
        return df_combined
    else:
        logging.error("Nenhum dado foi carregado.")
        return pd.DataFrame()

def process_data(df):
    """
    Realiza o pré-processamento dos dados, incluindo conversões de tipo e criação de colunas adicionais.
    """
    # Remover espaços em branco nas colunas de texto e padronizar para minúsculas
    text_columns = ['School_Name', 'Server_Responsible', 'Option_1', 'Option_2', 'Option_3', 'Option_4', 'City_District']
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)  # Remove espaços múltiplos

    # Converter 'Visit_DateTime' para datetime
    df['Visit_DateTime'] = pd.to_datetime(df['Visit_DateTime'], errors='coerce')
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Student_Count'] = pd.to_numeric(df['Student_Count'], errors='coerce')
    df['Min_Age'] = pd.to_numeric(df['Min_Age'], errors='coerce')
    df['Max_Age'] = pd.to_numeric(df['Max_Age'], errors='coerce')

    # Criar coluna 'Faixa_Etaria'
    df['Faixa_Etaria'] = df['Min_Age'].astype(str) + ' - ' + df['Max_Age'].astype(str)

    # Classificar ratings para NPS
    df['NPS_Category'] = pd.cut(df['Rating'], bins=[0, 2, 3, 5], labels=['Detractors', 'Passives', 'Promoters'], right=True)

    # Extrair data e hora se 'Visit_DateTime' foi convertido com sucesso
    if pd.api.types.is_datetime64_any_dtype(df['Visit_DateTime']):
        df['Date'] = df['Visit_DateTime'].dt.date
        df['Hour'] = df['Visit_DateTime'].dt.hour
    else:
        logging.warning("'Visit_DateTime' não pôde ser convertido para datetime.")

    return df


def calculate_nps(df):
    """
    Calcula o NPS geral e por distrito.
    """
    # NPS geral
    nps_counts = df['NPS_Category'].value_counts()
    total_responses = nps_counts.sum()
    promoters = nps_counts.get('Promoters', 0)
    detractors = nps_counts.get('Detractors', 0)
    nps_score = ((promoters - detractors) / total_responses) * 100 if total_responses > 0 else 0

    # NPS por distrito
    df_nps = df.groupby('City_District')['NPS_Category'].value_counts().unstack(fill_value=0)
    df_nps['Total'] = df_nps.sum(axis=1)
    df_nps['NPS'] = ((df_nps.get('Promoters', 0) - df_nps.get('Detractors', 0)) / df_nps['Total']) * 100

    return nps_score, df_nps.reset_index()

def get_option_counts(df):
    """
    Conta as ocorrências das opções selecionadas pelos visitantes.
    """
    options = df[['Option_1', 'Option_2', 'Option_3', 'Option_4']].apply(lambda x: x.str.strip())
    options = options.values.flatten()
    options = pd.Series(options)
    option_counts = options.value_counts().dropna()
    return option_counts.reset_index().rename(columns={'index': 'Option Text', 0: 'count'})

def create_figures(df):
    """
    Cria os gráficos para visualização no dashboard.
    """
    figures = {}

    # Cálculo do NPS
    nps_score, df_nps = calculate_nps(df)
    figures['nps_score'] = nps_score
    figures['df_nps'] = df_nps if not df_nps.empty else pd.DataFrame({'City_District': [], 'NPS': []})

    # Contagem de Opções Selecionadas
    df_option_counts = get_option_counts(df)
    if not df_option_counts.empty:
        figures['option_counts'] = px.bar(df_option_counts, x='Option Text', y='count', title='Ocorrências das Opções Selecionadas')
    else:
        figures['option_counts'] = {}

    # Contagem de Distritos
    df_city_district_counts = df['City_District'].value_counts().reset_index()
    df_city_district_counts.columns = ['City District', 'count']
    if not df_city_district_counts.empty:
        figures['city_district_counts'] = px.bar(df_city_district_counts, x='City District', y='count', title='Ocorrências dos Distritos')
    else:
        figures['city_district_counts'] = {}

    # Student Count by School
    df_student_count = df.groupby('School_Name', as_index=False)['Student_Count'].sum()
    if not df_student_count.empty:
        figures['student_count'] = px.bar(df_student_count, x='School_Name', y='Student_Count', title='Total de Alunos por Escola')
    else:
        figures['student_count'] = {}

    # Visits by District
    df_district_visits = df.groupby('City_District', as_index=False)['Student_Count'].sum()
    if not df_district_visits.empty:
        figures['district_visits'] = px.bar(df_district_visits, x='City_District', y='Student_Count', title='Total de Visitas por Distrito')
    else:
        figures['district_visits'] = {}

    # Average Rating by District
    df_avg_rating = df.groupby('City_District', as_index=False)['Rating'].mean()
    if not df_avg_rating.empty:
        figures['avg_rating'] = px.bar(df_avg_rating, x='City_District', y='Rating', title='Média de Avaliação por Distrito')
    else:
        figures['avg_rating'] = {}

    # Age Distribution
    df_age_distribution = df.groupby('Faixa_Etaria', as_index=False)['Student_Count'].sum()
    if not df_age_distribution.empty:
        figures['age_distribution'] = px.bar(df_age_distribution, x='Faixa_Etaria', y='Student_Count', title='Distribuição de Faixa Etária dos Visitantes')
    else:
        figures['age_distribution'] = {}

    # Visits by Date
    if 'Date' in df.columns:
        df_visits_by_day = df.groupby('Date', as_index=False)['Student_Count'].sum()
        if not df_visits_by_day.empty:
            figures['visits_by_day'] = px.line(df_visits_by_day, x='Date', y='Student_Count', title='Visitas Totais por Data')
        else:
            figures['visits_by_day'] = None
    else:
        figures['visits_by_day'] = None

    # Visits by Hour
    if 'Hour' in df.columns:
        df_visits_by_hour = df.groupby('Hour', as_index=False)['Student_Count'].sum()
        if not df_visits_by_hour.empty:
            figures['visits_by_hour'] = px.line(df_visits_by_hour, x='Hour', y='Student_Count', title='Visitas Totais por Hora')
        else:
            figures['visits_by_hour'] = None
    else:
        figures['visits_by_hour'] = None

    # NPS Breakdown
    nps_counts = df['NPS_Category'].value_counts().reset_index()
    nps_counts.columns = ['Group', 'count']
    if not nps_counts.empty:
        figures['nps_breakdown'] = px.pie(nps_counts, names='Group', values='count', title='Distribuição de Promotores, Passivos e Detratores')
    else:
        figures['nps_breakdown'] = {}

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
            corr_matrix = df_option_rating_corr.pivot_table(index='Option_Text', values='Rating')
            if not corr_matrix.empty:
                plt.figure(figsize=(10, 8))
                sns_heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', cbar=True)
                plt.title("Correlação entre Opções e Avaliação")

                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                figures['heatmap_image'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
            else:
                figures['heatmap_image'] = None
        else:
            figures['heatmap_image'] = None
    else:
        figures['heatmap_image'] = None

    return figures

def build_dashboard(df):
    """
    Constrói o layout do dashboard do Dash com filtros interativos.
    """
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Listas para os filtros
    school_names = df['School_Name'].dropna().unique().tolist()
    city_districts = df['City_District'].dropna().unique().tolist()
    min_date = df['Visit_DateTime'].min().date() if not df['Visit_DateTime'].isnull().all() else None
    max_date = df['Visit_DateTime'].max().date() if not df['Visit_DateTime'].isnull().all() else None
    ratings = sorted(df['Rating'].dropna().unique())

    # Layout principal com filtros
    app.layout = dbc.Container([
        html.H1(children='Parque da Ciência - Dashboard', className="text-center my-4"),

        # Filtros
        dbc.Row([
            dbc.Col([
                html.Label("Escola:"),
                dcc.Dropdown(
                    id='school-filter',
                    options=[{'label': school, 'value': school} for school in school_names],
                    value=school_names,  # Seleciona todas as escolas por padrão
                    multi=True,
                    placeholder="Selecione a escola"
                ),
            ], md=3),
            dbc.Col([
                html.Label("Distrito:"),
                dcc.Dropdown(
                    id='district-filter',
                    options=[{'label': district, 'value': district} for district in city_districts],
                    value=city_districts,  # Seleciona todos os distritos por padrão
                    multi=True,
                    placeholder="Selecione o distrito"
                ),
            ], md=3),
            dbc.Col([
                html.Label("Avaliação:"),
                dcc.Dropdown(
                    id='rating-filter',
                    options=[{'label': str(rating), 'value': rating} for rating in ratings],
                    value=ratings,  # Seleciona todas as avaliações por padrão
                    multi=True,
                    placeholder="Selecione a avaliação"
                ),
            ], md=2),
            dbc.Col([
                html.Label("Período:"),
                dcc.DatePickerRange(
                    id='date-filter',
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
                    display_format='DD/MM/YYYY',
                    start_date_placeholder_text='Data inicial',
                    end_date_placeholder_text='Data final',
                ),
            ], md=4),
        ], className="mb-4"),

        # Cards de métricas (iremos atualizar via callback)
        html.Div(id='cards-metrics'),

        html.Hr(),

        # Gráficos (iremos atualizar via callback)
        html.Div(id='graphs-content'),

        html.Hr(),

        html.H3("Feedbacks dos Visitantes"),
        dash_table.DataTable(
            id='table-feedbacks',
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        ),

        html.Footer("Desenvolvido por Daniel Yudi de Carvalho", className="text-center mt-4 mb-2")
    ], fluid=True)

    # Callbacks para atualizar os gráficos e métricas
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
        # Filtragem dos dados
        df_filtered = df.copy()

        # Se os filtros forem nulos ou vazios, carregar todos os dados
        if not selected_schools:
            selected_schools = df['School_Name'].unique().tolist()  # Todos os valores de escolas
        if not selected_districts:
            selected_districts = df['City_District'].unique().tolist()  # Todos os valores de distritos
        if not selected_ratings:
            selected_ratings = df['Rating'].unique().tolist()  # Todos os valores de avaliações

        # Aplicar filtragem conforme os valores selecionados
        df_filtered = df_filtered[df_filtered['School_Name'].isin(selected_schools)]
        df_filtered = df_filtered[df_filtered['City_District'].isin(selected_districts)]
        df_filtered = df_filtered[df_filtered['Rating'].isin(selected_ratings)]

        # Filtrar por data, se disponível
        if start_date and end_date:
            df_filtered = df_filtered[(df_filtered['Visit_DateTime'] >= start_date) & (df_filtered['Visit_DateTime'] <= end_date)]

        # Verificação após filtragem
        logging.info(f"Número de linhas após filtragem: {len(df_filtered)}")

        if df_filtered.empty:
            # Retornar componentes vazios ou mensagens informativas
            cards = dbc.Row([
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("NPS Score Geral", className="card-title"),
                        html.H2("Sem dados", className="card-text"),
                    ]),
                    color="secondary", inverse=True
                ), width=4),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Total de Visitantes", className="card-title"),
                        html.H2("Sem dados", className="card-text"),
                    ]),
                    color="secondary", inverse=True
                ), width=4),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Média de Avaliação", className="card-title"),
                        html.H2("Sem dados", className="card-text"),
                    ]),
                    color="secondary", inverse=True
                ), width=4),
            ], className="mb-4")

            graphs = html.Div("Nenhum dado disponível com os filtros selecionados.", className="text-center")

            table_data = []

            return cards, graphs, table_data
        else:
            # Recalcular métricas e figuras com os dados filtrados
            nps_score, df_nps = calculate_nps(df_filtered)
            total_visitors = df_filtered['Student_Count'].sum()
            avg_rating = df_filtered['Rating'].mean()

            # Cards de métricas
            cards = dbc.Row([
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("NPS Score Geral", className="card-title"),
                        html.H2(f"{nps_score:.2f}", className="card-text"),
                    ]),
                    color="primary", inverse=True
                ), width=4),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Total de Visitantes", className="card-title"),
                        html.H2(f"{total_visitors}", className="card-text"),
                    ]),
                    color="info", inverse=True
                ), width=4),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4("Média de Avaliação", className="card-title"),
                        html.H2(f"{avg_rating:.2f}", className="card-text"),
                    ]),
                    color="success", inverse=True
                ), width=4),
            ], className="mb-4")

            # Recriar as figuras com os dados filtrados
            figures = create_figures(df_filtered)

            # Conteúdo dos gráficos
            graphs = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.H3("Ocorrências das Opções Selecionadas"),
                        dcc.Graph(figure=figures['option_counts']) if figures['option_counts'] else html.Div("Sem dados para exibir o gráfico de opções selecionadas.")
                    ], width=6),
                    dbc.Col([
                        html.H3("Ocorrências dos Distritos"),
                        dcc.Graph(figure=figures['city_district_counts']) if figures['city_district_counts'] else html.Div("Sem dados para exibir o gráfico de distritos.")
                    ], width=6),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.H3("Total de Alunos por Escola"),
                        dcc.Graph(figure=figures['student_count']) if figures['student_count'] else html.Div("Sem dados para exibir o gráfico de alunos por escola.")
                    ], width=6),
                    dbc.Col([
                        html.H3("Total de Visitas por Distrito"),
                        dcc.Graph(figure=figures['district_visits']) if figures['district_visits'] else html.Div("Sem dados para exibir o gráfico de visitas por distrito.")
                    ], width=6),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.H3("Média de Avaliação por Distrito"),
                        dcc.Graph(figure=figures['avg_rating']) if figures['avg_rating'] else html.Div("Sem dados para exibir o gráfico de média de avaliação por distrito.")
                    ], width=6),
                    dbc.Col([
                        html.H3("Distribuição de Faixa Etária dos Visitantes"),
                        dcc.Graph(figure=figures['age_distribution']) if figures['age_distribution'] else html.Div("Sem dados para exibir o gráfico de faixa etária.")
                    ], width=6),
                ]),

                html.Hr(),

                html.H3("NPS por Distrito"),
                dcc.Graph(
                    figure=px.bar(figures['df_nps'], x='City_District', y='NPS', title='NPS por Distrito (Satisfação dos Visitantes)', color='NPS', color_continuous_scale='Viridis')
                ) if not figures['df_nps'].empty else html.Div("Sem dados para exibir o NPS por distrito."),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        html.H3("Visitas por Data"),
                        dcc.Graph(figure=figures['visits_by_day']) if figures['visits_by_day'] else html.Div('Dados de data indisponíveis.')
                    ], width=6),
                    dbc.Col([
                        html.H3("Visitas por Hora"),
                        dcc.Graph(figure=figures['visits_by_hour']) if figures['visits_by_hour'] else html.Div('Dados de hora indisponíveis.')
                    ], width=6),
                ]),

                html.Hr(),

                html.H3("Análise de Promotores, Passivos e Detratores"),
                dcc.Graph(figure=figures['nps_breakdown']) if figures['nps_breakdown'] else html.Div("Sem dados para exibir o gráfico de NPS."),

                html.Hr(),

                html.H3("Correlação entre Opções e Avaliação"),
                html.Img(src=f'data:image/png;base64,{figures["heatmap_image"]}', style={'width': '100%', 'height': 'auto'}) if figures['heatmap_image'] else html.Div("Sem dados para gerar o heatmap."),
            ])

            # Dados da tabela de feedbacks
            table_data = df_filtered.to_dict('records')

            return cards, graphs, table_data

    return app  # Certifique-se de retornar o objeto 'app' no final da função


# Função main
def main():
    dataset_directory = 'data'
    df_combined = load_data(dataset_directory)
    if df_combined.empty:
        logging.error("Nenhum dado para processar. Encerrando o programa.")
        return

    # Verificações adicionais
    print("Colunas do DataFrame:", df_combined.columns)
    print("Primeiras linhas do DataFrame:")
    print(df_combined.head())

    df_processed = process_data(df_combined)

    # Verificar tipos de dados
    print("Tipos de dados após processamento:")
    print(df_processed.dtypes)

    print("Valores únicos em 'School_Name':", df_processed['School_Name'].unique())
    print("Valores únicos em 'City_District':", df_processed['City_District'].unique())
    print("Valores únicos em 'Rating':", df_processed['Rating'].unique())

    print("Valores nulos por coluna:")
    print(df_processed[['School_Name', 'City_District', 'Rating', 'Visit_DateTime']].isnull().sum())

    app = build_dashboard(df_processed)

    # Run the app
    app.run_server(debug=True)


if __name__ == '__main__':
    main()

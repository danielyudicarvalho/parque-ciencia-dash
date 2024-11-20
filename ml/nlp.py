# Importar pacotes adicionais para análise de sentimentos
import os
import glob
import logging
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dash import Dash, dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Baixar recursos necessários do NLTK
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# Configurar logging
logging.basicConfig(level=logging.INFO)

def carregar_dados(diretorio_dados):
    # (Função permanece a mesma)
    # ...

def processar_dados(df):
    # (Função permanece a mesma)
    # ...
    return df

def analisar_sentimentos(df):
    """
    Realiza a análise de sentimentos nos feedbacks dos visitantes.

    Parâmetros:
    - df (pd.DataFrame): O DataFrame pré-processado.

    Retorna:
    - pd.DataFrame: O DataFrame com uma nova coluna 'Sentimento' contendo o resultado da análise.
    """
    # Inicializar o analisador de sentimentos
    sid = SentimentIntensityAnalyzer()

    # Função para limpar o texto
    def limpar_texto(texto):
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        import string

        # Converter para minúsculas
        texto = texto.lower()
        # Remover pontuação
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        # Tokenizar palavras
        palavras = word_tokenize(texto)
        # Remover stopwords
        palavras = [palavra for palavra in palavras if palavra not in stopwords.words('portuguese')]
        # Reunir palavras limpas
        texto_limpo = ' '.join(palavras)
        return texto_limpo

    # Aplicar limpeza de texto
    df['Feedback_Limpo'] = df['Feedback'].fillna('').apply(limpar_texto)

    # Função para classificar sentimento
    def classificar_sentimento(texto):
        if texto.strip() == '':
            return 'Neutro'
        pontuacao = sid.polarity_scores(texto)
        if pontuacao['compound'] >= 0.05:
            return 'Positivo'
        elif pontuacao['compound'] <= -0.05:
            return 'Negativo'
        else:
            return 'Neutro'

    # Aplicar análise de sentimentos
    df['Sentimento'] = df['Feedback_Limpo'].apply(classificar_sentimento)

    return df

def calcular_nps(df):
    # (Função permanece a mesma)
    # ...

def obter_contagens_opcoes(df):
    # (Função permanece a mesma)
    # ...

def criar_figuras(df):
    # (Parte do código permanece a mesma)
    figuras = {}
    # ...

    # Gráfico de Sentimentos
    df_sentimentos = df['Sentimento'].value_counts().reset_index()
    df_sentimentos.columns = ['Sentimento', 'Contagem']
    if not df_sentimentos.empty:
        figuras['sentimentos'] = px.bar(
            df_sentimentos, x='Sentimento', y='Contagem',
            title='Distribuição dos Sentimentos dos Feedbacks',
            labels={'Contagem': 'Número de Feedbacks', 'Sentimento': 'Sentimento'},
            color='Sentimento',
            color_discrete_map={'Positivo': 'green', 'Neutro': 'gray', 'Negativo': 'red'}
        )
    else:
        figuras['sentimentos'] = None

    return figuras

def construir_dashboard(df):
    # (Parte do código permanece a mesma)
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    # ...

    # Atualizar o layout para incluir o novo gráfico
    app.layout = dbc.Container([
        # (Layout existente)
        # ...

        html.Hr(),

        # Gráfico de Sentimentos
        html.Div(id='conteudo-sentimentos'),

        html.Hr(),

        # (Restante do layout)
        # ...
    ], fluid=True)

    # Atualizar o callback para incluir o gráfico de sentimentos e a coluna na tabela
    @app.callback(
        [
            Output('cards-metricas', 'children'),
            Output('conteudo-graficos', 'children'),
            Output('tabela-feedbacks', 'data'),
            Output('conteudo-sentimentos', 'children'),
        ],
        [
            Input('filtro-escola', 'value'),
            Input('filtro-distrito', 'value'),
            Input('filtro-avaliacao', 'value'),
            Input('filtro-data', 'start_date'),
            Input('filtro-data', 'end_date'),
        ]
    )
def atualizar_dashboard(escolas_selecionadas, distritos_selecionados, avaliacoes_selecionadas, data_inicio, data_fim):
    # (Parte do código permanece a mesma)
    # ...

    if df_filtrado.empty:
        # (Parte do código permanece a mesma)
        # ...

        return cards, graficos, tabela_dados, None
    else:
        # (Recalcular métricas e figuras com os dados filtrados)
        # ...

        # Conteúdo do gráfico de sentimentos
        grafico_sentimentos = dcc.Graph(figure=figuras['sentimentos']) if figuras['sentimentos'] else html.Div("Sem dados para exibir o gráfico de sentimentos.")

        # Atualizar a tabela de feedbacks para incluir a coluna 'Sentimento'
        colunas_tabela = [{'name': col, 'id': col} for col in df_filtrado.columns]

        # Dados para a tabela de feedbacks
        tabela_dados = df_filtrado.to_dict('records')

        return cards, graficos, tabela_dados, grafico_sentimentos

    return app  # Certifique-se de retornar o objeto 'app' no final da função

def main():
    diretorio_dados = os.getenv('DIRETORIO_DADOS', 'data')
    df_combinado = carregar_dados(diretorio_dados)
    if df_combinado.empty:
        logging.error("Nenhum dado para processar. Encerrando o programa.")
        return

    df_processado = processar_dados(df_combinado)

    # Adicionar a análise de sentimentos
    df_processado = analisar_sentimentos(df_processado)

    app = construir_dashboard(df_processado)

    # Executar a aplicação
    app.run_server(debug=True)

if __name__ == '__main__':
    main()

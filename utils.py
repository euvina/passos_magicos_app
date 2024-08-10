import os
import re
from typing import List
from itertools import product
from pprint import pprint

import pandas as pd
import numpy as np
from io import BytesIO

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

pd.set_option('future.no_silent_downcasting', True)

# ----------------------------------------------------------------------------------- #
# ------------------------------[ NOTEBOOK FUNCTIONS ]------------------------------- #
# ----------------------------------------------------------------------------------- #

# Verificação de valores únicos em todas as colunas
def print_unique_values(df):
    """
    Função para imprimir de forma mais legível os valores únicos de todas as colunas de um DataFrame.
    Utiliza pprint para uma formatação mais agradável.

    Parâmetros:
    - df (pd.DataFrame): DataFrame do qual os valores únicos serão impressos.
    """
    for column in df.columns:
        print(f'Valores únicos na coluna "{column}":')
        pprint(sorted(df[column].dropna().unique()), compact=True)
        print('\n' + '-' * 50)
    return None

# Salvar DataFrames em arquivos CSV
def save_dataframes(dfs: List[pd.DataFrame], names: List[str], folder: str = 'data'):
    """
    Salva DataFrames em arquivos CSV no diretório especificado.

    Parâmetros:
    - dfs (List[pd.DataFrame]): Lista de DataFrames a serem salvos.
    - names (List[str]): Lista de nomes para os DataFrames.
    - folder (str): Diretório onde os arquivos serão salvos. O padrão é 'data'.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    for df, name in zip(dfs, names):
        df.to_csv(f'{folder}/{name}.csv', index=False)
    return None

# ----------------------------------------------------------------------------------- #
# --------------------------[ MAIN APPLICATION  FUNCTIONS ]-------------------------- #
# ----------------------------------------------------------------------------------- #

#                                [   Prepare  Data   ]                                #

# Leitura dos dados
def read_data(data_path: str, sep: str = ';', encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Lê um arquivo csv e retorna um DataFrame. 
    Se o arquivo não puder ser lido, tenta ler com separador ',', 
    se não funcionar, tenta ler com separador '\t'. Se não funcionar, retorna um erro.
    
    Parâmetros:
    data_path : str
        Caminho do arquivo csv.
    sep : str, opcional
        Separador dos dados. O padrão é ';'.
    encoding : str, opcional
        Codificação do arquivo. O padrão é 'utf-8'.
    
    Retorna:
    pd.DataFrame
        DataFrame com os dados do arquivo csv.
    """
    try:
        df = pd.read_csv(data_path, sep=sep, encoding=encoding)
    except ValueError:
        try:
            df = pd.read_csv(data_path, sep=',', encoding=encoding)
        except ValueError:
            try:
                df = pd.read_csv(data_path, sep='\t', encoding=encoding)
            except ValueError:
                try:
                    df = pd.read_csv(data_path, sep=';', encoding='latin1')
                except ValueError:
                    try:
                        df = pd.read_csv(data_path, sep=',', encoding='latin1')
                    except ValueError:
                        df = pd.read_csv(data_path, sep='\t', encoding='latin1')
                        try:
                            df = pd.read_csv(data_path, sep=';')
                        except ValueError:
                            try:
                                df = pd.read_csv(data_path, sep=',')
                            except ValueError:
                                df = pd.read_csv(data_path, sep='\t')
                            except ValueError:
                                print('Erro ao ler o arquivo. Verifique o separador e a codificação.')
    return df

# Limpeza inicial de dados
def clean_dataframe_separate_issues(df: pd.DataFrame, pattern: str = '[A-Z]\d+(?:/\d+)?', remove_duplicates: bool = False) -> tuple:
    """
    Separa linhas com erros, como 'ERRO' e '#NULO!' em um novo DataFrame para verificação do usuário.
    Além disso, cria um DataFrame para valores removidos baseados no padrão.

    Parâmetros:
    - df (pd.DataFrame): DataFrame a ser modificado.
    - pattern (str): Padrão regex para identificar valores codificados em todas as colunas. Corrigido para evitar aviso de grupos de captura.
    O valor padrão é '[A-Z]\d+(?:/\d+)?'.
    - remove_duplicates (bool): Se True, remove duplicatas após a limpeza inicial.

    Retorna:
    - tuple: Contendo o DataFrame limpo, DataFrame com valores a serem verificados pelo usuário, e DataFrame com valores removidos baseados no padrão.
    """
    
    # Corrigindo o aviso de grupos de captura no padrão regex, usando ?: para grupos não capturantes
    pattern_compiled = re.compile(pattern)
    
    # Identificar linhas com valores codificados em todas as colunas
    encoded_mask = df.apply(lambda x: x.str.contains(pattern_compiled, na=False) if x.dtype == "object" else False)
    
    # Identificar linhas com valores específicos para verificação
    specific_values_mask = df.isin(['ERRO', '#NULO!', 'nan'])
    
    # Separar DataFrames
    df_to_verify = df[specific_values_mask.any(axis=1)]
    df_removed_based_on_pattern = df[encoded_mask.any(axis=1)]
    df_cleaned = df[~encoded_mask.any(axis=1)]
    
    # replace values 'ERRO', '#NULO!', 'nan' with np.nan
    df_cleaned = df_cleaned.replace(['ERRO', '#NULO!', 'nan'], np.nan)
    
    # Remover duplicatas, se solicitado
    if remove_duplicates:
        df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned, df_to_verify, df_removed_based_on_pattern


# Tratar Sim e Não para booleano
def replace_yes_no(df):
    """
    Função para substituir valores 'Sim' e 'Não' por True e False, respectivamente, em todas as colunas de um DataFrame.

    Parâmetros:
    - df (pd.DataFrame): DataFrame no qual os valores serão substituídos.
    """
    df.replace({'Sim': True, 'Não': False}, inplace=True)
    # if col contains True or False, convert to boolean
    for col in df.columns:
        if df[col].isin([True, False]).all():
            df[col] = df[col].astype(bool)
    
    return df


# Criar DataFrame com todas as colunas separadas por ano
def create_df_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função para criar um único DataFrame 'df_all' a partir das colunas de um DataFrame de entrada,
    onde os anos são separados por uma coluna e não mais em variáveis únicas.
    As colunas sem ano são mantidas em todas as linhas, enquanto as colunas com ano são separadas
    em linhas diferentes com uma coluna 'ano' identificando o ano correspondente.

    Parâmetros:
    - df (pd.DataFrame): O DataFrame de entrada contendo as colunas a serem separadas.

    Retorna:
    - pd.DataFrame: DataFrame concatenado com uma coluna 'ano' identificando o ano de cada linha.
    """

    # Separar colunas sem ano e com ano
    cols_without_year = [col for col in df.columns if not re.search(r'_[0-9]{4}$', col)]
    cols_with_year = [col for col in df.columns if re.search(r'_[0-9]{4}$', col)]

    # Identificar os anos presentes nas colunas
    years = sorted(set([int(re.search(r'_(\d{4})$', col).group(1)) for col in cols_with_year]))

    # Lista para armazenar os DataFrames temporários
    temp_dfs = []

    # Criar um DataFrame para cada ano
    for year in years:
        year_cols = [col for col in cols_with_year if f'_{year}' in col]
        selected_cols = cols_without_year + year_cols
        df_year = df[selected_cols].copy()

        # Renomear colunas para remover o sufixo do ano
        df_year.columns = [col.replace(f'_{year}', '') for col in df_year.columns]

        # Adicionar coluna 'ano'
        df_year['ano'] = year

        temp_dfs.append(df_year)

    # Concatenar todos os DataFrames temporários
    df_all = pd.concat(temp_dfs, ignore_index=True)
    
    # Padronizar nomes das colunas para minúsculas
    df_all.columns = map(str.lower, df_all.columns)

    # Remover linhas onde todos os indicadores especificados são vazios
    inde_indices = ['iaa', 'ieg', 'ips', 'ida', 'ipp', 'ipv', 'ian']
    df_all = df_all.dropna(subset=inde_indices, how='all')
    
    # Resetar o índice
    df_all = df_all.reset_index(drop=True)

    return df_all

# Separar valores por fase e turma
def split_fase_turma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Separa a coluna 'fase_turma' em colunas separadas 'fase' e 'turma' em um DataFrame.

    Esta função recebe um DataFrame que inclui uma coluna 'fase_turma', onde 'fase' representa
    a fase (como um número) e 'turma' representa a classe (como uma letra maiúscula). Ela divide
    esses valores em colunas separadas 'fase' e 'turma'. Linhas onde 'fase' e 'turma' são ambos
    nulos são especificamente visadas para esta operação. Após a divisão, a coluna 'fase_turma'
    é removida do DataFrame.

    Parâmetros:
    - df (pd.DataFrame): O DataFrame de entrada contendo a coluna 'fase_turma' junto com as existentes
      colunas 'fase' e 'turma' que podem ter valores nulos.

    Retorna:
    - pd.DataFrame: Um DataFrame com a coluna 'fase_turma' dividida em colunas 'fase' e 'turma',
      e a coluna original 'fase_turma' removida.

    Exemplo:
    >>> df = pd.DataFrame({
    ...     'fase_turma': ['1A', None, '2B'],
    ...     'fase': [None, 2, None],
    ...     'turma': [None, 'B', None]
    ... })
    >>> split_fase_turma(df)
       fase turma
    0     1     A
    1     2     B
    2     2     B
    """

    # Cria uma cópia das linhas onde 'fase' e 'turma' são ambos nulos
    df_fase_null = df.loc[df['fase'].isnull() & df['turma'].isnull()].copy()

    # Extrai 'fase' e 'turma' de 'fase_turma' usando regex
    df_fase_null[['fase', 'turma']] = df_fase_null['fase_turma'].str.extract('(\d+)([A-Z])')

    # Converte 'fase' para inteiro
    df_fase_null['fase'] = df_fase_null['fase'].astype(int)

    # Preenche valores nulos no dataframe original com valores de df_fase_null
    df['fase'] = df['fase'].fillna(df_fase_null['fase'])
    df['turma'] = df['turma'].fillna(df_fase_null['turma'])

    # Remove a coluna 'fase_turma'
    df = df.drop(columns=['fase_turma'])

    return df


# Divisão Escolar/Universitário (Fases 1-7 e Fase 8)
def add_group_fase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona uma coluna 'grupo_fase' ao DataFrame, categorizando as fases em 'Fases 1-7' para fases de 1 a 7,
    e 'Fase 8' para a fase 8. A função tenta converter a coluna 'fase' para inteiros antes de aplicar a categorização.

    Parâmetros:
    - df (pd.DataFrame): DataFrame que contém a coluna 'fase'.

    Retorna:
    - pd.DataFrame: DataFrame com a coluna 'grupo_fase' adicionada.

    Exceções:
    - Lança uma exceção se a conversão da coluna 'fase' para inteiros falhar.

    Exemplo:
    >>> df = pd.DataFrame({'fase': [1, 5, 8, None]})
    >>> add_group_fase(df)
       fase  grupo_fase
    0     1  Fases 1-7
    1     5  Fases 1-7
    2     8      Fase 8
    3  None        None
    """
    try:
        df['fase'] = df['fase'].astype(int)
    except Exception as e:
        print(f"Erro ao converter coluna fase: {e}")
    # Cria a coluna 'grupo_fase' com base na coluna 'fase'
    df['grupo_fase'] = df['fase'].apply(lambda x: x if pd.isna(x) else 'Fases 1-7' if x < 8 else 'Fase 8')

    return df


# Ajustar valores de indicadores para os limites 0 e 10
def adjust_indicators(df: pd.DataFrame, num_indicators: List[str] = ['inde', 'ian', 'iaa', 'ida', 'ieg', 'ipp', 'ipv']) -> pd.DataFrame:
    """
    Ajusta os valores dos indicadores numéricos de um DataFrame, definindo um limite inferior de 0 e um limite superior de 10.
    Qualquer valor de indicador abaixo de 0 é ajustado para 0, e qualquer valor acima de 10 é ajustado para 10.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os indicadores a serem ajustados.
    - num_indicators (List[str]): Lista de colunas no DataFrame que representam os indicadores numéricos a serem ajustados.
      O padrão inclui 'inde', 'ian', 'iaa', 'ida', 'ieg', 'ipp', 'ipv'.

    Retorna:
    - pd.DataFrame: DataFrame com os valores dos indicadores ajustados conforme os limites especificados.
    """
    # Converte os indicadores para float para garantir operações numéricas
    df[num_indicators] = df[num_indicators].astype('float')

    # Aplica a condição de limite inferior e superior usando np.clip para otimização
    df[num_indicators] = df[num_indicators].apply(lambda x: np.clip(x, 0, 10))

    return df


def calculate_inde(dataframe, user_weights=None) -> pd.DataFrame:
    """
    Calcula o Índice de Desempenho Educacional (INDE) com base em variáveis de avaliação e conselho,
    aplicando ponderações específicas para a fase 8 e padrão para as demais fases.
    
    Parâmetros:
        dataframe (pd.DataFrame): DataFrame contendo as colunas necessárias para o cálculo do INDE.
        user_weights (dict, opcional): Dicionário contendo ponderações personalizadas para as variáveis.
                                       As chaves devem ser os nomes das variáveis e os valores as ponderações.
    
    Retorna:
        pd.DataFrame: DataFrame original com uma nova coluna 'inde' contendo os valores calculados do INDE.
    
    Notas:
        - As variáveis de avaliação são: IAA (Avaliação Acadêmica), IAN (Avaliação de Necessidades),
          IEG (Engajamento), IDA (Desempenho Acadêmico).
        - As variáveis de conselho são: IPS (Participação Social), IPP (Participação dos Pais),
          IPV (Visão de Futuro).
        - A função verifica se a coluna 'inde' não existe ou se existe mas contém valores nulos ou zero.
        - Se 'user_weights' não for fornecido, serão usadas as ponderações padrão.
        - Ignora linhas onde todas as variáveis de indicadores são vazias e preenche valores vazios com 0
          se alguma variável de indicador estiver preenchida na mesma linha.
    """
    
    # Ponderações padrão
    default_weights = {
        'iaa': 0.1, 'ian': 0.1, 'ieg': 0.2, 'ips': 0.1, 'ipp': 0.1, 'ipv': 0.2, 'ida': 0.2
    }
    
    # Atualiza as ponderações padrão com as personalizadas, se fornecidas
    if user_weights is not None:
        default_weights.update(user_weights)
    
    indicator_columns = list(default_weights.keys())
    
    # Ignora linhas onde todas as variáveis de indicadores são vazias
    dataframe = dataframe.dropna(subset=indicator_columns, how='all')
    
    # Preenche valores vazios com 0 se alguma variável de indicador estiver preenchida na mesma linha
    dataframe[indicator_columns] = dataframe[indicator_columns].fillna(0)
    
    # Converte as variáveis de indicadores para float
    dataframe[indicator_columns] = dataframe[indicator_columns].astype(float)
    
    # Função para calcular o INDE por linha
    def calc_inde(row):
        if row['fase'] == 8:
            # Ponderações para a fase 8
            weights = default_weights.copy()
            weights.update({'ida': 0.4, 'ips': 0.2, 'ipp': 0, 'ipv': 0})
        else:
            weights = default_weights
        
        # Cálculo do INDE
        inde = sum(row[var] * weight for var, weight in weights.items())
        return inde
    
    # Verifica se a coluna 'inde' precisa ser calculada
    if 'inde' not in dataframe.columns or dataframe['inde'].isnull().any() or (dataframe['inde'] == 0).any():
        dataframe['inde'] = dataframe.apply(calc_inde, axis=1)
    
    return dataframe


# Remover linhas com 'inde' nulo
def dropna_inde(df):
    """
    Remove linhas do DataFrame onde a coluna 'inde' possui valores nulos.

    Esta função filtra o DataFrame para manter apenas as linhas onde a coluna 'inde'
    tem valores não nulos. Após a remoção, os índices do DataFrame são redefinidos para
    uma sequência contínua começando de 0, sem manter os índices antigos.

    Parâmetros:
    - df (pd.DataFrame): O DataFrame de entrada que contém a coluna 'inde'.

    Retorna:
    - pd.DataFrame: Um DataFrame com as linhas onde a coluna 'inde' é não nula,
      e os índices redefinidos.

    Exemplo:
    >>> df = pd.DataFrame({
    ...     'inde': [1, None, 2, 0],
    ...     'outra_coluna': ['a', 'b', 'c', 'd']
    ... })
    >>> dropna_inde(df)
       inde outra_coluna
    0     1            a
    1     2            c
    """
    # Remove linhas onde 'inde' é nulo ou ZERP
    df = df[df.inde.notna() & (df.inde != 0)]
    df = df.reset_index(drop=True)

    return df


# Preencher a coluna 'pedra' com base no valor de 'inde'
def fill_pedra(df: pd.DataFrame, return_stats: bool = False, recalculate: bool = False) -> pd.DataFrame:
    """
    Preenche 'pedra' com valores categóricos baseados no valor da coluna 'inde',
    agrupados por 'ano'. Cria a coluna 'pedra' se não existir.
    - 'Quartzo' para valores de INDE menores ou iguais à média menos um desvio padrão;
    - 'Ágata' para valores de INDE maiores que a média menos um desvio padrão e menores ou iguais à média;
    - 'Ametista' para valores de INDE maiores que a média e menores ou iguais à média mais um desvio padrão;
    - 'Topázio' para valores de INDE maiores que a média mais um desvio padrão.
    O valor de INDE igual a zero é considerado como NA.
    
    Parâmetros:
    - df (pd.DataFrame): DataFrame de entrada.
    - return_stats (bool): Se True, retorna um DataFrame com estatísticas das pedras.
    - recalculate (bool): Se True, recalcula e preenche a coluna 'pedra', mesmo que já exista.

    Retorna:
    - pd.DataFrame: DataFrame com a coluna 'pedra' preenchida.
    - pd.DataFrame (opcional): DataFrame com estatísticas das pedras, se return_stats é True.
    """
    if recalculate and 'pedra' in df.columns:
        df.drop(columns=['pedra'], inplace=True)
    
    if 'pedra' not in df.columns:
        df['pedra'] = pd.NA

    df['inde'] = df['inde'].astype(float)
    
    # Calculando média e desvio padrão por 'ano' em uma única operação
    stats = df.query('inde > 0').groupby('ano')['inde'].agg(['mean', 'std']).reset_index()

    # Preparando os limites para cada categoria
    stats['mean_minus_std'] = stats['mean'] - stats['std']
    stats['mean_plus_std'] = stats['mean'] + stats['std']
    
    # Mapeando os valores para a nova coluna 'pedra' com base nos limites calculados
    for _, row in stats.iterrows():
        year, mean, std, mean_minus_std, mean_plus_std = row
        df.loc[(df['ano'] == year) & (df['inde'] <= mean_minus_std), 'pedra'] = 'Quartzo'
        df.loc[(df['ano']== year) & (mean_minus_std < df['inde']) & (df['inde'] <= mean), 'pedra'] = 'Ágata'
        df.loc[(df['ano'] == year) & (mean < df['inde']) & (df['inde'] <= mean_plus_std), 'pedra'] = 'Ametista'
        df.loc[(df['ano'] == year) & (mean_plus_std < df['inde']), 'pedra'] = 'Topázio'

    if return_stats:
        # Calculando estatísticas para cada pedra
        pedra_stats = df.groupby(['ano']).size().reset_index(name='contagem')
        pedra_stats = pedra_stats.merge(stats, on='ano', how='left')
        pedra_stats['min'] = df.groupby(['ano'])['inde'].min().values
        pedra_stats['max'] = df.groupby(['ano'])['inde'].max().values
        pedra_stats.rename(columns={'mean': 'media', 
                                    'std': 'desvio_padrao', 
                                    'mean_minus_std': 'limite_inferior', 
                                    'mean_plus_std': 'limite_superior'}, inplace=True)
        return df, pedra_stats

    return df


# Identificar alunos que saíram
def identify_saiu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica e marca os alunos que saíram da instituição, baseando-se na sua presença ou ausência nos registros
    de anos subsequentes. A função adiciona uma coluna 'saiu' ao DataFrame, marcando com True apenas o registro do
    último ano em que o aluno foi registrado antes de sair.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo as colunas 'ano', 'nome' e 'fase', representando, respectivamente,
      o ano de registro, o nome do aluno e a fase de ensino em que o aluno estava inscrito.

    Retorna:
    - pd.DataFrame: O DataFrame original com uma coluna adicional 'saiu', indicando com um valor booleano
      (True ou False) se aquele registro corresponde ao último ano de presença do aluno antes de sair.

    Processo:
    1. Inicializa a coluna 'saiu' com False para todos os registros.
    2. Itera sobre os anos registrados no DataFrame, exceto o último ano, para identificar alunos presentes
       em um ano, mas ausentes no ano seguinte.
    3. Marca como True a coluna 'saiu' para o registro do último ano em que o aluno foi identificado antes de sua ausência.

    Nota:
    - A função assume que o DataFrame está completo em termos de registros anuais para cada aluno e que um aluno
      ausente em um ano subsequente ao seu último registro é considerado como tendo saído.
    """
    # Inicializa a coluna 'saiu' com False
    df['saiu'] = False
    
    # Extrai nomes únicos de alunos por ano
    students_by_year = {year: set(df[df['ano'] == year]['nome']) for year in df['ano'].unique()}
    
    for year in sorted(students_by_year.keys())[:-1]:  # Exclui o último ano
        students_this_year = students_by_year[year]
        students_next_year = students_by_year.get(year + 1, set())
        
        # Alunos neste ano, mas não no próximo
        students_left = students_this_year - students_next_year
        
        # Marca os alunos que saíram apenas no ano em que saíram
        df.loc[(df['nome'].isin(students_left)) & (df['ano'] == year), 'saiu'] = True
    
    return df


def identify_outliers(df: pd.DataFrame, indicators: List[str] = ['inde', 'ian', 'iaa', 'ida', 'ieg', 'ipp', 'ipv'], generate_dataframe: bool = False) -> pd.DataFrame:
    """
    Caulcula outliers através do intervalo interquartil dos indicadores, agregados por fase e ano.
    Adiciona colunas 'atencao' e 'destaque' para identificar alunos outliers em 2 ou mais indicadores.
    Se solicitado, separa os registros outliers em um DataFrame separado.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados.
    - indicators (List[str]): Lista de colunas indicadoras para as quais as estatísticas serão calculadas.
      Por padrão, inclui 'inde', 'ian', 'iaa', 'ida', 'ieg', 'ipp', 'ipv'.
    - generate_dataframe (bool): Se True, retorna um DataFrame separado com os registros outliers.

    Retorna:
    - pd.DataFrame: DataFrame original com as colunas 'atencao' e 'destaque' adicionadas para identificar outliers.
    - pd.DataFrame (opcional): DataFrame separado com os registros outliers, se generate_dataframe for True.
    """    
    stats_df = pd.DataFrame()
    outliers_df = pd.DataFrame()

    # Gerar dataframe com estatísticas para cada indicador
    for indicator in indicators:
        # Calcula estatísticas descritivas básicas
        indicator_stats = df.groupby(['fase', 'ano'])[indicator].describe().reset_index()
        # Calcula a moda para cada grupo
        moda = df.groupby(['fase', 'ano'])[indicator].agg(lambda x: x.mode().values[0] if not x.mode().empty else None).reset_index(name='moda')
        indicator_stats = pd.merge(indicator_stats, moda, on=['fase', 'ano'])
        # Renomeia as colunas para nomes mais descritivos
        indicator_stats.columns = ['fase', 'ano', 'contagem', 'media', 'dp', 'min', 'p25', 'mediana', 'p75', 'max', 'moda']
        # Calcula o intervalo interquartil (IQR)
        indicator_stats['iqr'] = indicator_stats['p75'] - indicator_stats['p25']
        # Calcula os limites para detecção de outliers
        indicator_stats['limite_inferior'] = indicator_stats['p25'] - 1.5 * indicator_stats['iqr']
        indicator_stats['limite_superior'] = indicator_stats['p75'] + 1.5 * indicator_stats['iqr']
        # Adiciona uma coluna para identificar o indicador
        indicator_stats['indicador'] = indicator
        # Reordena as colunas para ter 'indicador' como primeira coluna, 'ano' como segunda e 'fase' como terceira
        cols = indicator_stats.columns.tolist()
        cols = cols[-1:] + cols[1:2] + cols[0:1] + cols[2:-1]
        indicator_stats = indicator_stats[cols]
        # Concatena os resultados com o DataFrame de resultados
        stats_df = pd.concat([stats_df, indicator_stats], ignore_index=True)

        # Gerar dataframe com outliers, se solicitado
        for _, row in indicator_stats.iterrows():
            # Filtra os outliers para o indicador, fase e ano específicos
            outliers = df[(df['fase'] == row['fase']) & (df['ano'] == row['ano']) & ((df[indicator] < row['limite_inferior']) | (df[indicator] > row['limite_superior']))].copy()
            # Adiciona uma coluna para identificar o indicador
            outliers['indicador_outlier'] = indicator
            # Outlier pode ser inferior ou superior
            outliers['outlier_tipo'] = np.where(outliers[indicator] < row['limite_inferior'], 'inferior', 'superior')
            # Adiciona uma coluna para identificar se é um outlier inferior ou superior
            outliers_df = pd.concat([outliers_df, outliers], ignore_index=True)
            
        # Localiza em df alunos que aparecem em outliers_df mais de uma vez como outlier do tipo inferior e adiciona False se não houver
        df['atencao'] = df['nome'].map(outliers_df[outliers_df['outlier_tipo'] == 'inferior'].groupby('nome', observed=False)['indicador_outlier'].count() >= 2)
        df['atencao'] = np.where(df['atencao'].isna(), False, df['atencao'])
         # O mesmo para o tipo superior
        df['destaque'] = df['nome'].map(outliers_df[outliers_df['outlier_tipo'] == 'superior'].groupby('nome', observed=False)['indicador_outlier'].count() >= 2)
        df['destaque'] = np.where(df['destaque'].isna(), False, df['destaque'])

    if generate_dataframe:
        return df, outliers_df
    else:
        return df


def set_dtypes(df):
    """
    Ajusta os tipos de dados de um DataFrame.

    Parâmetros:
    - df (pd.DataFrame): DataFrame de entrada.

    Retorna:
    - pd.DataFrame: DataFrame com os tipos de dados ajustados.
    """
    df.replace(['nan', 'NAN', '#NULO!', 'ERRO'], np.nan, inplace=True)
    df.infer_objects(copy=False)
    
    col_types = {
        'nome': 'category',
        'instituicao_ensino_aluno': 'category',
        'idade_aluno': 'int8',
        'anos_pm': 'int8',
        'fase': 'int8',
        'turma': 'category',
        'grupo_fase': 'category',
        'ponto_virada': 'bool',
        'inde': 'float64',
        'inde_conceito': 'category',
        'pedra': 'category',
        'destaque_ieg': 'category',
        'destaque_ida': 'category',
        'destaque_ipv': 'category',
        'iaa': 'float64',
        'ieg': 'float64',
        'ips': 'float64',
        'ida': 'float64',
        'ipp': 'float64',
        'ipv': 'float64',
        'ian': 'float64',
        'rec_equipe_1': 'category',
        'rec_equipe_2': 'category',
        'rec_equipe_3': 'category',
        'rec_equipe_4': 'category',
        'defasagem': 'int8',
        'bolsista': 'bool',
        'cg': 'int',
        'cf': 'int',
        'ct': 'int',
        'nota_port': 'float64',
        'nota_mat': 'float64',
        'nota_ing': 'float64',
        'qtd_aval': 'int8',
        'saiu': 'bool',
        'atencao': 'bool',
        'destaque': 'bool'
    }
    
    for col, col_type in col_types.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(col_type, errors='ignore')
            except Exception as e:
                print(f"Erro ao converter coluna {col}: {e}")
    
    return df


def generate_stats(df: pd.DataFrame, indicators: list = ['inde', 'ida', 'ian', 'iaa', 'ips', 'ipp', 'ieg', 'ipv'], aggregate: str = None) -> pd.DataFrame:
    """
    Calcula estatísticas (média e desvio padrão) para indicadores numéricos especificados,
    agregando os dados por ano e, opcionalmente, por um segundo critério de agrupamento.
    
    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados.
    - indicators (list): Lista de strings com os nomes das colunas dos indicadores numéricos.
    - aggregate (str): String opcional que especifica um segundo critério de agrupamento além do ano.
    
    Retorna:
    - pd.DataFrame: DataFrame com as estatísticas calculadas.
    
    A função calcula 'pontos_perdidos' como a diferença da média do ano atual para o ano anterior,
    e 'variacao' como a variação percentual da média do ano atual em relação ao ano anterior,
    para cada grupo específico definido pelos critérios de agregação.
    """
    # Ano é sempre um critério de agregação
    years = df['ano'].unique()
    aggregation_levels = [list(df.ano.unique())]
    # Se houver um segundo critério de agregação, adicioná-lo à lista
    if aggregate:
        df[aggregate] = df[aggregate].astype(str)
        additional_groups = list(df[aggregate].unique())
        aggregation_levels.append(additional_groups)
    # Lista para armazenar os dicionários com as estatísticas
    stats_list = []
    # Iterar sobre todas as combinações de ano, grupo e indicador
    for elements in product(*aggregation_levels, indicators):
        year = elements[0]
        # Se houver um segundo critério de agregação, definir o grupo e o indicador
        if aggregate:
            group = elements[1]
            indicator = elements[2]
            # Criar string de consulta para filtrar o DataFrame
            query_str = f'ano == {year} and {aggregate} == "{group}"'
        # Caso contrário, definir o indicador e criar a string de consulta apenas com o ano 
        else:
            group = None
            indicator = elements[1]
            query_str = f'ano == {year}'
        # Dicionário para armazenar as estatísticas
        stats = {'ano': year, 'indicador': indicator}
        # Se houver um segundo critério de agregação, adicionar o grupo ao dicionário
        if aggregate:
            stats[aggregate] = group
        # Calcular a contagem de valores não nulos
        stats['contagem'] = df.query(query_str)[indicator].count()
        # Calcular a média
        stats['media'] = df.query(query_str)[indicator].mean()
        # Calcular o desvio padrão
        stats['desvio_padrao'] = df.query(query_str)[indicator].std()
        # Média menos 1 dp
        stats['media_menos_1dp'] = stats['media'] - stats['desvio_padrao']
        # Média mais 1 dp
        stats['media_mais_1dp'] = stats['media'] + stats['desvio_padrao']
        # Mínimo
        stats['minimo'] = df.query(query_str)[indicator].min()
        # Máximo
        stats['maximo'] = df.query(query_str)[indicator].max()
        # Calcular o percentil 25
        stats['percentil_25'] = df.query(query_str)[indicator].quantile(0.25)
        # Calcular a mediana
        stats['mediana'] = df.query(query_str)[indicator].median()
        # Calcular o percentil 75
        stats['percentil_75'] = df.query(query_str)[indicator].quantile(0.75)
        # IQR
        stats['iqr'] = stats['percentil_75'] - stats['percentil_25']
        # Limite inferior
        stats['limite_inferior'] = stats['percentil_25'] - 1.5 * stats['iqr']
        # Limite superior
        stats['limite_superior'] = stats['percentil_75'] + 1.5 * stats['iqr']
        # Moda
        stats['moda'] = df.query(query_str)[indicator].mode().values[0] if not df.query(query_str)[indicator].mode().empty else None
        # Adicionar o dicionário à lista
        stats_list.append(stats)
    # Criar DataFrame a partir da lista de dicionários
    stats_df = pd.DataFrame(stats_list)
    # Ordenar o DataFrame por ano e, em seguida, por critério de agregação, se houver
    sort_columns = ['ano']
    if aggregate:
        sort_columns.insert(1, aggregate)
    stats_df.sort_values(by=sort_columns, inplace=True)
    # Calcular pontos perdidos e variação percentual
    group_columns = ['indicador']
    if aggregate:
        group_columns.append(aggregate)
    stats_df['pontos_perdidos'] = stats_df.groupby(group_columns)['media'].diff()
    stats_df['variacao_percentual'] = stats_df.groupby(group_columns)['media'].pct_change(fill_method=None)
    # Remover linhas com valores nulos em todas as colunas
    stats_df.dropna(subset=['media', 'desvio_padrao', 'media_menos_1dp', 'media_mais_1dp', 
                            'minimo', 'maximo', 'percentil_25', 'mediana', 'percentil_75', 'iqr', 'limite_inferior', 
                            'limite_superior', 'moda', 'pontos_perdidos', 'variacao_percentual'], how='all', inplace=True)
    return stats_df

#                                [   Plot  Data   ]                                #


# Gráfico de barras para média dos indicadores em um ano ou em todos os anos
def plot_media_indicadores(df: pd.DataFrame, ano: int = None, indicadores: list = ['INDE', 'IDA', 'IAN', 'IAA', 'IEG', 'IPP', 'IPV'], display_in_streamlit=False):
    """
    Plota um gráfico de barras com a média dos indicadores selecionados para um ano específico ou para todos os anos,
    caso nenhum ano seja especificado.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados de desempenho dos alunos.
    - ano (int, opcional): Ano específico para o qual os indicadores serão plotados. Se None, compara todos os anos.
    - display_in_streamlit (bool, opcional): Se True exibe no Streamlit, se False exibe no console.

    A função ajusta o DataFrame para substituir valores booleanos por 'Sim' e 'Não', calcula a média dos indicadores
    para o ano especificado ou para todos os anos, e plota um gráfico de barras com essas médias, utilizando cores
    distintas para cada indicador.
    """

    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.upper()
    df_temp = df_temp.replace({True: 'Sim', False: 'Não'})
    
    # Cores para os indicadores
    cores = px.colors.qualitative.Pastel

    # Filtrando o DataFrame para o ano de interesse ou para todos os anos
    if ano is not None:
        df_temp = df_temp[df_temp['ANO'] == ano]
        df_media = df_temp[indicadores].mean().reset_index()
        df_media.columns = ['Indicador', 'Média']
        titulo = f"Indicadores de Desempenho para o Ano de {ano}"
        fig = px.bar(df_media, x='Indicador', y='Média', color='Indicador', color_discrete_sequence=cores)
    # Comparando os indicadores para todos os anos
    else:
        df_temp['ANO'] = df_temp['ANO'].astype(str)
        df_melted = df_temp.melt(id_vars=['ANO'], value_vars=indicadores, var_name='Indicador', value_name='Valor')
        df_media = df_melted.groupby(['ANO', 'Indicador']).mean().reset_index()
        titulo = "Comparação dos Indicadores de Desempenho por Ano"
        fig = px.bar(df_media, x='Indicador', y='Valor', color='ANO', barmode='group', category_orders={"Indicador": indicadores}, color_discrete_sequence=cores)
    
    # Atualizando o layout para melhorar a visualização e definir o tema escuro
    fig.update_layout(title=titulo, 
                      width=800, 
                      height=500, 
                      xaxis_title='Indicadores', 
                      yaxis_title='Média', 
                      template="plotly_dark")
    # Salvando o gráfico em um arquivo HTML ou exibindo na tela
    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()


# Boxplot ou Violinplot para indicadores com agregação
def plot_percentis_indicador(df: pd.DataFrame, indicador: str, agregacao='ano', plot_type='boxplot', display_in_streamlit=False):
    """
    Plota os percentis de todos os alunos para todos os anos disponíveis no DataFrame, 
    para um indicador específico, utilizando um gráfico boxplot ou violinplot do Plotly,
    conforme escolhido pelo usuário. Agora, ordena o eixo X de forma ascendente para qualquer agregação.

    Parâmetros:
- df (pd.DataFrame): DataFrame contendo os dados de desempenho dos alunos.
- indicador (str): Nome do indicador de desempenho para o qual o desempenho será plotado.
- agregacao (str): Coluna pela qual os dados serão agregados ('ano', 'pedra', 'grupo_fase', 'fase').
- plot_type (str): Tipo de plot ('boxplot' ou 'violinplot').
- display_in_streamlit (bool): Se True exibe no Streamlit, se False exibe no console.
"""

    df_temp = df.copy()
    df_temp.columns = df_temp.columns.str.upper()
    # Substituindo valores de df_temp por Sim/Não
    df_temp = df_temp.replace({True: 'Sim', False: 'Não'})
    indicador = indicador.upper()
    # agg as string
    agregacao = agregacao.upper()

    if agregacao not in df_temp.columns:
        raise ValueError(f"Agregação {agregacao} não encontrada no DataFrame.")

    # Ordenando os valores de agregação de forma ascendente
    valores_agregacao = sorted(df_temp[agregacao].unique())

    # Inicializando a figura do Plotly
    fig = go.Figure()

    cores = px.colors.qualitative.Pastel_r

    for i, valor in enumerate(valores_agregacao):
        df_filtrado = df_temp[df_temp[agregacao] == valor]
        cor = cores[i % len(cores)]
        if plot_type == 'boxplot':
            fig.add_trace(go.Box(y=df_filtrado[indicador], name=str(valor), marker_color=cor))
        elif plot_type == 'violinplot':
            fig.add_trace(go.Violin(y=df_filtrado[indicador], name=str(valor), box_visible=True, meanline_visible=True, line_color=cor))


    # Atualizando o layout para melhorar a visualização e definir o tema escuro
    fig.update_layout(
        title=f"Distribuição do {indicador} por {agregacao.replace('_', ' ').title()}",
        showlegend=False,
        width=800,
        height=500,
        xaxis_title=agregacao.replace('_', ' ').lower(),
        yaxis_title=indicador,
        template="plotly_dark"
    )

    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()
        
# Scatterplot 2D ou 3D para visualizar a variação de 2 ou 3 indicadores
def plot_scatter(df: pd.DataFrame, x: str, y: str, z: str = None, display_in_streamlit=False):
    """
    Plota um scatterplot 2D ou 3D para visualizar a variação de 2 ou 3 indicadores escolhidos pelo usuário,
    em um ano específico ou por todos os anos disponíveis, com o ano como hue se não especificado.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados.
    - x (str): Nome da primeira variável (indicador) a ser plotada.
    - y (str): Nome da segunda variável (indicador) a ser plotada.
    - z (str, opcional): Nome da terceira variável (indicador) a ser plotada. Se None, plota um scatterplot 2D.
    - display_in_streamlit (bool): Se True, exibe o gráfico no Streamlit. Se False, exibe o gráfico no console.
    """
    x, y, z = x.lower(), y.lower(), z.lower() if z else None
    cores = px.colors.qualitative.Pastel
    df['ano'] = df['ano'].astype(str)  # Garantir que ano é uma string para a legenda
    color = 'ano'  # Usa o ano como hue se nenhum ano específico é fornecido

    # Plotar scatterplot 2D ou 3D dependendo da presença da variável z
    if z is None:
        # Scatterplot 2D
        fig = px.scatter(df, x=x, y=y, color=color, 
                         title=f"Scatterplot 2D de {x.upper()} vs {y.upper()}", labels={x: x.upper(), y: y.upper()}, 
                         color_discrete_map={str(year): cor for year, cor in zip(df['ano'].unique(), cores)})
    else:
        # Scatterplot 3D com cores Pastel
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, 
                            title=f"Scatterplot 3D de {x.upper()}, {y.upper()} e {z.upper()}", 
                            labels={x: x.upper(), y: y.upper(), z: z.upper()}, 
                            color_discrete_map={str(year): cor for year, cor in zip(df['ano'].unique(), cores)})

    fig.update_layout(showlegend=True,
                      height=800, 
                      width=800,
                      template="plotly_dark")

    # Exibir gráfico no Streamlit ou no console
    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()


# Heatmap de correlação entre os indicadores
def plot_correlation_heatmap(df, indicadores=['INDE', 'IAN', 'IAA', 'IDA', 'IEG', 'IPP', 'IPV'], ano=None, display_in_streamlit=False):
    """
    Plota um heatmap de correlação entre as variáveis selecionadas pelo usuário ou todas as variáveis,
    para um ano específico ou todos os anos, utilizando a biblioteca Plotly.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados.
    - indicadores (list, opcional): Lista de colunas (indicadores) para calcular a correlação. Se None, usa todas as colunas.
    - ano (int, opcional): Ano específico para filtrar os dados. Se None, considera todos os dados disponíveis.
    - display_in_streamlit (bool, opcional): Se True, exibe o gráfico no Streamlit. Se False, exibe no Jupyter Notebook.

    Exemplo de uso:
    >>> plot_correlation_heatmap(df, ['INDE', 'IAN', 'IAA'], 2020)
    """
    # Criando um DataFrame temporário
    df_temp = df.copy()

    # Convertendo os nomes das colunas para maiúsculas
    df_temp.columns = df_temp.columns.str.upper()

    # Filtrando por ano, se fornecido
    if ano is not None:
        df_temp = df_temp[df_temp['ANO'] == ano]

    # Selecionando colunas indicadas pelo usuário ou todas se não especificado
    if indicadores is not None:
        # Garantindo que os nomes das colunas estejam em maiúsculas
        indicadores = [indicador.upper() for indicador in indicadores]
        df_temp = df_temp[indicadores]  # Removido 'ANO' da seleção para evitar erro
    else:
        # Se não selecionar colunas, considera todas as colunas numéricas
        df_temp = df_temp.select_dtypes(include=['number'])

    # Calculando a matriz de correlação
    correlation_matrix = df_temp.corr()

    # Plotando o heatmap de correlação usando Plotly Graph Objects para maior customização
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                    x=correlation_matrix.index,
                                    y=correlation_matrix.columns,
                                    colorscale='rdbu',
                                    zmin=-1,
                                    zmax=1,
                                    textfont=dict(size=14, weight='bold'),
                                    texttemplate="%{z:.2f}",
                                    hoverinfo="none"))

    # Customizando a exibição dos nomes das colunas e valores em negrito
    fig.update_xaxes(tickfont=dict(size=12, color='white', weight='bold'))
    fig.update_yaxes(tickfont=dict(size=12, color='white', weight='bold'))

    fig.update_layout(title_text='Correlação entre Indicadores',
                      title_font=dict(size=20, color='white', weight='bold'),
                      template='plotly_dark',
                      height=800,
                      width=800)

    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()



# Plotando os destaques para um indicador e ano específicos
def plot_destaque_alunos(df: pd.DataFrame, indicador: str, ano: int, display_in_streamlit=False, remove_zero: bool = True):
    """
    Plota dois subplots com os 10 alunos destaques positivos e negativos baseado em um indicador e ano específicos.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados dos alunos.
    - indicador (str): Indicador específico para análise.
    - ano (int): Ano específico para análise.
    - display_in_streamlit (bool): Se True exibe no Streamlit, se False exibe no console.
    - remove_zero (bool): Se True, remove alunos com valores zero dos destaques negativos.
    """
    df_temp = df.copy()
    
    # Ajustando o nome das colunas para maiúsculas
    df_temp.columns = df_temp.columns.str.upper()

    # Filtrando o DataFrame pelo ano e indicador especificados
    df_filtrado = df_temp.query(f"ANO == {ano}")[['NOME', indicador.upper()]]
    
    # Removendo valores nulos
    df_filtrado = df_filtrado.dropna(subset=[indicador.upper()])
    
    # if remove zero, remove alunos com valores zero
    if remove_zero:
        df_filtrado = df_filtrado[df_filtrado[indicador.upper()] != 0]
    
    # Ordenando os alunos pelos valores do indicador
    top_positivos = df_filtrado.sort_values(by=indicador.upper(), ascending=False).head(10)
    top_negativos = df_filtrado.sort_values(by=indicador.upper(), ascending=True).head(10)
    
    # se todos os valores negativos forem zero, não plotar
    if top_negativos[indicador.upper()].sum() == 0 and remove_zero:
        top_negativos = top_negativos.iloc[1:]
    
    # Criando subplots com títulos ajustados para "+" e "-"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("+", "-"))
    
    # Adicionando os gráficos de barra para cada subplot
    fig.add_trace(go.Bar(x=top_positivos[indicador.upper()], 
                         y=top_positivos['NOME'], 
                         orientation='h', 
                         marker_color=px.colors.qualitative.Pastel[0]),
                  row=1, col=1)

    fig.add_trace(go.Bar(x=top_negativos[indicador.upper()], 
                         y=top_negativos['NOME'], 
                         orientation='h', 
                         marker_color=px.colors.qualitative.Pastel[2]), 
                  row=1, col=2)
    
    # Atualizando o layout para ambos os subplots
    fig.update_layout(height=600, 
                      width=1000, 
                      showlegend=False,
                      title_text=f"Alunos Destaques de {ano} - {indicador.upper()}", 
                      template="plotly_dark")

    # Atualizando configurações de eixo x para ambos os subplots
    fig.update_xaxes(showline=False, showticklabels=False, showgrid=False, zeroline=False, range=[0, 10], row=1, col=1)
    fig.update_xaxes(showline=False, showticklabels=False, showgrid=False, zeroline=False, range=[0, 10], row=1, col=2)
    
    # Exibindo o gráfico
    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()




# Plotar indicadores do aluno - gráfico de radar
def plot_radar_aluno(df: pd.DataFrame, aluno: str, display_in_streamlit=False):
    """
    Plota o desempenho de um aluno específico em comparação com a média da sua fase em diversos indicadores,
    utilizando um gráfico radar do Plotly.

    A função realiza os seguintes passos:
    1. Filtra o DataFrame para o aluno específico, usando o nome do aluno como critério.
    2. Converte os nomes das colunas do DataFrame para letras maiúsculas para padronização.
    3. Identifica a fase máxima do aluno, assumindo que um aluno pode ter registros em múltiplas fases.
    4. Define os indicadores de desempenho a serem analisados.
    5. Obtém os anos únicos presentes nos registros do aluno para iterar sobre eles.
    6. Calcula a média dos indicadores de desempenho para todos os alunos da mesma fase.
    7. Inicializa uma figura do Plotly para o gráfico radar.
    8. Adiciona um traço no gráfico para representar a média dos indicadores da fase do aluno.
    9. Adiciona um traço para cada ano de registro do aluno, permitindo a visualização da evolução do seu desempenho.
    10. Atualiza o layout do gráfico para melhorar a visualização, incluindo a definição de um tema escuro.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados de desempenho dos alunos. Deve incluir, no mínimo,
      colunas para o nome do aluno, a fase, o ano e os indicadores de desempenho especificados.
    - aluno (str): Nome do aluno para o qual o desempenho será plotado.
    - display_in_streamlit (bool): Se True, exibe o gráfico no Streamlit. Se False, exibe o gráfico no console.

    O gráfico resultante permite a comparação visual do desempenho do aluno em relação à média da sua fase,
    destacando áreas de força e oportunidades de melhoria.
    """
    
    # Filtrando o DataFrame para o aluno específico
    aluno_df = df.query(f'nome == "{aluno}"').copy()
    aluno_df.columns = aluno_df.columns.str.upper()
    fase_aluno = aluno_df['FASE'].max()

    indicadores = ['INDE', 'IAN', 'IAA', 'IDA', 'IEG', 'IPP', 'IPV']

    # Obtendo os anos únicos para iterar
    anos = aluno_df['ANO'].unique()

    # média dos indicadores para o aluno da mesma fase
    fase_df = df.query('fase == @fase_aluno')
    fase_df.columns = fase_df.columns.str.upper()
    fase_df = fase_df.groupby('ANO')[indicadores].mean().reset_index()

    # Inicializando a figura do Plotly
    fig = go.Figure()

    # Cores para os traços
    cores = ['#E6C145', '#EA71B0', '#0BA5FF', '#4FE88B']

    # Adicionando um traço para a média dos indicadores da fase
    valores_fase = [fase_df[f].values[0] for f in indicadores]
    fig.add_trace(go.Scatterpolar(
        r=valores_fase,
        theta=indicadores,
        fill='toself',
        name=f'Média da Fase {fase_aluno}',
        line=dict(color=cores[0])
    ))

    # Adicionando um traço para cada ano
    for i, ano in enumerate(anos):
        df_ano = aluno_df[aluno_df['ANO'] == ano]
        valores = [df_ano[ind].values[0] for ind in indicadores]
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=indicadores,
            fill='toself',
            name=f'Ano {ano}',
            line=dict(color=cores[(i + 1) % len(cores)])
        ))

    # Atualizando o layout para melhorar a visualização e definir o tema escuro
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title=f"Desempenho do Aluno {aluno}",
        showlegend=True,
        width=800,
        height=700,
        font=dict(size=14),
        template="plotly_dark"
    )
    
    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()

# Plotar evolução do indicador ao lindo dos anos
def plot_aluno_indicador(df, aluno, indicador, display_in_streamlit=False):
    """
    Plota um gráfico de barras do desempenho de um aluno específico em um único indicador, utilizando a biblioteca Plotly,
    tratando o ano como string mas garantindo sua ordenação correta no eixo.

    A função realiza os seguintes passos:
    1. Filtra o DataFrame para o aluno específico, usando o nome do aluno como critério.
    2. Converte os nomes das colunas do DataFrame para letras maiúsculas para padronização.
    3. Converte o ano para string para garantir que seja tratado corretamente como categoria.
    4. Ordena os dados do aluno pelo ano, garantindo uma visualização temporal correta.
    5. Verifica se existem resultados não nulos para o aluno no indicador especificado.
    6. Plota um gráfico de barras para o indicador, com uma barra para cada ano disponível nos dados do aluno.
    7. Atualiza o layout do gráfico para melhorar a visualização, incluindo a definição de um tema escuro e ajustes na legenda e título.
    8. Permite ao usuário escolher entre salvar o gráfico como um arquivo HTML ou exibi-lo diretamente.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados de desempenho dos alunos. Deve incluir, no mínimo,
      colunas para o nome do aluno, o ano e os indicadores de desempenho especificados.
    - aluno (str): Nome do aluno para o qual o desempenho será plotado.
    - indicador (str): Nome do indicador de desempenho a ser plotado.
    - display_in_streamlit (bool): Se True, exibe o gráfico no Streamlit. Se False, exibe o gráfico no console.

    Exemplo de uso:
    >>> plot_aluno_indicador(df, "ALUNO-1", "INDE", display_in_streamlit=True)
    """
    # Filtrando o DataFrame para o aluno específico e padronizando os nomes das colunas
    aluno_df = df.query(f'nome == "{aluno}"').copy()
    aluno_df.columns = aluno_df.columns.str.upper()

    # Convertendo o ano para string e ordenando
    aluno_df['ANO'] = aluno_df['ANO'].astype(str)
    aluno_df.sort_values('ANO', inplace=True)

    # Inicializando a figura do Plotly
    fig = go.Figure()

    cores = px.colors.qualitative.Pastel_r
    
    # Verificando se existem resultados não nulos e plotando o indicador
    for ano in aluno_df['ANO'].unique():
        df_ano = aluno_df[aluno_df['ANO'] == ano]
        if not df_ano[indicador].isnull().all():  # Verifica se existem resultados não nulos
            fig.add_trace(go.Bar(x=[ano], 
                                 y=[df_ano[indicador].values[0]], 
                                 name=f'Ano {ano}', 
                                 marker=dict(color=cores[int(ano) % len(cores)]))
                          )
    # Atualizando o layout do gráfico
    fig.update_layout(
        title=f"Desempenho do Aluno {aluno} em {indicador}",
        showlegend=True,
        width=800,
        height=500,
        font=dict(size=14),
        template="plotly_dark",
        barmode='group',
        xaxis=dict(type='category'),
        yaxis=dict(range=[0, 10])
    )
    # Escolha entre salvar o gráfico como HTML ou exibi-lo diretamente
    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()
    
#  Plotar variação do aluno em relação ao ano anterior ou à média da turma
def plot_aluno_variacao(df: pd.DataFrame, aluno: str, indicador: str, display_in_streamlit=False):
    """
    Função ajustada para plotar a variação percentual do desempenho de um aluno em um indicador específico ao longo dos anos.
    - Caso exista apenas um ano de dados, compara o desempenho do aluno com a média da turma usando um gráfico de barras.
    - Caso existam dois anos de dados, utiliza um gráfico de barras para mostrar a variação entre os dois anos.
    - Caso existam três ou mais anos, utiliza um gráfico de linhas para mostrar a variação percentual ao longo dos anos.

    Parâmetros:
    - df (pd.DataFrame): DataFrame contendo os dados de desempenho.
    - aluno (str): Nome do aluno.
    - indicador (str): Indicador de desempenho a ser analisado.
    - display_in_streamlit (bool): Se True exibe no Streamlit, se False exibe no console.
    """
    # Filtrar DataFrame para o aluno específico
    indicador = indicador.lower()
    aluno_df = df.query(f'nome == "{aluno}"')[['ano', indicador]].copy()
    aluno_df.sort_values('ano', inplace=True)

    # Inicializar figura Plotly
    fig = go.Figure()

    unique_years = len(aluno_df['ano'].unique())

    if unique_years == 1:
        # Comparar com a média da turma se houver apenas um ano
        ano = aluno_df['ano'].iloc[0]
        media_ano = df.query(f'ano == {ano}')[indicador].mean()
        variacao = (aluno_df[indicador].iloc[0] - media_ano) / media_ano * 100
        min_range = min(aluno_df[indicador].iloc[0], media_ano) - 10
        max_range = max(aluno_df[indicador].iloc[0], media_ano) + 10
        fig.add_trace(go.Bar(x=[ano], y=[variacao], name=f'Variação de {indicador.upper()}'))
    elif unique_years == 2:
        # Plotar variação com um gráfico de barras para dois anos
        aluno_df['Variação'] = aluno_df[indicador].pct_change() * 100
        min_range = aluno_df['Variação'].min() - 10
        max_range = aluno_df['Variação'].max() + 10
        fig.add_trace(go.Bar(x=aluno_df['ano'].iloc[1:], y=aluno_df['Variação'].iloc[1:], name=f'Variação de {indicador.upper()}'))
    else:
        # Plotar variação com um gráfico de linhas para três ou mais anos
        aluno_df['Variação'] = aluno_df[indicador].pct_change() * 100
        # Ignorar a primeira linha, que terá um valor NaN
        aluno_df = aluno_df.iloc[1:]
        # Se melhorou ou piorou em relação ao ano anterior
        min_range = aluno_df['Variação'].min() - 10
        max_range = aluno_df['Variação'].max() + 10
        fig.add_trace(go.Scatter(x=aluno_df['ano'], y=aluno_df['Variação'], mode='lines+markers', name=f'Variação de {indicador.upper()}'))

    # Atualizar layout do gráfico
    fig.update_layout(
        title=f"Variação do {indicador.upper()} - {aluno}",
        showlegend=True,
        width=800,
        height=500,
        font=dict(size=14),
        template="plotly_dark",
        xaxis=dict(title='', type='category', showgrid=False),
        yaxis=dict(title='Variação (%)', range=[min_range, max_range])
    )
    # Escolha entre salvar o gráfico como HTML ou exibi-lo diretamente
    if display_in_streamlit:
        st.plotly_chart(fig)
    else:
        fig.show()


#                                [   Streamplit Functions   ]                                #

def save_dfs_to_excel(**dataframes):
    """
    Salva um ou mais DataFrames em um arquivo Excel e retorna o conteúdo do arquivo.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name)
    output.seek(0)  # Move o cursor para o início do buffer
    return output.getvalue()  # Retorna o conteúdo do buffer

def footer_section():
    # área de contato
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('---')
    # texto -> Agradecimentos
    st.markdown('''
                <p style="font-size: 18px; text-align: center;">
                    Obrigado por acompanhar este projeto! 🚀
                <br></p>''', unsafe_allow_html=True)
    linkedin = 'https://www.linkedin.com/in/viniplima/'
    github = 'https://github.com/euvina/'
    mail = 'pradolimavinicius@gmail.com'
    subject = 'Contato via Streamlit - Projeto Passos Mágicos'
    st.markdown('''
        <p style="font-size: 18px; text-align: center;">
        📧 Entre em contato:<br>
            <a href="mailto:{}?subject={}">
                <img src="https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=Gmail&logoColor=white" alt="Gmail">
            </a>
            <a href="{}">
                <img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white" alt="GitHub">
            </a>
            <a href="{}">
                <img src="https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=LinkedIn&logoColor=white" alt="LinkedIn">
            </a>
        </p>'''.format(mail, subject, github, linkedin), unsafe_allow_html=True)


def download_section(df_cleaned, df_to_verify, df_removed, df_stats, df_outliers):
    dataframes = {
        "Limpos": df_cleaned,
        "Verificar": df_to_verify,
        "Removidos": df_removed,
        "Estatísticas": df_stats,
        "Outliers": df_outliers
    }
    # Salvar os DataFrames em um arquivo Excel
    excel_file = save_dfs_to_excel(**dataframes)
    
    # Adicionar botão de download
    st.download_button(label="Baixar dados",
                    data=excel_file,
                    file_name="passos_magicos_base.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
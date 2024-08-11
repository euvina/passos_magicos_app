# Imports
import streamlit as st
import pandas as pd
import time
from PIL import Image
from io import BytesIO
import time

from utils import *

st.set_page_config(page_title="Passos Mágicos - Análise", page_icon="🧙‍♂️", layout="centered")

# Decorador para cache
@st.cache_data
def load_and_process_data(uploaded_file, recalculate_pedra=False):
    if uploaded_file is not None:
        df_input = read_data(uploaded_file)
    else:
        df_input = read_data("data/PEDE_PASSOS_DATASET_FIAP.csv")
    initial_shape = df_input.shape
    df_cleaned, df_to_verify, df_removed = clean_dataframe_separate_issues(df_input)
    df_cleaned = replace_yes_no(df_cleaned)
    df = create_df_all(df_cleaned)
    df = split_fase_turma(df)
    df = add_group_fase(df)
    df = adjust_indicators(df)
    df = calculate_inde(df) 
    df = dropna_inde(df)
    df = fill_pedra(df, recalculate=recalculate_pedra)
    df = identify_saiu(df)
    df, outliers = identify_outliers(df, generate_dataframe=True)
    df = set_dtypes(df)
    stats = generate_stats(df)
    final_shape = df.shape

    return df, df_to_verify, df_removed, stats, outliers, initial_shape, final_shape

def main():
    st.sidebar.title("Navegação")
    # Escolher página
    page = st.sidebar.radio("Ir para", ["Home", "Análise"])
    with st.expander("⚙️ **Dados do projeto**"):
        st.markdown("""#### 📤 Importar""")
        uploaded_file = st.file_uploader("*Carregue a base de dados em um arquivo no formato CSV.*", type="csv")
        recalculate_pedra = st.checkbox("recalcular informações de **pedra**")
        if uploaded_file is not None or "data_loaded" not in st.session_state:
            time.sleep(2)
            df, df_to_verify, df_removed, stats, outliers, initial_shape, final_shape = load_and_process_data(uploaded_file, recalculate_pedra)
            st.session_state.data_loaded = True
            st.session_state.df = df
            st.session_state.df_to_verify = df_to_verify
            st.session_state.df_removed = df_removed
            st.session_state.stats = stats
            st.session_state.outliers = outliers
            st.session_state.initial_shape = initial_shape
            st.session_state.final_shape = final_shape
        
        if "data_loaded" in st.session_state:
                st.success(f'**✅ Processamento concluído:** {st.session_state.initial_shape[1]} colunas x {st.session_state.initial_shape[0]} linhas ⭢ {st.session_state.final_shape[1]} colunas x **{st.session_state.final_shape[0]}** linhas')
                anos = st.session_state.df['ano'].unique()
                st.write(f"*Dados carregados com anos de {min(anos)} a {max(anos)}.*")
                # Exibir informações gerais
                st.markdown("#### 📥 Exportar")
                st.write("""
                            <p style="font-size: 14px">
                                A planilha reúne, em abas:
                                dados limpos, a serem verificados, removidos, estatísticas por ano e outliers.
                            </p>
                        """, unsafe_allow_html=True)
                
                download_section(st.session_state.df, st.session_state.df_to_verify, st.session_state.df_removed, st.session_state.stats, st.session_state.outliers)
                # Exibir informações gerais
                st.markdown("#### 🔎 Visualizar")
                # exibir dataset com 3 opções, df, df_to_verify, df_removed
                dataset = st.radio("Selecione o conjunto de dados a ser exibido:", ["limpos", "verificar", "removidos"])
                if dataset == "limpos":
                    # expandir
                    st.markdown("##### O que foi feito?")
                    st.write("""
                                <p style="font-size: 14px">
                                    - coluna <code>ano</code> adicionada através da nomenclatura de colunas (ex. <i>"INDE_<b>2021</b>"</i>).<br>
                                    - coluna <code>nome</code> agora pode apresentar valores duplicados, para informações de diferentes anos.<br>
                                    - colunas <code>turma</code> e <code>fase</code> mantidas separadas, com remoção de colunas desnecessárias.<br>
                                    - coluna <code>grupo_fase</code> adicionada para identificar alunos escolares (<code>Fases 0-7</code>) e universitários (<code>Fase 8</code>).<br>
                                    - coluna <code>saiu</code> adicionada para alunos que saíram da instituição.<br>
                                    - coluna <code>atencao</code> adicionada, identifica "atenção" para outliers negativos em relação a dois ou mais indicadores.<br>
                                    - coluna <code>destaque</code> adicionada: identifica "destaque" para outliers positivos em relação a dois ou mais indicadores.<br>
                                    - linhas inconsistentes removidas.<br>
                                    - valores <code>Sim</code> ou <code>Não</code> substituídos por <code>1</code> ou <code>0</code>, respectivamente.<br>
                                </p style="font-size: 10px">
                            """, unsafe_allow_html=True)
                    st.write(st.session_state.df)
                elif dataset == "verificar":
                    st.write("""
                                <p style="font-size: 14px">
                                    - registros com alguma informação relevante faltando ou inconsistente.<br>
                                    - dados mantidos no conjunto de dados limpos.<br>
                                </p style="font-size: 10px">
                            """, unsafe_allow_html=True)
                    st.write(st.session_state.df_to_verify)
                else:
                    st.write("""
                                <p style="font-size: 14px">
                                    - registros corrompidos ou fora do padrão de leitura.<br>
                                    - dados removidos do conjunto de dados limpos.<br>
                                </p style="font-size: 10px">
                            """, unsafe_allow_html=True)
                    st.write(st.session_state.df_removed)
                    
                    # separador
                    st.write("<br>", unsafe_allow_html=True)           

    if page == "Home":
        show_cover_page()
        footer_section()
    elif page == "Análise":
        show_analysis_page()
        footer_section()
    else:
        footer_section()
        pass

def show_cover_page():
    st.title("Passos Mágicos")
    st.write("""
                <p style="font-size: 16px">
                    A <a href="https://passosmagicos.org.br"><b>Passos Mágicos</b></a> é uma ONG fundada por Michelle Flues e Dimetri Ivanoff em 1992, no município de Embu-Guaçu — estado de São Paulo.
                    A associação transforma a vida de crianças e adolescentes através da Educação de qualidade, com apoio psicopedagógico. 
                    Aqui, a palavra Educação é escrita com “e” maiúsculo, pois vai além da instrução acadêmica — com programas estudantis, psicológicos e culturais.
                <br></p>
            """, unsafe_allow_html=True)
    st.subheader("✨ O projeto")
    
    st.write("""            
                Aplicação para automatizar a análise de dados da **Associação Passos Mágicos**.<br><br>
                **📍 Objetivos:** Facilitar a visualização dos dados de alunos, com foco em identificar padrões e outliers, além de auxiliar no desenvolvimento das Pesquisas Extensivas de Desenvolvimento Educacional (PEDE).<br>
                **📗 Artigo:** [Passos Mágicos: a Educação como Agente de Mudança](https://medium.com/@viniplima/passos-mágicos-a-educação-como-agente-de-mudança-6d1c01f9b3b5)<br>
                **🐈‍ Código:** [Repositório GitHub](https://github.com/euvina/passos_magicos_app)<br>
                **👨‍💻 Autor:** [Vinícius Prado Lima](https://github.com/euvina)<br>
            """, unsafe_allow_html=True)
    
    st.write("---")
    
    st.subheader("Estatísticas Gerais")
    if "data_loaded" in st.session_state:
        df = st.session_state.df
        stats = st.session_state.stats
        anos = df['ano'].unique()
        st.write(f"*Dados carregados com anos {min(anos)} a {max(anos)}.*")
      
        st.write("""
        **Índice de Desenvolvimento Educacional:**
        - INDE médio no último ano: **{:.2f}**
        - INDE médio no primeiro ano: **{:.2f}**
        - Variação entre primeiro e último ano: **{:.2f}%**
    """.format(
            stats.query("ano == @stats['ano'].max() and indicador == 'inde'").media.values[0],
            stats.query("ano == @stats['ano'].min() and indicador == 'inde'").media.values[0],
            ((stats.query("ano == @stats['ano'].max() and indicador == 'inde'").media.values[0] - stats.query("ano == @stats['ano'].min() and indicador == 'inde'").media.values[0]) / stats.query("ano == @stats['ano'].min() and indicador == 'inde'").media.values[0]) * 100
            )
    ) 
    st.write("<br>", unsafe_allow_html=True)
    st.write(stats)
    
def show_analysis_page():
    st.title("Análise de Dados")
    
    if "data_loaded" in st.session_state:
        df = st.session_state.df
        stats = st.session_state.stats
        anos = df['ano'].unique()
        indicadores = ['INDE', 'IAN', 'IAA', 'IDA', 'IEG', 'IPP', 'IPV']
        agrupamentos = {'Ano': 'ano',
                        'Grupo': 'grupo_fase',
                        'Fase': 'fase',
                        'Pedra': 'pedra',
                        'Ponto de Virada': 'ponto_virada',
                        'Atenção': 'atencao',
                        'Destaque': 'destaque'}
    
    dimension = st.selectbox(
    'Selecione a dimensão da análise:', ['Passos Mágicos', 'Alunos'])
    st.write("<br>", unsafe_allow_html=True)
    
    if dimension == 'Passos Mágicos':
        plot_media_indicadores(df, display_in_streamlit=True)
        st.write("""
                    - Indicador de destaque ao longo dos anos: **{}**, com média de {:.2f} no ano **{}**.
                    - Indicador de atenção ao longo dos anos: **{}**, com média de {:.2f} no ano **{}**.
                    - Indicador com maior variação absoluta: **{}**, com {:.2f}%.
                    - Maior INDE médio no ano {} | Menor INDE médio no ano {}
                    - Maior IDA médio no ano {} | Menor IDA médio no ano {}
                    - Maior IAN médio no ano {} | Menor IAN médio no ano {}
                    - Maior IAA médio no ano {} | Menor IAA médio no ano {}
                    - Maior IEG médio no ano {} | Menor IEG médio no ano {}
                    - Maior IPP médio no ano {} | Menor IPP médio no ano {}
                    - Maior IPV médio no ano {} | Menor IPV médio no ano {}
                 """.format(
                        stats.loc[stats['media'] == max(stats['media']), 'indicador'].values[0].upper(),
                        stats.media.max(),
                        stats.loc[stats['media'] == max(stats['media']), 'ano'].values[0],
                        stats.loc[stats['media'] == min(stats['media']), 'indicador'].values[0].upper(),
                        stats.media.min(),
                        stats.loc[stats['media'] == min(stats['media']), 'ano'].values[0],
                        stats.loc[stats.variacao_percentual.abs().idxmax()]['indicador'].upper(),
                        stats.loc[stats.variacao_percentual.abs().idxmax()]['variacao_percentual'],
                        stats.query("indicador == 'inde'").loc[stats.query("indicador == 'inde'")['media'].idxmax(), 'ano'],
                        stats.query("indicador == 'inde'").loc[stats.query("indicador == 'inde'")['media'].idxmin(), 'ano'],
                        stats.query("indicador == 'ida'").loc[stats.query("indicador == 'ida'")['media'].idxmax(), 'ano'],
                        stats.query("indicador == 'ida'").loc[stats.query("indicador == 'ida'")['media'].idxmin(), 'ano'],
                        stats.query("indicador == 'ian'").loc[stats.query("indicador == 'ian'")['media'].idxmax(), 'ano'],
                        stats.query("indicador == 'ian'").loc[stats.query("indicador == 'ian'")['media'].idxmin(), 'ano'],
                        stats.query("indicador == 'iaa'").loc[stats.query("indicador == 'iaa'")['media'].idxmax(), 'ano'],
                        stats.query("indicador == 'iaa'").loc[stats.query("indicador == 'iaa'")['media'].idxmin(), 'ano'],
                        stats.query("indicador == 'ieg'").loc[stats.query("indicador == 'ieg'")['media'].idxmax(), 'ano'],
                        stats.query("indicador == 'ieg'").loc[stats.query("indicador == 'ieg'")['media'].idxmin(), 'ano'],
                        stats.query("indicador == 'ipp'").loc[stats.query("indicador == 'ipp'")['media'].idxmax(), 'ano'],
                        stats.query("indicador == 'ipp'").loc[stats.query("indicador == 'ipp'")['media'].idxmin(), 'ano'],
                        stats.query("indicador == 'ipv'").loc[stats.query("indicador == 'ipv'")['media'].idxmax(), 'ano'],
                        stats.query("indicador == 'ipv'").loc[stats.query("indicador == 'ipv'")['media'].idxmin(), 'ano']                        
                    ))
        st.write("<br>", unsafe_allow_html=True)
        
        with st.expander("⚙️ Configurações"):
            indicador = st.radio("Selecione o indicador:", indicadores, horizontal=True)
            agg_usuario = st.radio("Selecione o agrupamento:", list(agrupamentos.keys()), horizontal=True)
        agrupamento = agrupamentos[agg_usuario]
        if agrupamento != 'ano':
            agg_stats = generate_stats(df, 
                                      indicators=['inde', 'ida', 'ian', 'iaa', 'ips', 'ipp', 'ieg', 'ipv'], 
                                      aggregate=agrupamento)
        else:
            agg_stats = stats.copy()
        
        chart_type = st.radio("Selecione o tipo de gráfico:", ["Boxplot", "Violinplot"], 
                                horizontal=True)
        
        if chart_type == "Boxplot":
            plot_percentis_indicador(df, indicador, agrupamento, display_in_streamlit=True)
        else:
            plot_percentis_indicador(df, indicador, agrupamento, plot_type='violinplot', display_in_streamlit=True)
        try:
            st.write("""
                        - Maior média de {} em **{}**, com {:.2f} | Maior mediana em **{}**, com {:.2f}.
                        - Menor média de {} em **{}**, com {:.2f} | Menor mediana em **{}**, com {:.2f}.
                        """.format(
                            indicador.upper(),
                            agg_stats.query("indicador == @indicador.lower()").loc[agg_stats.query("indicador == @indicador.lower()")['media'].idxmax(), agrupamento],
                            agg_stats.query("indicador == @indicador.lower()")['media'].max(),
                            agg_stats.query("indicador == @indicador.lower()").loc[agg_stats.query("indicador == @indicador.lower()")['mediana'].idxmax(), agrupamento],
                            agg_stats.query("indicador == @indicador.lower()")['mediana'].max(),
                            indicador.upper(),
                            agg_stats.query("indicador == @indicador.lower()").loc[agg_stats.query("indicador == @indicador.lower()")['media'].idxmin(), agrupamento],
                            agg_stats.query("indicador == @indicador.lower()")['media'].min(),
                            agg_stats.query("indicador == @indicador.lower()").loc[agg_stats.query("indicador == @indicador.lower()")['mediana'].idxmin(), agrupamento],
                            agg_stats.query("indicador == @indicador.lower()")['mediana'].min()  
                        ))
        except:
            pass
        
        # Caixa de seleção para exibir todos os indicadores
        if st.checkbox("Exibir todos os indicadores"):
            st.write("**Estatísticas dos indicadores agrupadas por {}:**".format(agg_usuario.replace('_', ' ').lower()))
            st.write(agg_stats)
        else:
            st.write("**Estatísticas do indicador {} agrupado por {}:**".format(indicador.upper(), agg_usuario.replace('_', ' ').lower()))
            st.write(agg_stats.query("indicador == @indicador.lower()"))
        
        st.write("<br>", unsafe_allow_html=True)
        with st.expander("⚙️ Configurações"):
            # Selecione indicadores x, y, z e ano=None
            x_selector = st.radio("Indicador para o eixo X:", indicadores, horizontal=True)
            y_selector = st.radio("Indicador para o eixo Y:", indicadores, horizontal=True)
            # Plotar eixo z?
            plot_z = st.checkbox("habilitar eixo Z")
            # Explicar plot
            if plot_z:
                z_selector = st.radio("Indicador para o eixo Z:", indicadores, horizontal=True)
            else:
                z_selector = None
        plot_scatter(df, x=x_selector, y=y_selector, z=z_selector, display_in_streamlit=True)
        
        # selecionar indicadores multiplos
        multi_indicadores = st.multiselect("Selecione os indicadores:", indicadores, default=indicadores)
        plot_correlation_heatmap(df, multi_indicadores, display_in_streamlit=True) 
        
    else: 
        # Checar se dados foram carregados
        if "data_loaded" not in st.session_state:
            df, df_to_verify, df_removed, stats, outliers, initial_shape, final_shape = load_and_process_data(None)
            st.session_state.df = df
            st.session_state.stats = stats
            st.session_state.data_loaded = True
        else:
            df = st.session_state.df
            stats = st.session_state.stats
            
        st.subheader("👩🏾‍🦰 Visão Individual")
        # Selecione o aluno
        alunos = df['nome'].unique()
        aluno = st.selectbox("Selecione o aluno:", alunos)
        pedra = df.query("nome == @aluno")['pedra'].values[0]
        fase = df.query("nome == @aluno")['fase'].values[0]
        # Atingiu o ponto de virada
        if df.query("nome == @aluno")['ponto_virada'].values[0] == 1:
            ponto_virada = "sim"
        else:
            ponto_virada = "não"
        # Bolsista
        if df.query("nome == @aluno")['bolsista'].values[0] == 1:
            bolsista = "sim"
        else:
            bolsista = "não"
        # Se aluno saiu, trazer ano em que saiu
        if df.query("nome == @aluno")['saiu'].values[0] == 1:
            ano_saida = df.query("nome == @aluno")['ano'].max()
        else:
            ano_saida = "ativo"
        # Atenção
        if df.query("nome == @aluno")['atencao'].values[0] == 1:
            atencao = "sim"
        else:
            atencao = "não"
        # Destaque
        if df.query("nome == @aluno")['destaque'].values[0] == 1:
            destaque = "sim"
        else:
            destaque = "não"
        st.write("""
                    💎 **{}** | 💠 **Fase {}**
                    
                    🔄 Ponto de virada: **{}**  | ⤴️ Bolsista: **{}**  | ⏏️ Saída: **{}** | 🔽 Atenção: **{}** | 🔼 Destaque: **{}**
                """.format(pedra, fase, ponto_virada, bolsista, ano_saida, atencao, destaque))
        # Exibir radar e gráficos
        plot_radar_aluno(df, aluno, display_in_streamlit=True)
        st.write("<br>", unsafe_allow_html=True)
        indicadores = ['INDE', 'IAN', 'IAA', 'IDA', 'IEG', 'IPP', 'IPV']
        with st.expander("⚙️ Configurações"):
            indicador = st.radio("Indicador", indicadores, horizontal=True)
        plot_aluno_indicador(df, aluno, indicador, display_in_streamlit=True)
        plot_aluno_variacao(df, aluno, indicador, display_in_streamlit=True)
        plot_aluno_indicador(df, aluno, indicador, display_in_streamlit=True)
        plot_aluno_variacao(df, aluno, indicador, display_in_streamlit=True)


if __name__ == "__main__":
    main()

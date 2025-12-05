import pandas as pd
import sidrapy
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. Coleta dados IBGE (Referência Censo 2022) ---
def coletar_ibge():
    
    
    
    print("1. Coletando dados de População do IBGE (Censo 2022)...")
    
    mapa_estados = {
        'Acre': 'AC', 'Alagoas': 'AL', 'Amapá': 'AP', 'Amazonas': 'AM', 'Bahia': 'BA', 'Ceará': 'CE',
        'Distrito Federal': 'DF', 'Espírito Santo': 'ES', 'Goiás': 'GO', 'Maranhão': 'MA',
        'Mato Grosso': 'MT', 'Mato Grosso do Sul': 'MS', 'Minas Gerais': 'MG', 'Pará': 'PA',
        'Paraíba': 'PB', 'Paraná': 'PR', 'Pernambuco': 'PE', 'Piauí': 'PI', 'Rio de Janeiro': 'RJ',
        'Rio Grande do Norte': 'RN', 'Rio Grande do Sul': 'RS', 'Rondônia': 'RO', 'Roraima': 'RR',
        'Santa Catarina': 'SC', 'São Paulo': 'SP', 'Sergipe': 'SE', 'Tocantins': 'TO'
    }

    try:
        # Tabela 9514 é específica do Censo Demográfico 2022
        dados = sidrapy.get_table(
            table_code="9514", territorial_level="3", ibge_territorial_code="all", 
            variable="93", period="all"
        )
    except Exception as e:
        print(f"Erro API IBGE: {e}")
        return None

    if dados is None or dados.empty:
        return None

    # Ajuste de cabeçalho
    dados.columns = dados.iloc[0]
    dados = dados.iloc[1:]
    
    df = dados[['Unidade da Federação', 'Valor']].copy()
    df.rename(columns={'Valor': 'populacao_ref', 'Unidade da Federação': 'uf_nome'}, inplace=True)
    df['populacao_ref'] = pd.to_numeric(df['populacao_ref'], errors='coerce')
    
    # Criar sigla
    df['uf_sigla'] = df['uf_nome'].map(mapa_estados)
    
    return df[['uf_sigla', 'populacao_ref']]

# --- 2. Processamento CSV Multi-anos ---
def processar_datasets_anos(anos_lista):
    dfs_consolidados = []
    
    # Possíveis nomes de colunas de estado (o dataset pode mudar de nome entre anos)
    colunas_possiveis = ['State', 'uf', 'estado', 'living_state', 'Estado onde mora']

    for ano in anos_lista:
        arquivo = f'state_of_data_{ano}.csv'
        print(f"2. Processando {arquivo}...")
        
        if not os.path.exists(arquivo):
            print(f"   [!] Arquivo {arquivo} não encontrado. Pulando.")
            # Gera dados fictícios apenas se nenhum arquivo for encontrado na primeira rodada para teste
            if len(dfs_consolidados) == 0 and ano == anos_lista[0]: 
                print("   [i] Criando dados fictícios para teste (DEBUG)...")
                df_mock = pd.DataFrame({'State': ['SP']*100 + ['MG']*50 + ['RJ']*30})
                df_mock['uf_sigla'] = 'SP' # Simplificação mock
                # Re-simulando distribuição para o mock
                contagem = pd.DataFrame([['SP', 100], ['MG', 50], ['RJ', 30]], columns=['uf_sigla', 'qtd_profissionais'])
                contagem['ano'] = ano
                dfs_consolidados.append(contagem)
            continue

        try:
            df = pd.read_csv(arquivo)
            
            # Tenta encontrar a coluna de estado automaticamente
            coluna_encontrada = next((col for col in colunas_possiveis if col in df.columns), None)
            
            if coluna_encontrada:
                # Normalização da sigla
                df['uf_sigla'] = df[coluna_encontrada].astype(str).str.strip().str.upper()
                
                # Tratamento para remover termos como "Brazil" ou nulos se houver
                df = df[df['uf_sigla'].str.len() == 2] 

                # Agrupamento
                df_contagem = df['uf_sigla'].value_counts().reset_index()
                df_contagem.columns = ['uf_sigla', 'qtd_profissionais']
                df_contagem['ano'] = ano # Marca o ano do dataset
                
                dfs_consolidados.append(df_contagem)
            else:
                print(f"   [Erro] Coluna de estado não encontrada em {arquivo}")

        except Exception as e:
            print(f"   [Erro] Falha ao ler {arquivo}: {e}")

    if dfs_consolidados:
        return pd.concat(dfs_consolidados, ignore_index=True)
    else:
        return None

# --- 3. Execução ---
if __name__ == "__main__":
    # Define os anos que queremos analisar
    anos_analise = [2022, 2023, 2024]
    
    # 1. Busca População (Base 2022)
    df_ibge = coletar_ibge()
    
    # 2. Busca Dados de Profissionais (Loop 2022-2024)
    df_sod = processar_datasets_anos(anos_analise)
    
    if df_ibge is not None and df_sod is not None:
        # Merge: Trazemos a população para cada linha do dataset de dados
        # A população será repetida para 2023 e 2024 (o que é aceitável para essa análise)
        df_final = pd.merge(df_sod, df_ibge, on='uf_sigla', how='left')
        
        df_final['qtd_profissionais'] = df_final['qtd_profissionais'].fillna(0)
        
        print("\n--- RESUMO DOS DADOS CONSOLIDADOS ---")
        print(df_final.groupby('ano')['qtd_profissionais'].sum().reset_index())
        
        # Salvar arquivo unificado
        df_final.to_csv("analise_state_of_data_2022_2024.csv", index=False)
        print("\n[OK] Arquivo salvo: analise_state_of_data_2022_2024.csv")

        # --- Visualização ---
        plt.figure(figsize=(14, 7))
        
        # Filtrar apenas Top 5 estados (pelo total geral) para não poluir o gráfico
        top_estados = df_final.groupby('uf_sigla')['qtd_profissionais'].sum().nlargest(5).index
        df_plot = df_final[df_final['uf_sigla'].isin(top_estados)]
        
        # Gráfico agrupado
        ax = sns.barplot(
            data=df_plot, 
            x='uf_sigla', 
            y='qtd_profissionais', 
            hue='ano',  # Aqui está a mágica: separa as barras por ano
            palette='viridis'
        )
        
        plt.title('Evolução de Profissionais de Dados por Estado (2022-2024)')
        plt.xlabel('Estado')
        plt.ylabel('Número de Respondentes')
        plt.legend(title='Ano da Pesquisa')
        
        # Adicionar rótulos nas barras
        for container in ax.containers:
            ax.bar_label(container, padding=3, fmt='%d', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Não foi possível gerar a análise por falta de dados.")
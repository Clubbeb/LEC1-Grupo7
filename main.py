import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configurar estilo de gráficos
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

# =============================================================================
# 1. CARGA Y CONTEXTUALIZACIÓN DEL DATASET
# =============================================================================
print("=" * 80)
print("ANÁLISIS EXPLORATORIO DE DATOS - INDICADORES MACROECONÓMICOS MUNDIALES")
print("=" * 80)

# Cargar el dataset
df = pd.read_csv('world_bank_data_2025.csv')

# Contextualización del dataset
print("\n1. CONTEXTUALIZACIÓN DEL DATASET")
print("-" * 40)
print("Este dataset contiene indicadores macroeconómicos de diversos países")
print("a lo largo de varios años. Los datos provienen del Banco Mundial y")
print("otras fuentes internacionales, proporcionando una visión completa")
print("del desempeño económico global.")

# Motivación para la selección del dataset
print("\nMOTIVACIÓN:")
print("- Los indicadores macroeconómicos son fundamentales para entender")
print("  el desarrollo económico de los países")
print("- Permite analizar relaciones entre diferentes variables económicas")
print("- Ofrece oportunidades para estudiar crisis económicas y recuperaciones")
print("- Los datos son relevantes para políticas económicas y decisiones de inversión")

# Exploración inicial
print(f"\n📊 DIMENSIONES DEL DATASET: {df.shape}")
print(f"📅 RANGO TEMPORAL: {df['year'].min()} - {df['year'].max()}")
print(f"🌍 PAÍSES INCLUIDOS: {df['country_name'].nunique()}")

# =============================================================================
# 2. LIMPIEZA Y PREPARACIÓN DE DATOS
# =============================================================================
print("\n\n2. LIMPIEZA Y PREPARACIÓN DE DATOS")
print("-" * 40)

# 2.1. Identificación de valores faltantes
print("2.1. VALORES FALTANTES POR VARIABLE:")
print("-" * 35)

null_info = pd.DataFrame({
    'Valores_Nulos': df.isnull().sum(),
    'Porcentaje_Nulos': (df.isnull().sum() / len(df)) * 100
})
print(null_info[null_info['Valores_Nulos'] > 0])

# 2.2. Imputación de valores faltantes
print("\n2.2. IMPUTACIÓN DE VALORES FALTANTES:")
print("-" * 35)

# Definir variables económicas
economic_cols = [
    'Inflation (CPI %)', 'GDP (Current USD)', 'GDP per Capita (Current USD)',
    'Unemployment Rate (%)', 'Interest Rate (Real, %)', 'Inflation (GDP Deflator, %)',
    'GDP Growth (% Annual)', 'Current Account Balance (% GDP)',
    'Government Expense (% of GDP)', 'Government Revenue (% of GDP)',
    'Tax Revenue (% of GDP)', 'Gross National Income (USD)', 'Public Debt (% of GDP)'
]

# Imputar valores faltantes usando la mediana por país
for col in economic_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        initial_nulls = df[col].isnull().sum()
        df[col] = df.groupby('country_name')[col].transform(
            lambda x: x.fillna(x.median() if not np.isnan(x.median()) else df[col].median())
        )
        final_nulls = df[col].isnull().sum()
        print(f"✅ {col}: {initial_nulls} → {final_nulls} valores nulos")

# 2.3. Manejo de variables categóricas
print("\n2.3. VARIABLES CATEGÓRICAS:")
print("-" * 25)

categorical_cols = ['country_name', 'country_id']
for col in categorical_cols:
    if col in df.columns:
        print(f"{col}: {df[col].nunique()} categorías únicas")
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
            print(f"✅ {col}: Valores nulos imputados")

# 2.4. Detección de valores atípicos y errores
print("\n2.4. DETECCIÓN DE VALORES ATÍPICOS:")
print("-" * 30)

# Verificar rangos esperados para variables económicas
if 'Unemployment Rate (%)' in df.columns:
    unusual_unemployment = df[(df['Unemployment Rate (%)'] < 0) | (df['Unemployment Rate (%)'] > 50)]
    print(f"Tasa de desempleo inusual: {len(unusual_unemployment)} registros")

if 'Inflation (CPI %)' in df.columns:
    unusual_inflation = df[(df['Inflation (CPI %)'] < -10) | (df['Inflation (CPI %)'] > 1000)]
    print(f"Inflación inusual: {len(unusual_inflation)} registros")

if 'GDP Growth (% Annual)' in df.columns:
    unusual_growth = df[(df['GDP Growth (% Annual)'] < -20) | (df['GDP Growth (% Annual)'] > 50)]
    print(f"Crecimiento económico inusual: {len(unusual_growth)} registros")

# =============================================================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS
# =============================================================================
print("\n\n3. ANÁLISIS EXPLORATORIO DE DATOS")
print("-" * 35)

# 3.1. Tipos de variables y estructura del dataset
print("3.1. ESTRUCTURA DEL DATASET:")
print("-" * 25)
print(df.info())

# 3.2. Estadísticas descriptivas
print("\n3.2. ESTADÍSTICAS DESCRIPTIVAS:")
print("-" * 30)

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe())

# 3.3. Análisis por países
print("\n3.3. ANÁLISIS POR PAÍSES:")
print("-" * 20)

if 'GDP (Current USD)' in df.columns:
    top_countries_gdp = df.groupby('country_name')['GDP (Current USD)'].mean().sort_values(ascending=False).head(10)
    print("TOP 10 PAÍSES POR PIB PROMEDIO:")
    for i, (country, gdp) in enumerate(top_countries_gdp.items(), 1):
        print(f"{i}. {country}: ${gdp:,.2f}")

if 'GDP Growth (% Annual)' in df.columns:
    top_growth = df.groupby('country_name')['GDP Growth (% Annual)'].mean().sort_values(ascending=False).head(10)
    print("\nTOP 10 PAÍSES POR CRECIMIENTO PROMEDIO:")
    for i, (country, growth) in enumerate(top_growth.items(), 1):
        print(f"{i}. {country}: {growth:.2f}%")


# =============================================================================
# 4. VISUALIZACIONES
# =============================================================================
print("\n\n4. VISUALIZACIONES")
print("-" * 15)

# Configuración de visualizaciones
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 4.1. Evolución temporal del PIB mundial
if 'year' in df.columns and 'GDP (Current USD)' in df.columns:
    yearly_gdp = df.groupby('year')['GDP (Current USD)'].mean()
    axes[0, 0].plot(yearly_gdp.index, yearly_gdp.values, marker='o', linewidth=2, color=colors[0])
    axes[0, 0].set_title('Evolución del PIB Mundial Promedio', fontweight='bold')
    axes[0, 0].set_xlabel('Año')
    axes[0, 0].set_ylabel('PIB Promedio (USD)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].ticklabel_format(style='plain', axis='y')

# 4.2. Top 10 países por PIB
if 'country_name' in df.columns and 'GDP (Current USD)' in df.columns:
    top_10_countries_gdp = df.groupby('country_name')['GDP (Current USD)'].mean().nlargest(10)
    bars = axes[0, 1].barh(range(len(top_10_countries_gdp)), top_10_countries_gdp.values, color=colors)
    axes[0, 1].set_yticks(range(len(top_10_countries_gdp)))
    axes[0, 1].set_yticklabels(top_10_countries_gdp.index)
    axes[0, 1].set_title('Top 10 Países por PIB Promedio', fontweight='bold')
    axes[0, 1].set_xlabel('PIB Promedio (USD)')
    axes[0, 1].ticklabel_format(style='plain', axis='x')

# 4.3. Relación entre PIB per cápita y desempleo
if 'GDP per Capita (Current USD)' in df.columns and 'Unemployment Rate (%)' in df.columns:
    sample_df = df.sample(min(1000, len(df)))
    scatter = axes[1, 0].scatter(sample_df['GDP per Capita (Current USD)'], 
                                sample_df['Unemployment Rate (%)'], 
                                alpha=0.6, color=colors[2], s=30)
    axes[1, 0].set_xlabel('PIB per Cápita (USD)')
    axes[1, 0].set_ylabel('Tasa de Desempleo (%)')
    axes[1, 0].set_title('Relación PIB per Cápita vs Desempleo', fontweight='bold')
    axes[1, 0].set_xscale('log')

# 4.4. Distribución del crecimiento económico
if 'GDP Growth (% Annual)' in df.columns:
    growth_data = df['GDP Growth (% Annual)'].dropna()
    axes[1, 1].hist(growth_data, bins=30, alpha=0.7, color=colors[3], edgecolor='black')
    axes[1, 1].axvline(x=growth_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {growth_data.mean():.2f}%')
    axes[1, 1].set_xlabel('Crecimiento del PIB (% Anual)')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución del Crecimiento Económico', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Análisis Exploratorio - Indicadores Macroeconómicos Mundiales', 
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 4.5. Matriz de correlación
print("\n4.5. MATRIZ DE CORRELACIÓN ENTRE VARIABLES ECONÓMICAS:")
print("-" * 55)

if len(numeric_cols) > 1:
    # Seleccionar variables económicas principales
    economic_cols_for_corr = [col for col in economic_cols if col in df.columns][:8]  # Limitar a 8 variables
    
    if len(economic_cols_for_corr) > 1:
        correlation_matrix = df[economic_cols_for_corr].corr()
        
        # Crear heatmap de correlación
        plt.figure(figsize=(10, 8))
        im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        plt.xticks(range(len(economic_cols_for_corr)), economic_cols_for_corr, rotation=45, ha='right')
        plt.yticks(range(len(economic_cols_for_corr)), economic_cols_for_corr)
        plt.title('Matriz de Correlación - Variables Económicas', fontweight='bold')
        
        # Añadir valores de correlación
        for i in range(len(economic_cols_for_corr)):
            for j in range(len(economic_cols_for_corr)):
                plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)
        
        # Añadir barra de color
        cbar = plt.colorbar(im)
        cbar.set_label('Coeficiente de Correlación')
        plt.tight_layout()
        plt.show()
        
        # Mostrar correlaciones más fuertes
        print("\nCORRELACIONES MÁS SIGNIFICATIVAS:")
        corr_pairs = correlation_matrix.unstack().sort_values(key=abs, ascending=False)
        corr_pairs = corr_pairs[corr_pairs != 1.0]  # Eliminar autocorrelaciones
        for i, ((var1, var2), corr) in enumerate(corr_pairs.head(5).items()):
            print(f"{i+1}. {var1} - {var2}: {corr:.3f}")

# =============================================================================
# 5. ANÁLISIS DE CALIDAD DE DATOS Y SELECCIÓN DE VARIABLES
# =============================================================================
print("\n\n5. ANÁLISIS DE CALIDAD DE DATOS")
print("-" * 30)

# 5.1. Calidad de datos por país
if 'country_name' in df.columns:
    data_quality = df.groupby('country_name').agg({
        'GDP (Current USD)': ['count', lambda x: x.isnull().sum()],
        'Inflation (CPI %)': lambda x: x.isnull().sum(),
        'Unemployment Rate (%)': lambda x: x.isnull().sum()
    })
    
    data_quality.columns = ['Total_Records', 'GDP_Nulls', 'Inflation_Nulls', 'Unemployment_Nulls']
    data_quality['Data_Quality_Score'] = 100 - (
        (data_quality['GDP_Nulls'] + data_quality['Inflation_Nulls'] + data_quality['Unemployment_Nulls']) / 
        data_quality['Total_Records'] * 100
    )
    
    print("PAÍSES CON MEJOR CALIDAD DE DATOS:")
    top_quality = data_quality.nlargest(5, 'Data_Quality_Score')[['Total_Records', 'Data_Quality_Score']]
    print(top_quality)
    
    print("\nPAÍSES CON PEOR CALIDAD DE DATOS:")
    bottom_quality = data_quality.nsmallest(5, 'Data_Quality_Score')[['Total_Records', 'Data_Quality_Score']]
    print(bottom_quality)

# 5.2. Selección de variables para análisis futuro
print("\n\n6. SELECCIÓN DE VARIABLES PARA ANÁLISIS FUTURO")
print("-" * 45)

print("VARIABLES PRINCIPALES SELECCIONADAS:")
print("1. GDP (Current USD) - Variable continua que representa el tamaño económico")
print("2. GDP Growth (% Annual) - Variable continua que mide el crecimiento económico")

print("\nVARIABLES SECUNDARIAS SELECCIONADAS:")
print("1. Inflation (CPI %) - Variable continua que mide la estabilidad de precios")
print("2. Unemployment Rate (%) - Variable continua que refleja el mercado laboral")

print("\nJUSTIFICACIÓN:")
print("- El PIB y su crecimiento son indicadores fundamentales del desempeño económico")
print("- La inflación y el desempleo son variables clave para políticas económicas")
print("- Existen correlaciones interesantes entre estas variables según el análisis exploratorio")
print("- Estas variables permiten formular preguntas relevantes sobre desarrollo económico")

# =============================================================================
# 6. FORMULACIÓN DE PREGUNTA DE INVESTIGACIÓN
# =============================================================================
print("\n\n7. PREGUNTA DE INVESTIGACIÓN")
print("-" * 30)

print("¿Existe una relación significativa entre el crecimiento económico (GDP Growth)")
print("y la tasa de desempleo, y cómo esta relación varía entre países desarrollados")
print("y países en desarrollo?")

print("\nCARACTERÍSTICAS DE LA PREGUNTA:")
print("✅ CLARA: Se especifican las variables y la población de estudio")
print("✅ ESPECÍFICA: Examina una relación particular entre variables económicas")
print("✅ MEDIBLE: Las variables están disponibles y son cuantificables")
print("✅ ABORDABLE: Puede responderse con los datos disponibles")
print("✅ RELEVANTE: Tiene implicaciones para políticas económicas")
print("✅ VERIFICABLE: Puede probarse mediante análisis estadístico")

print("\nVARIABLES INVOLUCRADAS:")
print("- Variable independiente: GDP Growth (% Annual)")
print("- Variable dependiente: Unemployment Rate (%)")
print("- Variable de agrupación: Nivel de desarrollo (a definir por PIB per cápita)")

print("\nENFOQUE METODOLÓGICO PREVISTO:")
print("- Análisis de correlación y regresión")
print("- Pruebas de hipótesis por grupos de países")
print("- Análisis de series de tiempo para tendencias")
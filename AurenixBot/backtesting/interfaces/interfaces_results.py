import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Tuple, Any
import os
import math

from backtesting.interfaces import position_chart 
from connection.broker_connection import broker_connection

from config.config_global import TRADES_FILE, CONFIG_FILE

PLOTLY_CONFIG = {
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d', 'autoScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines']
}

# ======================================================================
# ⚙️ UTILIDADES DE CARGA DE DATOS
# ======================================================================

@st.cache_data
def load_backtest_data() -> Tuple[pd.DataFrame | None, float, float, str, str]:
    """Carga los datos de trades y el capital desde los archivos guardados."""
    
    # 1. Verificar existencia del archivo de trades
    if not os.path.exists(TRADES_FILE):
        st.error(f"❌ Archivo de trades no encontrado ({TRADES_FILE}). Ejecute el backtesting primero.")
        return None, 0.0, 0.0, '', '', 0, '', ''

    try:
        trades_df = pd.read_parquet(TRADES_FILE)
        
        # 2. Cargar configuraciones (Balance Inicial/Final)
        config = {}
        if os.path.exists(CONFIG_FILE):
         with open(CONFIG_FILE, "r") as f:
             for line in f:
                 try:
                     key, value = line.strip().split(":")
                     try:
                         config[key.strip()] = float(value.strip())
                     except ValueError:
                         config[key.strip()] = value.strip()
                         
                 except ValueError:
                     continue
        
        initial_balance = config.get('initial_balance', 0.0)
        final_balance = config.get('final_balance', 0.0)
        total_days_analyze = config.get('total_days_analyze', '')
        total_days_to_analyze = config.get('total_days_to_analyze', '')
        decimal_symbol = config.get('decimal_symbol', 0)
        symbol = config.get('symbol', '')
        timeframe = config.get('timeframe', '')
        
        return trades_df, initial_balance, final_balance, total_days_analyze, total_days_to_analyze, int(decimal_symbol), symbol, timeframe
        
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None, 0.0, 0.0, '', '', 0, '',''

# ======================================================================
# 🎨 FUNCIONES DE ESTILIZACIÓN Y CÁLCULO DE MÉTRICAS
# ======================================================================
def calculate_metrics(trades_df: pd.DataFrame, initial_balance: float, final_balance: float, total_days_analyze: str, total_days_to_analyze: str) -> dict:
    """Calcula las métricas de rendimiento completas, usando nombres de clave internos en inglés."""
    
    total_trades = len(trades_df)
    
    # 1. Trade Grouping and Sums
    positive_trades = trades_df[trades_df['net_profit'] > 0]
    negative_trades = trades_df[trades_df['net_profit'] <= 0]
    
    total_positive = len(positive_trades)
    total_negative = len(negative_trades)
    
    total_net_profit = positive_trades['net_profit'].sum()  # Total amount won
    total_net_loss = negative_trades['net_profit'].sum()    # Total amount lost (negative or zero)
    
    total_profit_delta = final_balance - initial_balance
    
    # Absolute values for ratios
    abs_total_net_loss = abs(total_net_loss) 
    
    # --- CALCULATIONS FOR RENTABILITY AND PROPORTIONS (Keys in English) ---

    # 1. Win Rate and Loss Rate (%)
    if total_trades > 0:
        win_rate_op = (total_positive / total_trades) * 100
        loss_rate_op = (total_negative / total_trades) * 100
    else:
        win_rate_op = 0.0
        loss_rate_op = 0.0

    # 2. Money Proportions (%)
    total_money_analyzed = total_net_profit + abs_total_net_loss
    if total_money_analyzed > 0:
        win_proportion_money = (total_net_profit / total_money_analyzed) * 100
        loss_proportion_money = (abs_total_net_loss / total_money_analyzed) * 100
    else:
        win_proportion_money = 0.0
        loss_proportion_money = 0.0

    # 3. Average R/R and Expectancy
    avg_win = total_net_profit / total_positive if total_positive > 0 else 0.0
    avg_loss = abs_total_net_loss / total_negative if total_negative > 0 else 0.0

    if avg_loss > 0:
        rr_ratio_average = avg_win / avg_loss
    elif total_positive > 0 and total_negative == 0:
        rr_ratio_average = math.inf # Perfect case
    else:
        rr_ratio_average = 0.0
        
    # Expectancy (Edge)
    expectancy_value = (win_rate_op/100 * avg_win) - (loss_rate_op/100 * avg_loss)

    # 4. Profit Factor
    if abs_total_net_loss > 0:
        profit_factor_value = total_net_profit / abs_total_net_loss
        display_profit_factor = f"{profit_factor_value:.2f}"
    else:
        display_profit_factor = "Perfecto (∞)"

    # --- RETURN DICTIONARY (All Keys in English) ---
    return {
        "total_days_to_analyze": f"{total_days_to_analyze}",
        "total_days_analyzed": f"{total_days_analyze}",
        "total_trades": total_trades,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "margin_call_count": len(trades_df[trades_df['close_type'] == 'MARGIN_CALL']),
        "total_net_profit": f"{total_net_profit:.2f}",
        "total_profit_delta": f"{total_profit_delta:.2f}",
        "final_balance": f"{final_balance:.2f}",
        "initial_balance": f"{initial_balance:.2f}",
        "total_net_loss": f"{total_net_loss:.2f}",
        "profit_factor": display_profit_factor,
        
        # New 10 Rentability/Proportion Metrics (English Keys)
        "win_rate_op": win_rate_op,  
        "loss_rate_op": loss_rate_op,
        "win_proportion_money": win_proportion_money,
        "loss_proportion_money": loss_proportion_money,
        "rr_ratio_average": rr_ratio_average,
        "expectancy": expectancy_value,
        
        # Count and Money Totals for Display
        "positive_trades_count": total_positive,
        "negative_trades_count": total_negative,
        "total_gains_amount": total_net_profit,
        "total_losses_amount": abs_total_net_loss,
    }
def create_equity_curve_plot(trades_df: pd.DataFrame, initial_balance: float):
    """
    Genera la curva de equidad interactiva con Plotly.
    Ajustada para usar fondo transparente (para Streamlit) y aplicar padding/altura.
    """
    
    trades_df['cummulative_profit'] = trades_df['net_profit'].cumsum()
    trades_df['equity'] = initial_balance + trades_df['cummulative_profit']
    
    # Crea una Serie de tiempo para la gráfica
    time_range = pd.concat([
        pd.Series(initial_balance, index=[trades_df['entry_time'].min()]), 
        trades_df[['equity', 'exit_time']].set_index('exit_time')['equity']
    ]).sort_index()
    
    fig = px.line(time_range, 
                  y=time_range.values, 
                  x=time_range.index,
                  title='Curva de Equidad (Equity Curve)',
                  labels={'y': 'Equidad ($)', 'x': 'Fecha y Hora'})
    
    # Estilizado
    fig.update_traces(line_color='#2ecc71', line_width=2)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        font_color='#FAFAFA', 

        dragmode='pan',
        height=650, 
        
        margin=dict(l=50, r=50, t=50, b=50),
    )
    
    return fig

# ----------------------------------------------------------------------
# 🌟 FUNCIÓN DE FORMATO DEL HISTORIAL
# ----------------------------------------------------------------------

def format_trades_for_display(trades_df: pd.DataFrame,  decimal_symbol: int) -> Any:
    """Aplica la lógica de estilizado de color y formato en español."""
    
    df_display = trades_df.copy()
    
    column_mapping = {
        'symbol': 'Símbolo',
        'side': 'Tipo de Posición',
        'lot_size': 'Lotes',
        'entry_time': 'Apertura (Hora)',
        'entry_price': 'Precio Entrada',
        'exit_time': 'Cierre (Hora)',
        'exit_price': 'Precio Salida',
        'net_profit': 'Ganancia ($)',
        'pips_moved': 'Pips',
        'commission': 'Comisión ($)',
        'margin_used': 'Margen Usado ($)',
        'close_type': 'Tipo de Cierre'
    }
    existing_columns = {k: v for k, v in column_mapping.items() if k in df_display.columns}
    df_display = df_display.rename(columns=existing_columns)
    
    net_profit_original = trades_df['net_profit']

    # 2. Funciones de Estilización (Colores)
    
    # Estilo 1: Colorea solo la celda del Tipo de Posición (Buy/Sell)
    def highlight_side(s):
        """Colorea BUY (azul) y SELL (rojo)."""
        return ['color: blue; font-weight: 500' if v == 'BUY' else ('color: red; font-weight: 500' if v == 'SELL' else '') 
                for v in s]

    # Estilo 2: Colorea solo la celda de Ganancia ($)
    def highlight_pnl_net_col(s):
        """Colorea profit (verde si > 0, rojo si < 0) y gris claro si = 0."""

        return [
            'color: green; font-weight: bold' if net_profit_original.loc[idx] > 0 
            else 'color: red; font-weight: bold' if net_profit_original.loc[idx] < 0 
            else 'color: lightgray; font-weight: normal' if net_profit_original.loc[idx] == 0 
            else 'color: inherit' for idx in s.index
        ]

    # Estilo 3: Colorea solo la celda de Tipo de Cierre (Rojo, Verde, Naranja)
    def highlight_close_type_cell(s):
        """Colorea la celda de Tipo de Cierre basada en el tipo y PnL."""
        
        # Obtenemos los índices para acceder a los datos originales
        styles = []
        for idx, close_type in s.items():
            profit_original = net_profit_original.loc[idx] 
            color = 'inherit'

            if close_type in ['STOP_LOSS', 'MARGIN_CALL']:
                color = 'red'
            elif close_type in ['TP/DYNAMIC', 'TP/FIXED'] :
                if profit_original > 0:
                    color = 'green'
                elif profit_original <= 0:
                    color = 'orange' 
            
            # Aplicamos el estilo de color solo al texto de la celda
            styles.append(f'color: {color}; font-weight: 500;')

        return styles

    TIME_FORMAT_STRING = "%d %b %Y %I:%M %p"
    # 3. Aplicar Estilos y Formato con Styler
    styler = df_display.style \
        .apply(highlight_side, subset=['Tipo de Posición']) \
        .apply(highlight_pnl_net_col, subset=['Ganancia ($)']) \
        .apply(highlight_close_type_cell, subset=['Tipo de Cierre']) \
        .format({
            'Lotes': f"{{:.{2}f}}",
            'Apertura (Hora)': lambda t: t.strftime(TIME_FORMAT_STRING),
            'Precio Entrada': f"{{:.{decimal_symbol}f}}",
            'Cierre (Hora)': lambda t: t.strftime(TIME_FORMAT_STRING),
            'Precio Salida': f"{{:.{decimal_symbol}f}}",
            'Ganancia ($)': f"{{:.{2}f}}", 
            'Pips': f"{{:.{decimal_symbol}f}}", 
            'Comisión ($)': f"{{:.{2}f}}",
            'Margen Usado ($)': f"{{:.{2}f}}",
            })
    
    return styler

# ======================================================================
# GENERAR GRAFICO DE TRADES
# ======================================================================

def render_trading_dashboard(symbol: str, timeframe: str):
    
    csv = broker_connection.get_data_filename(symbol=symbol, timeframe=timeframe)
    # 1. Cargar y preparar datos usando la función del archivo de lógica
    df_ohlc, df_trades = position_chart.load_and_prepare_data( 
        csv, 
        TRADES_FILE
        )
    
    if df_ohlc is None:
        st.error("🛑 Error al cargar la data. Revisa las rutas y los archivos.")
        return 
    
    # 2. TITULO: GRÁFICO DE VELAS
    st.header("Gráfico de Velas con Trades")
    
    # 3. GENERA EL GRAFICO
    ohlc_figure = position_chart.plot_trades_on_ohlc_plotly(df_ohlc, df_trades)
    
    # 4. MOSTRAR EL GRAFICO
    st.plotly_chart(ohlc_figure, use_container_width=True)

# ======================================================================
# 🖥️ PUNTO DE ENTRADA DE STREAMLIT
# ======================================================================

def main_interface():
    trades_df, initial_balance, final_balance, total_days_analyze, total_days_to_analyze, decimal_symbol, symbol, timeframe = load_backtest_data()

    st.set_page_config(
        page_title="🤖 Resultados de Backtesting", 
        layout="wide", 
        initial_sidebar_state="expanded"
        )
    
    hide_streamlit_style = """
        <style>
        header {
            visibility: hidden;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        .css-1jc7ptx {
            display: none;
        }
        </style>
    """
    
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    

    cola, colb = st.columns([3, 1])
    with cola:
        st.title('Simulador de Backtesting de Trading 📊')

    # Columna 2: Información del Activo/Temporalidad (alineada a la derecha y más pequeña)
    with colb:
        # Usamos un contenedor de Markdown para darle formato y alineación a la derecha.
        # El HTML/CSS inyectado permite el tamaño más pequeño y la alineación.
        st.markdown(
            f"""
            <div style="text-align: right; font-size: x-large; margin-top: 30px;">
                <p>
                    Activo: <strong>{symbol}</strong> &nbsp; &nbsp;
                    Temp: <strong>{timeframe}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        #st.title(f'ACTIVO: {symbol} - TEMPORALIDAD: {timeframe}')
    
    if trades_df is None or trades_df.empty:
        st.warning("Aún no se han generado datos o hubo un error al cargar. Ejecute el backtesting primero.")
        st.info("Para generar el archivo de datos, ejecute `python backtesting_engine.py` en la terminal.")
        return

    metrics = calculate_metrics(trades_df, initial_balance, final_balance, total_days_analyze, total_days_to_analyze)
    
    st.markdown("---")

    # --- MÉTRICAS CLAVE ---

    
    total_profit_delta_float = float(metrics.get('total_profit_delta', 0.0))
    total_net_profit_float = float(metrics.get('total_net_profit', 0.0))
    total_net_loss_float = float(metrics.get('total_net_loss', 0.0))

    # --- 1. TIEMPO DE ANÁLISIS ---

    st.header("Tiempo de Analisis")
    colMA, colA, colMB, colB = st.columns(4)

    # Días Por Analizar (colMA)
    colMA.metric(
        "Tiempo Por Analizar", 
        metrics.get('total_days_to_analyze', "N/A"),
        help="Período de tiempo solicitado en la configuración del backtesting." # Ayuda agregada
    )

    # Días Analizados (colB)
    colB.metric(
        "Tiempo Analizado", 
        metrics.get('total_days_analyzed', "N/A"),
        help="Período de tiempo real de data que fue analizada." # Ayuda agregada
    )

    # ColA y ColMB quedan vacías en este bloque.

    st.markdown("---")

    # --- 2. MÉTRICAS CLAVE DEL RENDIMIENTO (Balance e Impacto Monetario) ---

    st.header("Métricas Clave del Rendimiento")
    col1, col2, col3, col4, = st.columns(4)

    # 1. Balance Inicial (col1)
    col1.metric("Balance Inicial", f"${float(metrics.get('initial_balance', 0.0)):,.2f}")

    # 2. Balance Final (col2)
    with col2:
        st.metric(
            label="Balance Final", 
            value=f"${float(metrics.get('final_balance', 0.0)):,.2f}",
            delta=total_profit_delta_float,
            help="Balance actual de la cuenta después de todas las operaciones (Ganancia neta + Balance Inicial)." # Ayuda agregada
        ) 

    # 3. Monto Ganado (col3)
    with col3:
        st.metric(
            label="Monto Ganado", 
            value=f"${metrics.get('total_gains_amount', 0.0):,.2f}",
            delta=f"+{metrics.get('win_proportion_money', 0.0):.2f} %",
            help="Suma total de todas las ganancias brutas. El delta es su proporción respecto al dinero total (Ganancia + Pérdida)." # Ayuda agregada
        )

    # 4. Monto Perdido (col4)
    with col4:
        st.metric(
            label="Monto Perdido", 
            value=f"${metrics.get('total_losses_amount', 0.0):,.2f}",
            delta=f"-{metrics.get('loss_proportion_money', 0.0):.2f} %",
            help="Suma total de todas las pérdidas brutas. El delta es su proporción respecto al dinero total (Ganancia + Pérdida)." # Ayuda agregada
        ) 

    st.markdown("---")

    # --- 3. DESGLOSE DE OPERACIONES Y RIESGO (Frecuencia) ---

    st.subheader("Desglose de Operaciones y Riesgo")
    col6, col7, col8, col9, col10 = st.columns(5)

    # Trades Realizados (col6)
    col6.metric(
        "Trades Realizados", 
        metrics.get('total_trades', 0),
        help="Número total de operaciones ejecutadas durante el período analizado." # Ayuda agregada
    )

    # Positivos (col7)
    with col7:
        st.metric(
            label="Positivos (Conteo)", 
            value=metrics.get('total_positive', 0), 
            delta=total_net_profit_float,
            help="Número de operaciones que resultaron en ganancia. El delta muestra el monto total ganado." # Ayuda agregada
        )

    # Negativos (col8)
    with col8:
        st.metric(
            label="Negativos (Conteo)", 
            value=metrics.get('total_negative', 0), 
            delta=total_net_loss_float,
            help="Número de operaciones que resultaron en pérdida o empate. El delta muestra el monto total perdido." # Ayuda agregada
        )

    # Win Rate (%) (col9)
    col9.metric(
        "Win Rate (%)", 
        f"{metrics.get('win_rate_op', 0.0):.2f}%",
        help="Frecuencia de éxito: Porcentaje de operaciones que fueron ganadoras." # Ayuda agregada
    )

    # Loss Rate (%) (col10)
    col10.metric(
        "Loss Rate (%)", 
        f"{metrics.get('loss_rate_op', 0.0):.2f}%",
        help="Frecuencia de fracaso: Porcentaje de operaciones que fueron perdedoras." 
    )


    st.markdown("---")

    # --- 4. EFECTIVIDAD Y EXPECTATIVA (Ratios y Edge) ---

    st.subheader("Frecuencia y Efectividad")
    col11, col12, col13, col14 = st.columns(4)

    # Profit Factor (col11)
    col11.metric(
        "Profit Factor", 
        metrics.get('profit_factor', 0.0), 
        help="Ganancia Bruta dividida por Pérdida Bruta. Un valor mayor a 1.0 indica rentabilidad." # Ayuda agregada
    )

    # Ratio R/R Promedio (col12)
    rr_ratio_value = metrics.get('rr_ratio_average', 0.0)
    col12.metric(
        "R/R Promedio", 
        f"{rr_ratio_value:.2f}:1",
        help="Relación entre la ganancia promedio por trade y la pérdida promedio por trade." # Ayuda agregada
    )

    # Expectativa Matemática (Edge) (col13)
    expectancy_value = metrics.get('expectancy', 0.0)
    col13.metric(
        "Expectativa (Edge)", 
        f"${expectancy_value:.2f}",
        delta_color="normal" if expectancy_value >= 0 else "inverse",
        help="Ganancia esperada por unidad de riesgo (por dólar o unidad arriesgada). Debe ser > $0.00." # Ayuda original mantenida
    )

    # Margin Calls (col14)
    col14.metric(
        "Margin Calls", 
        metrics.get('margin_call_count', 0), 
        delta_color="inverse",
        help="Número de veces que la cuenta alcanzó el nivel de 'Margin Call' (ejecución automática de cierre por falta de margen)." # Ayuda agregada
    )
    st.markdown("---")

    # --- GRÁFICO DE EQUIDAD ---
    st.header("Curva de Equidad")
    equity_fig = create_equity_curve_plot(trades_df.copy(), initial_balance)
    st.plotly_chart(equity_fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")
    
    # --- HISTORIAL COMPLETO ---
    st.header("Historial Detallado de Trades (Completo)")
    
    styler = format_trades_for_display(trades_df, decimal_symbol)
    st.dataframe(styler, width='stretch')

    # --- GRAFICO DE ENTRADAS ---
    st.markdown("---")
    render_trading_dashboard(symbol, timeframe)

if __name__ == '__main__':
    main_interface()

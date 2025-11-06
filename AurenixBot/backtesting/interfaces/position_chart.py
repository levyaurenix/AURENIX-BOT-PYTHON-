import pandas as pd
import os
import plotly.graph_objects as go

from config.config_global import TRADES_FILE

# --- 1. CARGA Y PREPARACIÓN DE DATOS (Data Loading and Preparation) ---

def load_and_prepare_data(csv_path: str, parquet_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga y prepara los DataFrames de velas (OHLC) y trades.
    Asegura que el DataFrame de velas tenga un DatetimeIndex y columnas OHLC mayúsculas.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró el archivo CSV en: {csv_path}")
        return None, None
    
    # Carga del CSV
    df_ohlc = pd.read_csv(
        csv_path,
        sep=',',
        parse_dates=['time']
    )

    # Renombramiento de Columnas (OHLC)
    df_ohlc.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
    }, inplace=True)

    # Establecer el Índice de Tiempo
    df_ohlc.set_index('time', inplace=True)
    df_ohlc = df_ohlc[['Open', 'High', 'Low', 'Close']].copy()

    # Carga de Trades
    if not os.path.exists(parquet_path):
        print(f"ERROR: No se encontró el archivo Parquet en: {parquet_path}")
        return df_ohlc, None
        
    df_trades = pd.read_parquet(parquet_path)
    
    # Convertir tiempos de trades a datetime para Plotly
    df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
    df_trades['exit_time'] = pd.to_datetime(df_trades['exit_time'])
    
    return df_ohlc, df_trades

# --- 2. GENERACIÓN DEL GRÁFICO ---

def plot_trades_on_ohlc_plotly(df_ohlc: pd.DataFrame, df_trades: pd.DataFrame) -> go.Figure:
    """
    Genera el gráfico interactivo para entradas y salidas.
    """
    
    # --- CONFIGURACIÓN DEL RANGO DE VISTA INICIAL ---
    N_VELAS_INICIALES = 100
    PADDING_Y = 0.0005 
    
    if len(df_ohlc) > N_VELAS_INICIALES:
        df_visible = df_ohlc.iloc[-N_VELAS_INICIALES:]
        start_date = df_visible.index[0]
        end_date = df_visible.index[-1]
    else:
        df_visible = df_ohlc
        start_date = df_ohlc.index[0]
        end_date = df_ohlc.index[-1]

    y_min_visible = df_visible['Low'].min()
    y_max_visible = df_visible['High'].max()
    range_y_start = y_min_visible - PADDING_Y
    range_y_end = y_max_visible + PADDING_Y


    # Trazado de Velas (Candlestick)
    fig = go.Figure(data=[go.Candlestick(
        x=df_ohlc.index,
        open=df_ohlc['Open'],
        high=df_ohlc['High'],
        low=df_ohlc['Low'],
        close=df_ohlc['Close'],
        name='OHLC',
        increasing_line_color='green', 
        decreasing_line_color='red',
        increasing_fillcolor='green',
        decreasing_fillcolor='red',
    )])
    
    #  Preparación de los datos: Separar Entradas y Salidas
    entry_data = df_trades[['entry_time', 'entry_price', 'side']].copy()
    entry_data = entry_data.rename(columns={'entry_time': 'Time', 'entry_price': 'Price'})

    exit_data = df_trades[['exit_time', 'exit_price', 'side']].copy()
    exit_data = exit_data.rename(columns={'exit_time': 'Time', 'exit_price': 'Price'})
    
    
    # --- CREACIÓN DE LOS CUATRO TRAZOS DE DISPERSIÓN ---
    
    # Definiciones de Estilo (Colores y Símbolos)
    styles = {
        'Entrada BUY':      {'color': '#0000FF', 'symbol': 'triangle-up', 'data': entry_data[entry_data['side'] == 'BUY']},     # Azul
        'Salida BUY':       {'color': '#0000FF', 'symbol': 'triangle-down', 'data': exit_data[exit_data['side'] == 'BUY']},     # Cyan
        'Entrada SELL':     {'color': "#CC00FF", 'symbol': 'triangle-down', 'data': entry_data[entry_data['side'] == 'SELL']},    # Amarillo
        'Salida SELL':      {'color': '#CC00FF', 'symbol': 'triangle-up', 'data': exit_data[exit_data['side'] == 'SELL']},      # Lavanda
    }
    
    for name, style in styles.items():
        df_sub = style['data']
        if not df_sub.empty:
            fig.add_trace(go.Scatter(
                x=df_sub['Time'],
                y=df_sub['Price'],
                mode='markers',
                marker=dict(
                    size=10, 
                    color=style['color'], 
                    symbol=style['symbol'], 
                    line=dict(width=1, color='white')
                ),
                name=name,
                hoverinfo='x+y+name' 
            ))

    # Configuración de Layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        xaxis_title='Tiempo',
        yaxis_title='Precio',
        height=650,
        hovermode="x unified",
        dragmode='pan', 
        
        xaxis=dict(
            range=[start_date, end_date], 
            rangeslider_visible=False, 
        ),
        
        yaxis=dict(
            range=[range_y_start, range_y_end], 
            autorange=False, 
        ),
        
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
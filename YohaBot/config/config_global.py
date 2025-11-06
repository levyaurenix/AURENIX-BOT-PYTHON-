import os
import datetime
import MetaTrader5 as mt5




MAX_RATES = 'MO_1'

RATES = {
    "M1": {
        "MO_1": 27600,
        "MO_3": 82800
    },
    "M2": {
        "MO_1": 13800,
        "MO_3": 41400
    },
    "M3": {
        "MO_1": 9200,
        "MO_3": 27600
    },
    "M4": {
        "MO_1": 6900,
        "MO_3": 20700
    },
    "M5": {
        "MO_1": 5520,
        "MO_3": 16560
    },
    "M6": {
        "MO_1": 4600,
        "MO_3": 13800
    },
    "M10": {
        "MO_1": 2760,
        "MO_3": 8280
    },
    "M12": {
        "MO_1": 2300
    },
    "M15": {
        "MO_1": 1840
    },
    "M30": {
        "MO_1": 920
    },
    "MAX_CANDLES_TEMP": 90000
}

# ------------------------------------- #
# --------- RUTAS DE ARCHIVOS --------- #
# ------------------------------------- #

# Ruta de la base de datos de historial
DATABASE_DATA_HISTORY = os.path.join('database', 'data_history')

# Archivo de interfaz
INTERFACE_ROUTE = os.path.join('interfaces', 'interfaces_results.py')

# Estadisitcas
RESULTS_DIR = os.path.join('database', 'stadistics_results')
TRADES_FILE = os.path.join(RESULTS_DIR, 'last_backtest_trades.parquet')
CONFIG_FILE = os.path.join(RESULTS_DIR, 'last_backtest_config.txt')

POSITIONS = os.path.join('database','positions_live','positions.json')

# --- CONFIGURACIÓN DE RUTAS ---
CSV_FILE_PATH = 'database/data_history/FX Vol 20_M1.csv'
PARQUET_FILE_PATH = 'database/stadistics_results/last_backtest_trades.parquet'

# link de interfaz
STREAMLIT_URL = "http://localhost:8501"

# ------------------------------------- #
# --------- XXXX --------- #
# ------------------------------------- #

# los nombres cortos de los Time Frames a sus constantes de MT5.
TIME_FRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# ------------------------------------- #
# --------- CICLOS Y HORARIOS --------- #
# ------------------------------------- #

# Tiempo despues de cierre para obtener los datos
POST_CLOSE_DELAY_SECONDS = 2

# Frecuencia de verificacion de pocision
CHECK_FREQUENCY_CYCLES = 3 

# timeframes a segundos
TIMEFRAME_SECONDS = {'M1': 60, 'M5': 300, 'M6': 360,'M15': 900, 'M30': 1800,'H1': 3600}

TIME_FRAME_MINUTES = {
        # Minutos
        "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M6": 6, 
        "M10": 10, "M12": 12, "M15": 15, "M20": 20, "M30": 30,
        # Horas (H * 60)
        "H1": 60, "H2": 120, "H3": 180, "H4": 240, "H6": 360, 
        "H8": 480, "H12": 720
}

#HORARIOS DE MERCADO
MARKET_SCHEDULES = {
    # -----------------------------------------------------------------
    # ORO (GOLD) - Típico cierre diario y semanal
    'GOLD': {
        # Pausa Diaria (Ej. 4:58 PM a 6:02 PM)
        'DAILY_CLOSE': datetime.time(hour=16, minute=58), 
        'DAILY_OPEN': datetime.time(hour=18, minute=2),   
        # Cierre Semanal (Viernes 4:58 PM a Domingo 6:02 PM)
        'WEEKEND_CLOSE_DAY': 4, # Viernes
        'WEEKEND_CLOSE_TIME': datetime.time(hour=16, minute=58),
        'WEEKEND_OPEN_DAY': 6,  # Domingo
        'WEEKEND_OPEN_TIME': datetime.time(hour=18, minute=2),
    },
    # -----------------------------------------------------------------
    # EURUSD (FOREX) - Cierre semanal típico (No tiene pausa diaria)
    'EURUSD': {
        'DAILY_CLOSE': None, # Sin pausa diaria
        'DAILY_OPEN': None,
        # Cierre Semanal (Viernes 4:59 PM a Domingo 5:05 PM, Hora NY / GMT-4)
        'WEEKEND_CLOSE_DAY': 4, # Viernes
        'WEEKEND_CLOSE_TIME': datetime.time(hour=16, minute=59),
        'WEEKEND_OPEN_DAY': 6,  # Domingo
        'WEEKEND_OPEN_TIME': datetime.time(hour=17, minute=5),
    },
    # -----------------------------------------------------------------
}

# ------------------------------------- #
# ------------------------------------- #
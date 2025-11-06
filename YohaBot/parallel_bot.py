import subprocess
import os
import sys
import argparse


from config.login import DATES_LOGIN
from config.live_strategy import BOTS_CONFIG_LIVE


from live.live import Live
from connection.connection_and_get_dates import connection_and_get_dates


def start_bots():
    """
    Función lanzadora (Padre). Lanza cada bot de la configuración en una nueva 
    ventana de terminal aislada utilizando 'subprocess.Popen' (Windows).

    Cada subproceso se llama a sí mismo pasando el índice del bot para que 
    la nueva terminal ejecute la lógica de trading.
    """

    print(f"🌐 Lanzando {len(BOTS_CONFIG_LIVE)} procesos de trading en paralelo...")

    current_script_path = os.path.abspath(__file__)


    for i, config in enumerate(BOTS_CONFIG_LIVE):

        symbol = config['symbol']
        timeframe = config['timeframe']

        # 1. Definir el título de la ventana
        window_title = f"Bot Trading | {symbol} - {timeframe} (bot {i})--{config['strategy']['name']}"

        # 2. Comando para abrir una NUEVA ventana de terminal en Windows
        command = [
            'start', window_title, 
            'cmd', '/k',           
            sys.executable,        
            current_script_path,   
            '--bot-index', str(i) 
            ]
        
        try:
        
            subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
            print(f"✅ Bot {i} ({symbol}-{timeframe}) lanzado en terminal separada.")

        except Exception as e:

            print(f"❌ Error al lanzar subprocess para {symbol}: {e}")
        
    print("✅ Todos los bots han sido lanzados. Monitoreando procesos...")


def intance_live(index: int):
    """
    Función de ejecución (Hijo). Es llamada por cada subproceso para inicializar
    y correr un único bot de trading en su propia terminal.

    Args:
        index (int): El índice del bot en la lista BOTS_CONFIG.
    """
    # Cada proceso hijo establece su propia conexión y sesión MT5.
    instrument_config = connection_and_get_dates( DATES_LOGIN, BOTS_CONFIG_LIVE[index]['symbol'])
    config = BOTS_CONFIG_LIVE[index]
    
    live_instance = Live()

    # Ejecuta el bucle de trading principal del bot.
    live_instance.run_live(config=config, instrument_config=instrument_config )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot-index', type=int, required=False)
    args = parser.parse_args()

    if args.bot_index is not None:
        # 🛑 MODO EJECUCIÓN ÚNICA (TERMINAL HIJA) 🛑
        
        index = args.bot_index
        
        print(f"\n--- INICIANDO PROCESO BOT ÍNDICE: {index} ---")
        
        # Ejecutamos la función con los argumentos recibidos
        intance_live(index=index) 
        
        print(f"\n--- PROCESO BOT {index} FINALIZADO. ---")
        
    else:
        # 🛑 MODO LANZADOR (TERMINAL PADRE) 🛑
        start_bots()
import argparse
from parallel_bot import start_bots
from backtesting.backtesting import backtesting

# FUNCIÓN PRINCIPAL DE INICIO
def main():
    """
    Define y procesa los argumentos de línea de comandos para seleccionar 
    el modo de operación del sistema: 'live' para trading en MT5 o 'backtesting'.
    """
    
    parser = argparse.ArgumentParser(
        description="Sistema de ejecución de Trading. Seleccione el modo de operación.",
        epilog="Ejemplos: python main.py live | python main.py backtesting"
    ) 
    
    # Definir el argumento posicional 'mode'
    
    parser.add_argument(
        'mode', 
        choices=['live', 'backtesting', 'backtesting_results'],
        help='El modo de ejecución: "live" para MT5 o "backtesting" para Backtrader.'
    )

    # Procesar los argumentos
    args = parser.parse_args()  


    # Llamar a la función correspondiente
    if args.mode == 'live':
        # Lanza los procesos de trading en MetaTrader 5 (MT5).
        start_bots()
    elif args.mode == 'backtesting':
        # Lanza los procesos de backtesting.
        backtesting.run_backtesting()  
    elif args.mode == 'backtesting_results':
        # Lanza los resultados del backtesting.
        backtesting.launch_streamlit_interface()

# PUNTO DE ENTRADA DEL SCRIPT
if __name__ == '__main__':
    main()
import os
import sys
import time
import webbrowser

from config.config_global import (
    TRADES_FILE,
    CONFIG_FILE,
    STREAMLIT_URL,
    INTERFACE_ROUTE
)
from config.backtesting_strategy import BOT_CONFIG_BACKTEST
from config.login import DATES_LOGIN

from backtesting.simulator.trade_simulator import TradeSimulator    
from connection.broker_connection import broker_connection
from connection.connection_and_get_dates import connection_and_get_dates

class Backtesting:


    def launch_streamlit_interface(self):

        """

        Lanza la aplicación en un subproceso, abre la página en el navegador,

        y cierra cualquier proceso anterior usando el puerto 8501.

        """

        # 1. ENCUENTRA LA RUTA Y EL COMANDO DE STREAMLIT

        streamlit_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), INTERFACE_ROUTE)

        streamlit_command = f"{sys.executable} -m streamlit run \"{streamlit_script_path}\" --server.port 8501 --server.headless true"



        print("\n==================================================")

        print("🚀 Lanzando Interfaz de Resultados...")

        print("==================================================")



        # 2. LIBERACIÓN DEL PUERTO (Kill del proceso anterior)

        if sys.platform.startswith('win'):

            # Windows: Encuentra el PID usando el puerto y lo mata

            try:

                # Encuentra el PID asociado al puerto 8501 y lo mata. Esto requiere permisos.

                kill_command = f'FOR /F "tokens=5" %P IN (\'netstat -ano ^| findstr :8501\') DO (TASKKILL /PID %P /F 2>nul)'

                os.system(kill_command)

                time.sleep(1) # Esperar un momento para liberar el puerto

            except Exception:

                # Puede fallar si no hay nada o si faltan permisos, se ignora y continua

                pass



        elif sys.platform.startswith('linux') or sys.platform.startswith('darwin'):

            # Linux/Mac: Encuentra el PID y lo mata con 'kill'

            try:

                # Encuentra el PID del proceso en el puerto 8501 y lo mata (fuser o lsof)

                kill_command = f'lsof -t -i tcp:8501 | xargs kill -9'

                os.system(kill_command)

                time.sleep(1) # Esperar un momento para liberar el puerto

            except Exception:

                pass



        # 3. LANZAR STREAMLIT EN UNA NUEVA TERMINAL

        try:

            if sys.platform.startswith('win'):

                # El comando 'start' lanza un nuevo CMD y ejecuta el proceso allí.

                full_command = f'start "Streamlit Backtesting Results" cmd /k "{streamlit_command}"'

                os.system(full_command)



            elif sys.platform.startswith('linux'):

                # Comando para lanzar una nueva terminal en Linux

                full_command = f'gnome-terminal --title="Streamlit Backtesting Results" -- bash -c "{streamlit_command}; exec bash"'

                os.system(full_command)



            elif sys.platform.startswith('darwin'):

                # Comando para lanzar en Mac Terminal

                full_command = f'open -a Terminal "{streamlit_command}"'

                os.system(full_command)



            else:

                print("❌ Advertencia: Sistema operativo no compatible con lanzamiento de nueva terminal.")

                return



            # 4. ABRIR LA PÁGINA AUTOMÁTICAMENTE

            print(f"\n⏳ Esperando 3 segundos para el inicio del servidor...")

            time.sleep(3)

            webbrowser.open_new_tab(STREAMLIT_URL)



            print(f"\n✅ Interfaz lanzada y abierta en tu navegador.")

            print(f"URL: {STREAMLIT_URL}")

            print("--------------------------------------------------")

            print("NOTA: Una nueva ventana de terminal se abrió para ejecutar el servidor web.")



        except Exception as e:

            print(f"❌ ERROR al lanzar Streamlit o abrir el navegador: {e}")


    def run_backtesting(self):
        """
        Ejecuta el backtest completo, GUARDA los resultados esenciales y lanza la interfaz.
        """

        try: 


            config = BOT_CONFIG_BACKTEST
            symbol = config['symbol']
            timeframe = config['timeframe']
            instrument_config = connection_and_get_dates(DATES_LOGIN, symbol)
            current_equity = config['balance']
            initial_balance = config['balance']
            strategy_config = config['strategy']
            StrategyClass = strategy_config['strategy_class']
            params = strategy_config['params']
            strategy_instance = StrategyClass(params=params, decimal_symbol=instrument_config['decimal_symbol'])
            open_method_name = strategy_config['open_method']
            close_method_name = strategy_config['close_method']
            open_method = getattr(strategy_instance, open_method_name)
            close_method = getattr(strategy_instance, close_method_name)
            candles_required = strategy_config['candles_required'] 



            # --------- Inizializacion clases y Obtencion de datos ---------- #

            trade_manager = TradeSimulator(
                symbol= symbol,
                instrument_config=instrument_config,
                start_cash=config['balance']
                ) 

            df_rates = broker_connection.get_historical_rates(symbol, timeframe)

            print(f'\n*** INICIO DE BACKTESTING ***')
            print(f'Símbolo: {config["symbol"]} | Velas: {len(df_rates)} | Capital Inicial: {trade_manager.balance:.2f}')
            print('==================================================')

            i = candles_required - 1
            total_bars = len(df_rates)

            while i < total_bars:
                current_bar = df_rates.iloc[i]
                current_time = current_bar.name
                current_price = current_bar['close']
                lot_mode = config['lot_mode'] 
                calculated_lot = round(config['lot_dynamic'](current_equity), 2) if lot_mode == True else config['lot_fixed']        
                MIN_LOT = 0.01
                MAX_LOT = 50.0
                lot_size = min(max(calculated_lot, MIN_LOT), MAX_LOT)

                # --- A. PREPARAR SLICE DESLIZANTE (LOOKBACK BARS) ---
                # El df_slice contiene la ventana histórica necesaria (candles_required) + la barra actual.
                df_slice = df_rates.iloc[i - candles_required + 1: i + 1]

                # ======================================================================
                # 🟢 CAMINO 2: HAY POSICIÓN ABIERTA (GESTIÓN DE RIESGO Y CIERRE)
                # ======================================================================
                if trade_manager.position:

                    # Recuperar detalles de la posición abierta
                    pos = trade_manager.position 
                    side = pos['side']
                    sl_mode = pos['sl_mode']
                    sl_price = pos['sl_price']
                    tp_mode = pos['tp_mode']
                    tp_price = pos['tp_price']
                    commission = pos['commission']
                    margin_required = pos['margin']


                    # --- 2.B: VERIFICACIÓN DE MARGIN CALL ---
                    current_equity = trade_manager.get_equity(current_price) 
                    # LLAMADA AL MARGEN: Si Equity cae por debajo del Margen
                    if current_equity < margin_required + commission:
                    
                        print(f"\n⛔ LLAMADA AL MARGEN. Equidad ({current_equity:.2f}) menor a total requerido ({margin_required + commission:.2f}).")

                        trade_manager.close_position(current_bar['close'], 'MARGIN_CALL', current_time)

                        print("⛔ Deteniendo Backtesting debido a (Margin Call) Fallo de Fondos.")
                        break
                    
                    # --- 2.A: VERIFICACIÓN DE STOP LOSS ---
                    close_sl = trade_manager.check_sl(current_bar['low'], current_bar['high'], current_time, side ,sl_price, sl_mode)
                    
                    if not close_sl:
                        # --- 2.C: VERIFICACIÓN DE TAKE PROFIT ---
                        trade_manager.check_tp( close_method, df_slice, current_bar['low'], current_bar['high'], current_time, side, tp_price, tp_mode)
                    
                # ======================================================================
                # 🔴 CAMINO 1: NO HAY POSICIÓN ABIERTA (BÚSQUEDA DE ENTRADA)
                # ======================================================================
                else:                
                    # --- 1.A: Buscar Señal de Entrada ---
                    # Se llama al método de apertura de la estrategia
                    order_entry = open_method(df_slice)
                    position_command = order_entry['command'] # Será 'BUY', 'SELL', 'NONE', o 'ERROR'

                    if position_command in ['BUY', 'SELL']:                    

                        success = trade_manager.open_position(
                            order_entry, 
                            current_price, 
                            lot_size, 
                            current_time
                        )

                        if not success: 
                            print(f"\n⛔ FONDOS INSUFICIENTES ({trade_manager.balance:.2f}). Margen Requerido + comision ({margin_required + commission:.2f}). Deteniendo Backtesting.")
                            break
                        
                        
                # ======================================================================
                # ⏹️ VERIFICACIÓN FINAL Y AVANCE DEL CICLO
                # ======================================================================

                # --- 4. VERIFICACIÓN DE CIERRE EN LA ÚLTIMA BARRA ---
                if i == total_bars - 1 and trade_manager.position:
                    print("\n⚠️ Última barra alcanzada. Cerrando posición abierta al precio final.")
                    trade_manager.close_position(current_price, 'END_OF_BACKTEST', current_time)

                # Avanzar a la siguiente barra
                i += 1


            balance = trade_manager.balance
            # Backtest ha concluido.
            print("\n✅ Backtesting finalizado.")
            print('\n==================================================')
            print(f'Balance Final: {balance:.2f}')
            print('==================================================')

            trades_df = trade_manager.get_trades_history()

            # 2. VALIDACIÓN DE TRADES Y CÁLCULO DE MÉTRICAS
            if trades_df.empty:
                 print("⛔ No se realizaron operaciones. No hay reporte que generar.")
                 return 


            # 3. GUARDAR LOS DATOS ESENCIALES
            try:
                # Historial de trades
                trades_df.to_parquet(TRADES_FILE, index=False)

                final_balance = trade_manager.balance

                total_to_analized = trade_manager.calculate_total_days_analyzed(total_bars,timeframe)
                total_analized = trade_manager.calculate_total_days_analyzed(i,timeframe)

                # Capital (inicial y final)
                with open(CONFIG_FILE, "w") as f:
                    f.write(f"initial_balance:{initial_balance}\n")
                    f.write(f"final_balance:{final_balance}\n")
                    f.write(f"total_days_analyze:{total_analized}\n")
                    f.write(f"total_days_to_analyze:{total_to_analized}\n")
                    f.write(f"decimal_symbol:{instrument_config['decimal_symbol']}\n")
                    f.write(f"symbol:{symbol}\n")
                    f.write(f"timeframe:{timeframe}\n")


                print(f"\n✅ Datos de reporte guardados en \n{TRADES_FILE}, \n{CONFIG_FILE}")

            except Exception as e:
                print(f"❌ ERROR al guardar archivos necesarios para la interfaz: {e}")
                return

            # 4. LANZAR LA INTERFAZ
            self.launch_streamlit_interface()

            return trades_df 

        except KeyboardInterrupt:

            print("\n🚨 Bot detenido por el usuario.")
            return

        except Exception as e:

            print(f"❌ Error crítico: {e}")
            return

backtesting = Backtesting()
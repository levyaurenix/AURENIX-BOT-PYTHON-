from typing import Callable

from connection.broker_connection import broker_connection
from live.config.db_positions import local_db
from strategy.monitor.monitor import monitor
from strategy.strategies.emas import Emas  # Importar la estrategia EMA


class StrategyLogic:

    def __init__(self):
        '''
        Inicializar lógica de estrategia con configuración de EMA.
        Prepara instancia de estrategia para reutilización.
        '''
        self.strategy_instance = None

    # GESTIÓN DE LA ESTRATEGIA DE TENDENCIA (ENTRADA PRINCIPAL)
    def strategy_logic(self, symbol: str, timeframe: str, config: dict, account_balance: int, open_method: str, close_method: str, candles_required: int):
        """
        Función principal de análisis y entrada de la estrategia.

        Orquesta el ciclo de análisis:
        1. Obtiene los datos del mercado.
        2. Inicializa la clase de estrategia EMA con parámetros de config.
        3. Llama al método de análisis de la estrategia.
        4. Procesa el resultado ('BUY', 'SELL', 'NONE', 'ERROR').
        5. Si la señal es 'BUY' o 'SELL', ejecuta la orden en el bróker.
        6. Registra la posición en la base de datos local.
        7. Transfiere el control al módulo de monitoreo.

        Args:
            symbol (str): Símbolo del activo.
            timeframe (str): Marco de tiempo.
            config (dict): Configuración del bot, incluyendo parámetros de estrategia.
            account_balance (int): Balance de la cuenta para cálculos de lotaje.
            open_method (str): Método de apertura (compatibilidad)
            close_method (str): Método de cierre (compatibilidad)
            candles_required (int): Número de velas requeridas para análisis
        """

        try:
            # OBTENER DATOS DEL MERCADO
            df = broker_connection.get_latest_rates(symbol=symbol, timeframe=timeframe, candles_required=candles_required)

            if df.empty:
                print("❌ Error: No se pudieron obtener datos del mercado.")
                return

            # INICIALIZAR ESTRATEGIA EMA
            # Obtener parámetros de estrategia desde config
            strategy_params = config.get('strategy_params', {
                'ema_f': 8,      # EMA rápida
                'ema_m': 21,     # EMA media
                'ema_s': 55,     # EMA lenta
                'atr_period': 14,
                'sl_offset': 2.0,
                'rsi_period': 14,
                'rsi_upper': 70,
                'rsi_lower': 30,
                'atr_mul_cons': 1.5,
                'atr_max_dist_fast': 2.0
            })

            # Obtener decimales del símbolo (típicamente 5 para GOLD)
            decimal_symbol = config.get('decimal_symbol', 5)

            # Crear instancia de estrategia EMA
            ema_strategy = Emas(params=strategy_params, decimal_symbol=decimal_symbol)

            print(f"📊 Ejecutando análisis EMA en {len(df)} velas de {symbol}-{timeframe}")

            # EJECUTAR ANÁLISIS DE ESTRATEGIA
            analysis_result = ema_strategy.order_entry(df=df)

            log_message = analysis_result.get('log_message', 'NONE')

            # Mostrar mensaje de análisis si no es NONE
            if log_message != 'NONE':
                print(log_message)

            position_command = analysis_result.get('command', 'NONE')  # Será 'BUY', 'SELL', 'NONE'
            sl_price = analysis_result.get('sl_price', 0.0)

            # Manejo de comandos
            if position_command == 'NONE' or position_command == 'WAIT':
                print("⏸️ Análisis: No hay señal de entrada. Esperando...")
                return

            if position_command not in ['BUY', 'SELL']:
                print(f"⚠️ Comando no reconocido: {position_command}")
                return

            position_type = position_command

            # --- GESTIÓN DE LA ENTRADA ---
            print(f"💰 SEÑAL {position_type}: Ejecutando orden.")

            # Ejecutar orden de mercado a través del módulo de conexión del bróker
            execution_result = broker_connection.open_position(
                symbol=symbol,
                order_type=position_type,
                sl_price=sl_price,
                config=config,
                balance=account_balance
            )

            # Verificar si la ejecución fue exitosa
            if execution_result is None:
                print("🚫 Fallo al ejecutar la orden. Verificar logs del broker.")
                return

            position_ticket, comment_text = execution_result

            # Obtener contexto de la posición desde MT5
            order_data = broker_connection.get_dates_position(position_ticket)

            # Confirmar ejecución
            if position_ticket:
                print(f"🟢 Orden {position_type} ejecutada con éxito. Ticket: {position_ticket} SL: {sl_price:.5f}")
            else:
                print("🚫 Fallo al ejecutar la orden principal. Verificar logs del broker.")
                return

            # Preparar datos para base de datos local
            db_record_data = {
                'ticket': position_ticket,
                'comment': comment_text,
            }

            # Guardar posición en base de datos local
            if local_db.create_position(symbol=symbol, position_data=db_record_data):
                print(f"✅ TICKET {position_ticket} guardado localmente bajo el símbolo {symbol}.")
            else:
                print(f"❌ Fallo al guardar TICKET {position_ticket} en la base de datos local.")

            # Crear función de cierre para el monitoreo
            def close_function(order_type, data, entry_price):
                '''
                Función de cierre que utiliza la estrategia EMA para decidir cuándo cerrar.
                '''
                return ema_strategy.tp_close(order_type, data, entry_price)

            # Iniciar monitoreo de la posición
            print(f"👁️ Iniciando monitoreo de posición {position_ticket}...")
            monitor.monitor(order_data, symbol, timeframe, close_function, candles_required)

        except Exception as e:
            print(f"❌ Error crítico en strategy_logic: {str(e)}")
            import traceback
            traceback.print_exc()

strategy = StrategyLogic()
import MetaTrader5 as mt5
import pandas as pd
import os
from typing import Dict, Any, Optional,Tuple, Union

from config.config_global import DATABASE_DATA_HISTORY, MAX_RATES, RATES,TIME_FRAMES

AccountInfoType = Union[mt5.AccountInfo, None]
SymbolInfoType = Union[mt5.SymbolInfo, None]

class BrokerConnection:


    # INICIO DE SESIÓN BROKER
    def login_to_broker(selft, login: int, password: str, server: str, symbol: str) -> Tuple[bool, float, AccountInfoType, SymbolInfoType]:
        """
        Autentica la conexión con la cuenta del bróker en MetaTrader 5 (MT5).

        Verifica el saldo de la cuenta y asegura que el símbolo de operación esté
        seleccionado.

        Args:
            login (int): Número de cuenta MT5.
            password (str): Contraseña de la cuenta MT5.
            server (str): Nombre del servidor del bróker.
            symbol (str): Símbolo principal a activar y obtener información.

        Returns:
            tuple: (Estado_Login: bool, Saldo: float, Info_Cuenta, Info_Simbolo)
        """

        # ----------------------------------------------------------------------
        # 1. INICIO DE SESIÓN EN MT5
        # ----------------------------------------------------------------------
        # mt5.login() devuelve False si falla.
        if not mt5.login(login, password=password, server=server):
            error_msg = mt5.last_error()
            print(f"❌ FALLO en el LOGIN. Cuenta {login} / Servidor {server}. Error: {error_msg}")
            return False, 0.0, None, None # Retorno de error unificado

        print(f"✅ ÉXITO en el LOGIN. Cuenta {login} conectada.")

        # ----------------------------------------------------------------------
        # 2. OBTENCIÓN DE DATOS DE LA CUENTA (Account Info)
        # ----------------------------------------------------------------------
        account_info = mt5.account_info()

        if account_info is None:
            error_msg = mt5.last_error()
            print(f"❌ FALLO al obtener Account Info. Login exitoso, pero datos no disponibles. Error: {error_msg}")
            # La conexión es inestable si falla aquí, tratamos como fallo general
            return False, 0.0, None, None

        account_balance = account_info.balance

        print(f"✅ DATOS DE CUENTA: Saldo: {account_balance:.2f} {account_info.currency}")

        # ----------------------------------------------------------------------
        # 3. ACTIVACIÓN Y OBTENCIÓN DE DATOS DEL SÍMBOLO
        # ----------------------------------------------------------------------

        # Asegurar que el símbolo esté seleccionado (visible en Market Watch)
        if not mt5.symbol_select(symbol, True):
            error_msg = mt5.last_error()
            print(f"⚠️ ADVERTENCIA: No se pudo seleccionar/activar el símbolo {symbol}. Error: {error_msg}")
            # Continuamos con el login, pero con advertencia y Info_Simbolo=None
            symbol_info = None
        else:
            # Si se selecciona, intentamos obtener la información detallada
            symbol_info = mt5.symbol_info(symbol)

            if symbol_info is None:
                error_msg = mt5.last_error()
                print(f"❌ FALLO al obtener Symbol Info para {symbol}. Error: {error_msg}")
                # Esto es un error crítico para operar, pero el login fue exitoso.
                return True, account_balance, account_info, None

            print(f"✅ SÍMBOLO ACTIVADO: {symbol}. Ticks: {symbol_info.description}")


        # ----------------------------------------------------------------------
        # 4. RETORNO FINAL
        # ----------------------------------------------------------------------

        # Retorna True solo si el login y el acceso a la cuenta fueron exitosos.
        return True, account_balance, account_info, symbol_info


    # ACTUALIIZAR BALANCE
    def get_and_update_balance(selft) -> Optional[float]:
        """
        Recupera el balance de la cuenta de trading del terminal MT5.

        Returns:
            float | None: El balance actual de la cuenta si la operación es exitosa,
                        o None si hay un fallo al obtener la información.
        """
        # 1. Obtener la información de la cuenta
        account_info = mt5.account_info()

        if account_info is None:
            print("❌ Error al obtener la información de la cuenta desde MT5.")
            return None

        # 2. Extraer el balance
        # El atributo 'balance' contiene el valor del balance actual.
        current_balance = account_info.balance

        print(f"✅ Balance de cuenta actualizado: {current_balance:.2f}")

        return current_balance


    # OBTENER ÚLTIMO TICK/PRECIO
    def get_last_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        '''
        Obtener último tick/precio disponible para un símbolo específico.

        Utiliza MT5 symbol_info_tick() para obtener la información más actual
        del precio bid/ask/last del símbolo solicitado.

        Args:
            symbol (str): Símbolo a consultar (ej: 'GOLD')

        Returns:
            Dict con datos del tick (bid, ask, last, time) o None si error
        '''
        try:
            # Verificar que MT5 esté inicializado
            if not mt5.initialize():
                print(f"❌ Error inicializando MT5 para obtener tick de {symbol}")
                return None

            # Obtener último tick del símbolo
            tick_info = mt5.symbol_info_tick(symbol)

            if tick_info is None:
                error_msg = mt5.last_error()
                print(f"⚠️ No se pudo obtener tick para {symbol}. Error: {error_msg}")
                return None

            # Convertir a diccionario para fácil acceso
            tick_data = {
                'bid': tick_info.bid,
                'ask': tick_info.ask,
                'last': tick_info.last,
                'time': tick_info.time,
                'price': tick_info.last,  # Alias para compatibilidad
                'symbol': symbol
            }

            return tick_data

        except Exception as e:
            print(f"❌ Error obteniendo tick de {symbol}: {e}")
            return None


    # OBTENER VELAS (RATES)
    def get_latest_rates(selft, symbol: str, timeframe: str, candles_required: int) -> pd.DataFrame:
        """
        Recupera las últimas CANDLES_TO_GET velas (datos históricos) para un símbolo y Time Frame.

        Convierte la data obtenida de MT5 (array) a un Pandas DataFrame,
        estableciendo el índice de tiempo para facilitar el análisis posterior.

        Args:
            symbol (str): Símbolo del activo (ej. 'GOLD#').
            timeframe (str): Clave del marco de tiempo (ej. 'M5'), debe existir en TIME_FRAMES.

        Returns:
            pd.DataFrame: DataFrame de Pandas con las columnas 'open', 'high', 'low', 'close', etc.
        """
        if timeframe not in TIME_FRAMES:
            print(f"Error: El marco de tiempo '{timeframe}' no es reconocido.")
            return pd.DataFrame()

        try:
            timeframe = TIME_FRAMES[timeframe]

            # 1. Recuperar los datos desde MT5
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, candles_required)

            if rates is None or len(rates) == 0:
                print(f"Advertencia: No se pudieron obtener datos para {symbol} en {timeframe}.")
                return pd.DataFrame()

            # 2. Convertir el array de MT5 a DataFrame de Pandas
            rates_frame = pd.DataFrame(rates)

            # 3. Formatear la columna de tiempo y establecer el índice
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
            rates_frame = rates_frame.set_index('time')

            return rates_frame

        except Exception as e:
            print(f"❌ Error al conectar o obtener datos de MT5: {e}")
            return pd.DataFrame()

    # ==============================================================================
    # FUNCIONES DE POSICIONES Y TRADING
    # ==============================================================================

    # EJECUTAR ORDEN
    def open_position(selft, symbol: str, order_type: str, sl_price: float, config: dict, balance: int) -> tuple[int, str] | None:
        """
        Función de envío de órdenes a MetaTrader 5 (MT5). Ejecuta una orden de mercado
        ('BUY' o 'SELL') con un Stop Loss (SL) fijo calculado previamente.

        Args:
            symbol (str): Símbolo del activo.
            order_type (str): Tipo de la posición ('BUY' o 'SELL').
            sl_price (float): Precio de Stop Loss (SL) precalculado.
            magic_number (int): Número mágico para la orden.
            timeframe (str): Marco de tiempo (para el monitor).
            lot_size (float): Volumen de la orden.

        Returns:
            int | None: El número de ticket de la posición si la orden se ejecuta con éxito, None si falla.
        """

        # 1. Mapeo y Verificación Inicial
        order_map = {'BUY': mt5.ORDER_TYPE_BUY, 'SELL': mt5.ORDER_TYPE_SELL}
        if order_type not in order_map:
            print(f"❌ (ERROR PREP) Tipo de orden no válido: {order_type}")
            return None
        mt5_order_type = order_map[order_type]

        magic_number = config['magic_number']
        timeframe = config['timeframe']
        lot_size = config['lot_size']

        # 2. Configuración de Punto y Información del Símbolo
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ (ERROR MT5) No se pudo obtener información del símbolo {symbol}")
            return None

        point = symbol_info.point

        if symbol_info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            print(f"❌ (ERROR PREP) El trading para {symbol} no está permitido en este momento")
            return None

        # 3. Definición del Precio de Ejecución
        if order_type == 'BUY':
            price = mt5.symbol_info_tick(symbol).ask
        else:  # 'SELL'
            price = mt5.symbol_info_tick(symbol).bid

        if price is None:
            print(f"❌ (ERROR MT5) No se pudo obtener el precio para {symbol}")
            return None

        # 4. Configurar Comentario de Identificación
        comment = f"BOT_{symbol}_{timeframe}_{order_type}_{lot_size}_{magic_number}"

        # 5. Estructura de la Orden
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5_order_type,
            "price": price,
            "sl": sl_price,
            "tp": 0.0,
            "deviation": 10,
            "magic": magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 6. Envío de la Orden
        result = mt5.order_send(request)

        # 7. Verificación del Resultado
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_desc = {
                mt5.TRADE_RETCODE_INVALID_STOPS: "Stops inválidos (SL/TP)",
                mt5.TRADE_RETCODE_INVALID_VOLUME: "Volumen inválido",
                mt5.TRADE_RETCODE_NO_MONEY: "Fondos insuficientes",
                mt5.TRADE_RETCODE_TRADE_DISABLED: "Trading deshabilitado",
                mt5.TRADE_RETCODE_MARKET_CLOSED: "Mercado cerrado",
                mt5.TRADE_RETCODE_INVALID_PRICE: "Precio inválido",
                mt5.TRADE_RETCODE_REQUOTE: "Requote necesario",
            }.get(result.retcode, "Error desconocido")

            print(f"❌ (FALLO ORDEN) {error_desc}. Código: {result.retcode}")
            print(f"   Detalles: {result}")
            return None

        # 8. Extracción del Ticket
        ticket = result.order
        print(f"✅ (ÉXITO) Orden ejecutada. Ticket: {ticket} | {order_type} {lot_size} lotes de {symbol} @ {price}")

        return ticket, comment

    # CERRAR POSICIÓN
    def close_position(selft, ticket: int, symbol: str, lot_size: float, magic_number: int, reason: str = "Manual") -> bool:
        """
        Cierra una posición específica identificada por su ticket.

        Determina automáticamente si la posición es BUY o SELL y ejecuta
        la orden de cierre correspondiente.

        Args:
            ticket (int): Número de ticket de la posición a cerrar.
            symbol (str): Símbolo del activo.
            lot_size (float): Volumen de la posición (debe coincidir con el original).
            magic_number (int): Número mágico de la posición.
            reason (str): Motivo del cierre (para el comentario).

        Returns:
            bool: True si la posición se cerró exitosamente, False en caso contrario.
        """
        # 1. Obtener información de la posición abierta
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            print(f"❌ (ERROR BUSQ) No se encontró la posición con ticket {ticket}")
            return False

        position = positions[0]

        # 2. Determinar el tipo de cierre basado en el tipo de posición original
        if position.type == mt5.ORDER_TYPE_BUY:
            # Para cerrar una posición BUY, debemos SELL
            close_type = mt5.ORDER_TYPE_SELL
            close_price = mt5.symbol_info_tick(symbol).bid
        else:  # position.type == mt5.ORDER_TYPE_SELL
            # Para cerrar una posición SELL, debemos BUY
            close_type = mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(symbol).ask

        if close_price is None:
            print(f"❌ (ERROR PRECIO) No se pudo obtener precio de cierre para {symbol}")
            return False

        # 3. Crear la solicitud de cierre
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": close_type,
            "position": ticket,  # Especifica la posición a cerrar
            "price": close_price,
            "deviation": 10,
            "magic": magic_number,
            "comment": f"CLOSE_{reason}_{ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 4. Enviar la orden de cierre
        result = mt5.order_send(close_request)

        # 5. Verificar el resultado
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_code = result.retcode
            error_desc = f"Error en cierre de posición: Código {error_code}"
            print(f"❌ (FALLO CIERRE) {error_desc}")
            print(f" Detalles del resultado: {result}")
            print(f" Código de error MT5: {error_code} - {error_desc}")
            return False

        print(f"✅ (ÉXITO CIERRE) Posición {ticket} cerrada exitosamente por {reason}")
        return True


    # VERIFICAR POSICIÓN ABIERTA
    def is_position_open(selft, ticket: int) -> bool:
        """
        Verifica el estado de una posición específica consultando al bróker por su ticket.

        Args:
            ticket (int): El número de ticket de la posición a verificar.

        Returns:
            bool: True si la posición sigue abierta, False si fue cerrada.
        """
        if not mt5.initialize():
            print("❌ Error al inicializar MT5 en is_position_open.")
            return False

        # mt5.positions_get(ticket=ticket) devuelve una tupla de objetos
        positions = mt5.positions_get(ticket=ticket)

        if positions is None:
            print(f"❌ Error al consultar posición {ticket}. Código: {mt5.last_error()}")
            # Si hay un error de conexión/consulta, asumimos que sigue abierta por seguridad.
            return True

        # Si la tupla no está vacía (len > 0), la posición está abierta.
        return len(positions) > 0


    # OBTENER POSICIONES ABIERTAS
    def get_open_positions(selft, symbol: str, magic_number: int = None) -> list[dict]:
        """
        Obtiene todas las posiciones abiertas para un símbolo y las retorna como una lista de diccionarios.

        Permite un filtrado opcional por el Magic Number para gestionar solo las posiciones del bot.

        Args:
            symbol (str): Símbolo del activo (ej. 'XAUUSD').
            magic_number (int, optional): Número mágico para filtrar. Si es None, no hay filtro.

        Returns:
            list[dict]: Una lista de diccionarios con los datos clave de cada posición.
                        Retorna una lista vacía si no hay posiciones o si falla la conexión.
        """

        # 1. Solicitar las posiciones para el símbolo
        positions_mt5 = mt5.positions_get(symbol=symbol)

        if positions_mt5 is None:
            print(f"❌ Error al obtener posiciones para {symbol}. Código de error: {mt5.last_error()}")
            return []

        if len(positions_mt5) == 0:
            return []

        position_list = []

        # 2. Iterar, filtrar (si aplica) y convertir a diccionario
        for pos in positions_mt5:

            # Lógica de FILTRADO por Magic Number.
            if magic_number is not None and pos.magic != magic_number:

                continue
            # Saltar las posiciones que no coinciden

            # Almacenar los datos clave de la posición en un diccionario
            position_list.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'volume': pos.volume,
                'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'time': pd.to_datetime(pos.time, unit='s'),
                'magic': pos.magic,
                'profit': pos.profit
            })

        return position_list

    # OBTENER DATOS HISTORICOS Y GUARDAR EN LOCAL

    # Crear el directorio si no existe
    os.makedirs(DATABASE_DATA_HISTORY, exist_ok=True)

    # Generar nombre
    def get_data_filename(self, symbol: str, timeframe: str) -> str:
        """Genera el nombre del archivo de datos basado en el activo y TF."""
        # Asegura que GOLD# se convierte a GOLD__M1.csv para el nombre de archivo
        safe_symbol = symbol.replace('#', '_')
        safe_symbol = symbol.replace(' ', '_')
        return os.path.join(DATABASE_DATA_HISTORY, f"{safe_symbol}_{timeframe}_{MAX_RATES}.csv")

    # Obtener datos y guardarlos
    def get_historical_rates(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Descarga el número máximo de velas históricas, priorizando la carga de
        datos locales si existen. Si descarga, guarda el DataFrame completo en CSV.

        :param symbol: Símbolo de trading (ej: "EURUSD").
        :param timeframe: Marco de tiempo (ej: "H1").
        :param MAX_RATES: Máximo de velas a descargar.
        :param TIME_FRAMES: Diccionario de timeframes.
        :return: DataFrame de Pandas con los datos de las velas.
        """

        filename = broker_connection.get_data_filename(symbol, timeframe)

        # --- 1. VERIFICAR SI EL ARCHIVO EXISTE Y CARGAR ---
        if os.path.exists(filename):
            print(f"✅ Datos locales encontrados: '{filename}'. Cargando...")
            try:
                # parse_dates=True y index_col='time' para mantener el índice datetime
                df = pd.read_csv(filename, index_col='time', parse_dates=True)

                # Verificación adicional para asegurar que el índice es datetime
                if not pd.api.types.is_datetime64_any_dtype(df.index):
                    df.index = pd.to_datetime(df.index)

                print(f"✅ Cargadas {len(df)} velas de datos locales.")
                return df
            except Exception as e:
                print(f"❌ Error al leer el archivo CSV: {e}. El archivo podría estar corrupto. Intentando descargar.")

        # --- 2. DESCARGAR DE MT5 CON LIMITE DE VELAS ---

        if not mt5.initialize():
            print("❌ Error al inicializar MT5. Verifique la conexión.")
            return pd.DataFrame()

        mt5_timeframe = TIME_FRAMES.get(timeframe)
        if not mt5_timeframe:
            print(f"Error: El marco de tiempo '{timeframe}' no es compatible.")
            mt5.shutdown()
            return pd.DataFrame()

        try:
            print(f"⏳ Iniciando descarga de las últimas {RATES[timeframe][MAX_RATES]} velas de {symbol} en {timeframe}...")

            rates = mt5.copy_rates_from_pos(
                symbol,
                mt5_timeframe,
                0,
                RATES[timeframe][MAX_RATES]
            )

            if rates is None or len(rates) == 0:
                print(f"❌ No se obtuvieron datos de MT5 para {symbol} en {timeframe}.")
                return pd.DataFrame()

            # 3. PROCESAR, LIMPIAR Y GUARDAR
            rates_frame = pd.DataFrame(rates)

            # Conversión del tiempo de Unix a datetime y establece el índice
            rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
            rates_frame = rates_frame.set_index('time')

            # Elimina duplicados si los hubiera
            rates_frame = rates_frame[~rates_frame.index.duplicated(keep='first')]

            # Guarda todas las columnas del DataFrame (open, high, low, close, tick_volume, spread, real_volume)
            rates_frame.to_csv(filename, index=True, index_label='time')

            print(f"\n==================================================")
            print(f"💾 Proceso COMPLETADO. TOTAL VELAS DESCARGADAS: {len(rates_frame)}")
            print(f"==================================================")

            return rates_frame

        except Exception as e:
            print(f"\n❌ Error durante la descarga o procesamiento de MT5: {e}")
            return pd.DataFrame()
        finally:
            mt5.shutdown()


broker_connection = BrokerConnection()
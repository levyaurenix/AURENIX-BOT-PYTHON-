import time
import datetime
import pandas as pd

from config.config_global import (
    TIMEFRAME_SECONDS,
    POST_CLOSE_DELAY_SECONDS,
    MARKET_SCHEDULES,
)

from connection.broker_connection import broker_connection


class Cycles:

    def __init__(self):
        self.broker_rhythm_offset = None    
        self.timeframe_seconds = None
        self.next_candle_time_ts = None

    #CALCULAR HORARIO
    def get_sleep_time(self, now: datetime.datetime, symbol: str) -> float:
        """
        Calcula el tiempo de espera hasta la próxima apertura de mercado,
        basándose en la hora local y los horarios específicos del símbolo.
        """
        if symbol not in MARKET_SCHEDULES:
            print(f"❌ Horario: Símbolo '{symbol}' no configurado. Asumiendo abierto.")
            return 0.0

        schedule = MARKET_SCHEDULES[symbol]
        weekday = now.weekday()
        now_time = now.time()
        
        # 1. Chequeo de Cierre de Criptomonedas (24/7)
        if schedule['WEEKEND_CLOSE_DAY'] is None and schedule['DAILY_CLOSE'] is None:
            return 0.0 # BTC está siempre abierto

        # 2. Chequeo de Cierre Semanal (Viernes Cierre a Domingo Apertura)
        wk_close_day = schedule['WEEKEND_CLOSE_DAY']
        wk_open_day = schedule['WEEKEND_OPEN_DAY']
        wk_close_time = schedule['WEEKEND_CLOSE_TIME']
        wk_open_time = schedule['WEEKEND_OPEN_TIME']
        
        if wk_close_day is not None and wk_open_day is not None:
            
            # A. Es Viernes y ya pasó la hora de cierre O es Sábado
            is_weekend_close_window = (weekday == wk_close_day and now_time >= wk_close_time) or \
                                    (weekday == 5)
            
            # B. Es Domingo antes de la hora de apertura
            is_weekend_open_window = (weekday == wk_open_day and now_time < wk_open_time)
            
            if is_weekend_close_window or is_weekend_open_window:
                print(f"😴 {symbol}: Fin de semana (Cierre).")
                
                # Calcular el próximo despertar (Domingo a la hora de apertura)
                next_open = datetime.datetime.combine(now.date(), wk_open_time)
                
                # Ajustar al Domingo más próximo (o el día de apertura si es diferente)
                days_to_open = (wk_open_day - weekday + 7) % 7 
                next_open += datetime.timedelta(days=days_to_open)
                
                # Si la hora de apertura ya pasó esta semana (ej. es Lunes), avanzar al próximo ciclo semanal
                if next_open <= now:
                    next_open += datetime.timedelta(days=7)
                    
                sleep_seconds = (next_open - now).total_seconds()
                return max(sleep_seconds, 120.0)

        # 3. Chequeo de Pausa Diaria (Solo aplica a GOLD en este ejemplo)
        dly_close_time = schedule['DAILY_CLOSE']
        dly_open_time = schedule['DAILY_OPEN']
        
        # Se aplica solo de Lunes a Viernes
        if weekday >= 0 and weekday <= 4 and dly_close_time is not None:
            if dly_close_time <= now_time < dly_open_time:
                print(f"😴 {symbol}: Pausa Diaria.")
                next_open = datetime.datetime.combine(now.date(), dly_open_time)
                sleep_seconds = (next_open - now).total_seconds()
                return max(sleep_seconds, 60.0)

        return 0.0

    # SINCRONIZAR CIERRE DE VELA Y DESCARGAR DATOS
    def candle_closing_sync(self, symbol: str, timeframe: str, candles_required: int) -> pd.DataFrame | None:
            """
            Sincroniza el bot con el bróker, calcula el próximo cierre de vela y bloquea 
            la ejecución hasta ese momento, devolviendo los datos actualizados del mercado.
            
            Esta función es de tipo "bloqueante" y está diseñada para ser llamada 
            repetidamente dentro del bucle de análisis (`analysis_entry_loop`).
            
            Args:
                symbol (str): Símbolo del activo (ej. 'XAUUSD').
                timeframe (str): Marco de tiempo (ej. 'M15').
                
            Returns:
                pd.DataFrame | None: DataFrame de Pandas con los datos de las últimas
                                    velas, o None si falla la sincronización o la descarga.
            """

            
            # 1. INICIALIZACIÓN (Solo se ejecuta en la primera llamada)
            if self.broker_rhythm_offset is None:
                if timeframe not in TIMEFRAME_SECONDS:
                    print(f"❌ Error: Timeframe '{timeframe}' no soportado.")
                    return None
                
                self.timeframe_seconds = TIMEFRAME_SECONDS[timeframe]
                print("⏳ Obteniendo datos iniciales para establecer el ritmo del broker...")
                initial_data = broker_connection.get_latest_rates(symbol, timeframe, candles_required)
                
                if initial_data.empty or len(initial_data) < 2:
                    print("❌ Fallo al obtener datos iniciales. No se puede sincronizar.")
                    return None
                    
                # Determinar el 'offset' (ritmo) basándose en la última vela cerrada
                last_closed_candle_start_time_ts = initial_data.index[-2].timestamp()
                self.broker_rhythm_offset = last_closed_candle_start_time_ts % self.timeframe_seconds
                
                # Calcular el primer próximo cierre alineado con el ritmo del bróker
                now_ts = time.time()
                # Calcula el inicio de la vela actual alineado con el offset
                current_candle_start_ts = ((now_ts - self.broker_rhythm_offset) // self.timeframe_seconds) * self.timeframe_seconds + self.broker_rhythm_offset
                self.next_candle_time_ts = current_candle_start_ts + self.timeframe_seconds
                
                # Si ya estamos después del cierre calculado, saltar al siguiente cierre
                if self.next_candle_time_ts < now_ts:
                    self.next_candle_time_ts+= self.timeframe_seconds
                    
                print(f"🔄 Sincronizado. Ritmo (Offset): {self.broker_rhythm_offset:.3f}s. Próximo Cierre Inicial: {datetime.datetime.fromtimestamp(self.next_candle_time_ts).strftime('%H:%M:%S')}")
                
            # --- LÓGICA DE ESPERA Y DESCARGA (Para cada ciclo) ---
            
            while True:
                now = datetime.datetime.now()
                now_ts = now.timestamp()
                
                # 2. GESTIÓN DE SUEÑO PROFUNDO (Cierre de Mercado)
                sleep_closed = self.get_sleep_time(now=now, symbol=symbol)
                
                if sleep_closed > 0:
                    print(f"💤 Mercado Cerrado/Pausa: Durmiendo {int(sleep_closed)}s.")
                    time.sleep(sleep_closed)
                    
                    # Recalcular el próximo cierre AL DESPERTAR, alineando con la hora actual
                    now_ts = time.time()
                    current_candle_start_ts = ((now_ts - self.broker_rhythm_offset) // self.timeframe_seconds) * self.timeframe_seconds + self.broker_rhythm_offset
                    self.next_candle_time_ts = current_candle_start_ts + self.timeframe_seconds
                    
                    if self.next_candle_time_ts < now_ts:
                        self.next_candle_time_ts += self.timeframe_seconds
                        
                    continue # Vuelve al inicio del while para calcular la espera
                    
                # --- MERCADO ABIERTO ---

                # 3. CÁLCULO DEL TIEMPO DE ESPERA (Precisión de vela)
                wait_time = self.next_candle_time_ts - now_ts + POST_CLOSE_DELAY_SECONDS

                # Corrección de Ejecución Tardía (Si el bucle anterior tardó demasiado)
                if wait_time <= 0:
                    print(f"⚠️ Ejecución tardía ({wait_time:.2f}s). Reajustando el próximo cierre.")
                    while wait_time <= 0:
                        self.next_candle_time_ts += self.timeframe_seconds
                        now_ts = time.time()
                        wait_time = self.next_candle_time_ts - now_ts + POST_CLOSE_DELAY_SECONDS
                
                # 4. DORMIR Y PETICIÓN
                next_request_dt = datetime.datetime.fromtimestamp(self.next_candle_time_ts + POST_CLOSE_DELAY_SECONDS)
                print(f"⏸️ Esperando {int(wait_time)}s. Petición a: {next_request_dt.strftime('%H:%M:%S')} (Cierre: {datetime.datetime.fromtimestamp(self.next_candle_time_ts).strftime('%H:%M:%S')})")
                time.sleep(wait_time) 

                # 5. DESPERTAR Y DESCARGAR DATOS
                print(f"\n🚨 {datetime.datetime.now().strftime('%H:%M:%S')}: Pidiendo datos al broker...")
                data = broker_connection.get_latest_rates(symbol, timeframe, candles_required)
                
                if data is None or data.empty or len(data) < 2:
                    print("❌ Fallo al obtener datos. Reintentando la espera.")
                    self.next_candle_time_ts += self.timeframe_seconds # Ajustamos para el siguiente ciclo
                    continue
                

                # 6. Preparar para el siguiente ciclo Y RETORNAR
                self.next_candle_time_ts += self.timeframe_seconds
                
                print(f"--- CIERRE DE VELA DETECTADO: {data.index[-2].strftime('%Y-%m-%d %H:%M:%S')} ---")
                
                return data # Devolvemos los datos al bucle principal
            
            # --- CONFIGURACIÓN DE CIERRES POR SÍMBOLO (hora local GMT-4) ---


cycles = Cycles()
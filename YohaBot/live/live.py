#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
YOHABOT LIVE TRADING SYSTEM
===========================

Sistema de trading en vivo con monitoreo activo y heartbeat continuo.
Implementa:
- Heartbeat cada M1 para mantener bot "vivo" visualmente
- Monitoreo de posiciones existentes
- Análisis continuo de mercado
- Manejo robusto de errores

EJECUCIÓN:
python main.py live

CONFIGURACIÓN:
- VERBOSE = True para narración completa
- Heartbeat automático cada 60 segundos
- Sincronización con cierres de vela
'''

import time
from typing import Dict, Any
from datetime import datetime

from connection.broker_connection import broker_connection
from connection.terminal_connection import teminal_connection
from live.config.db_positions import local_db
from strategy.monitor.monitor import monitor
from live.config.schedules import cycles
from strategy.strategy_logic import strategy

# === CONFIGURACIÓN DE NARRACIÓN EN VIVO ===
VERBOSE = True  # True para heartbeat activo, False para modo silencioso

def tf_seconds(tf: str) -> int:
    '''
    Convertir timeframe a segundos para cálculos de sincronización.
    Soporta desde M1 hasta H4 para máxima compatibilidad.
    '''
    timeframe_map = {
        "M1": 60, "M2": 120, "M3": 180, "M5": 300, "M6": 360, "M10": 600, "M12": 720,
        "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400
    }
    return timeframe_map.get(tf, 60)  # Default M1 si no se encuentra

def next_candle_close_epoch(now_epoch: float, period: int) -> int:
    '''
    Calcular timestamp del próximo cierre de vela.
    Redondea hacia arriba para sincronización precisa.
    '''
    return int(((int(now_epoch) // period) + 1) * period)

def heartbeat_print(symbol: str, tf: str, now_epoch: float, close_epoch: int, last_price: float):
    '''
    Imprimir heartbeat para mantener bot visualmente activo.
    Muestra símbolo, timeframe, tiempo actual, próximo cierre y precio.
    '''
    if VERBOSE:
        current_time = datetime.fromtimestamp(now_epoch).strftime("%H:%M:%S")
        close_time = datetime.fromtimestamp(close_epoch).strftime("%H:%M:%S")
        remaining = int(close_epoch - now_epoch)
        print(f"💓 {symbol}-{tf} | {current_time} -> {close_time} (⏰{remaining}s) | 💰{last_price}")

def decision_print(symbol: str, tf: str, msg: str):
    '''
    Imprimir decisiones de trading para seguimiento.
    '''
    if VERBOSE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"🧠 [{timestamp}] {symbol}-{tf} -> {msg}")

def status_print(msg: str, emoji: str = "ℹ️"):
    '''
    Imprimir estados generales del sistema.
    '''
    if VERBOSE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{emoji} [{timestamp}] {msg}")


class Live:
    '''
    Clase principal para trading en vivo con heartbeat activo.

    Implementa ciclo completo de trading:
    1. Heartbeat continuo (cada M1)
    2. Verificación de posiciones existentes
    3. Monitoreo de posiciones activas
    4. Análisis continuo de mercado
    5. Ejecución de estrategias
    '''

    def __init__(self):
        '''
        Inicializar sistema de trading en vivo.
        '''
        self.is_running = False
        self.heartbeat_active = True

    def run_live(self, config: Dict[str, Any], instrument_config: Dict[str, Any]):
        '''
        Función principal del ciclo de vida del bot.

        Ejecuta heartbeat continuo + análisis de estrategia de forma coordinada.
        El heartbeat mantiene al bot "vivo" visualmente mientras la estrategia
        opera en su timeframe configurado.

        Args:
            config: Configuración de trading (symbol, timeframe, etc.)
            instrument_config: Configuración del instrumento (balance, etc.)
        '''
        status_print("🚀 Iniciando YohaBot Live Trading System", "🚀")

        # Extraer configuración
        symbol = config['symbol']
        timeframe = config['timeframe']
        account_balance = instrument_config['account_balance']

        # Variables para estrategia (definir según tu configuración)
        candles_required = config.get('candles_required', 100)
        open_method = config.get('open_method', 'market')
        close_method = config.get('close_method', 'trailing')

        status_print(f"📊 Configuración: {symbol} en {timeframe} | Balance: ${account_balance:,.2f}")

        self.is_running = True

        try:
            # CICLO PRINCIPAL DE TRADING
            while self.is_running:

                # === FASE 1: HEARTBEAT Y SINCRONIZACIÓN ===
                status_print(f"💓 Iniciando heartbeat y sincronización con {timeframe}")
                self._heartbeat_with_strategy_sync(symbol, timeframe)

                # === FASE 2: VERIFICACIÓN DE POSICIONES EXISTENTES ===
                status_print("🔍 Verificando posiciones existentes...")
                position_ticket = self._check_existing_positions(symbol)

                # === FASE 3: MONITOREO DE POSICIÓN ACTIVA ===
                if position_ticket:
                    status_print(f"👁️ Monitoreando posición activa: {position_ticket}")
                    # Monitorear hasta que se cierre la posición
                    self._monitor_active_position(position_ticket, symbol, timeframe, close_method, candles_required)
                    # Actualizar balance después del cierre
                    account_balance = broker_connection.get_and_update_balance()
                    status_print("✅ Monitoreo finalizado. Continuando análisis...")

                # === FASE 4: ANÁLISIS DE MERCADO Y NUEVA ENTRADA ===
                status_print("🔎 Iniciando análisis de mercado para nueva entrada...")
                self._market_analysis_cycle(
                    symbol, timeframe, config, account_balance,
                    open_method, close_method, candles_required
                )

        except KeyboardInterrupt:
            status_print("🛑 Bot detenido por el usuario", "🛑")
            self.is_running = False

        except Exception as e:
            status_print(f"❌ Error crítico: {e}", "❌")
            time.sleep(30)  # Pausa antes de reintentar

        finally:
            # Shutdown ordenado
            status_print("🔌 Cerrando conexiones...", "🔌")
            teminal_connection.shutdown_terminal()
            status_print("✅ YohaBot finalizado correctamente", "✅")

    def _last_tick(self, symbol: str) -> float:
        '''
        Obtener último precio/tick disponible del broker.

        Intenta múltiples campos (last, bid, ask, price) para
        máxima compatibilidad con diferentes brokers.

        Args:
            symbol: Símbolo a consultar

        Returns:
            float: Último precio disponible o None si error
        '''
        try:
            tick_data = broker_connection.get_last_tick(symbol)
            if not tick_data:
                return None

            # Si viene como dict
            if isinstance(tick_data, dict):
                for price_field in ('last', 'bid', 'ask', 'price'):
                    price_value = tick_data.get(price_field)
                    if price_value is not None:
                        return float(price_value)
                return None

            # Si viene como objeto con atributos
            price_value = (
                getattr(tick_data, 'last', None) or
                getattr(tick_data, 'bid', None) or
                getattr(tick_data, 'ask', None) or
                getattr(tick_data, 'price', None)
            )
            return float(price_value) if price_value is not None else None

        except Exception as e:
            if VERBOSE:
                print(f"⚠️ Error obteniendo precio de {symbol}: {e}")
            return None

    def _heartbeat_with_strategy_sync(self, symbol: str, timeframe: str):
        '''
        Heartbeat continuo con sincronización al cierre de vela.

        Mantiene bot "vivo" imprimiendo cada ~60 segundos (M1)
        independientemente del timeframe de la estrategia.

        Args:
            symbol: Símbolo a monitorear
            timeframe: Timeframe de la estrategia
        '''
        # Calcular períodos
        strategy_period = tf_seconds(timeframe)  # Período de la estrategia (ej: M6 = 360s)
        heartbeat_period = 60  # Heartbeat cada M1 (60s) para mantener activo

        # Sincronizar con próximo cierre de vela de la estrategia
        now = time.time()
        strategy_close_time = next_candle_close_epoch(now, strategy_period)

        status_print(f"🎯 Sincronizando con {timeframe} | Próximo cierre: {datetime.fromtimestamp(strategy_close_time).strftime('%H:%M:%S')}")

        # Heartbeat hasta el cierre de vela de la estrategia
        while now < strategy_close_time and self.is_running:
            # Obtener precio actual
            last_price = self._last_tick(symbol)

            # Mostrar heartbeat
            heartbeat_print(symbol, timeframe, now, strategy_close_time,
                          last_price if last_price is not None else "N/A")

            # Calcular tiempo de sleep (mínimo heartbeat_period, máximo tiempo restante)
            remaining_time = strategy_close_time - now
            sleep_duration = min(heartbeat_period, max(1, remaining_time))

            time.sleep(sleep_duration)
            now = time.time()

        status_print(f"✅ Sincronización completada - {timeframe} cerrado")

    def _check_existing_positions(self, symbol: str) -> str:
        '''
        Verificar posiciones existentes en DB local y broker.

        Limpia posiciones cerradas automáticamente y retorna
        ticket de posición activa si existe.

        Args:
            symbol: Símbolo a verificar

        Returns:
            str: Ticket de posición activa o None
        '''
        # Obtener posiciones desde DB local
        positions_list = local_db.get_all_positions(symbol=symbol)

        if not positions_list:
            status_print(f"ℹ️ No hay posiciones guardadas localmente para {symbol}")
            return None

        # Verificar primera posición encontrada
        position_record = positions_list[0]
        position_ticket = position_record.get('ticket')

        status_print(f"📋 Posición encontrada en DB: Ticket {position_ticket} | Comentario: {position_record.get('comment')}")

        # Verificar si la posición sigue activa en el broker
        if not broker_connection.is_position_open(position_ticket):
            status_print(f"🔄 Posición {position_ticket} cerrada por broker (SL/TP/Manual). Limpiando registro local.")
            local_db.delete_position(position_ticket)
            return None
        else:
            status_print(f"✅ Posición {position_ticket} confirmada activa en broker")
            return position_ticket

    def _monitor_active_position(self, position_ticket: str, symbol: str, timeframe: str, close_method: str, candles_required: int):
        '''
        Monitorear posición activa hasta su cierre.

        Obtiene contexto completo de la posición y ejecuta
        monitoreo específico según tipo de estrategia.

        Args:
            position_ticket: Ticket de la posición a monitorear
            symbol: Símbolo de trading
            timeframe: Timeframe de análisis
            close_method: Método de cierre (trailing, fixed, etc.)
            candles_required: Velas requeridas para análisis
        '''
        # Obtener contexto completo de la posición desde broker
        open_position_data = broker_connection.get_dates_position(position_ticket)

        if not open_position_data:
            status_print(f"⚠️ No se pudo obtener contexto de posición {position_ticket}. Eliminando registro local.")
            local_db.delete_position(position_ticket)
            return

        # Extraer información de la posición
        strategy_type = open_position_data.get('strategy_type', 'TREND')
        ticket = open_position_data.get('ticket')

        status_print(f"🎯 Iniciando monitoreo: Ticket {ticket} | Tipo: {strategy_type}")

        # Ejecutar monitoreo según tipo de estrategia
        if strategy_type == 'TREND':
            decision_print(symbol, timeframe, f"Monitoreando estrategia TREND para ticket {ticket}")
            monitor.monitor(open_position_data, symbol, timeframe, close_method, candles_required)

        elif strategy_type == 'RANGE':
            decision_print(symbol, timeframe, f"Monitoreando estrategia RANGE para ticket {ticket}")
            # Implementar monitoreo de rango cuando esté disponible
            status_print("⚠️ Monitoreo RANGE no implementado aún")

        else:
            status_print(f"⚠️ Tipo de estrategia desconocido: {strategy_type}")

    def _market_analysis_cycle(self, symbol: str, timeframe: str, config: Dict, account_balance: float,
                             open_method: str, close_method: str, candles_required: int):
        '''
        Ciclo de análisis de mercado para nuevas entradas.

        Sincroniza con cierres de vela y ejecuta estrategia de trading
        hasta que se genere una nueva posición.

        Args:
            symbol: Símbolo de trading
            timeframe: Timeframe de análisis
            config: Configuración completa
            account_balance: Balance actual de cuenta
            open_method: Método de apertura de posiciones
            close_method: Método de cierre de posiciones
            candles_required: Velas requeridas para análisis
        '''
        status_print(f"🔄 Iniciando ciclo de análisis en {timeframe}")

        analysis_cycles = 0
        max_cycles = 100  # Límite de seguridad para evitar loops infinitos

        while analysis_cycles < max_cycles and self.is_running:
            analysis_cycles += 1

            try:
                status_print(f"📊 Ciclo de análisis #{analysis_cycles}")

                # Sincronizar con cierre de vela y obtener datos
                data = cycles.candle_closing_sync(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles_required=candles_required
                )

                if data is None:
                    status_print("⚠️ Error en sincronización de datos. Reintentando en 30s...")
                    time.sleep(30)
                    continue

                status_print(f"✅ Datos sincronizados. Ejecutando estrategia...")
                decision_print(symbol, timeframe, "Analizando oportunidad de entrada")

                # Ejecutar lógica de estrategia
                strategy.strategy_logic(
                    symbol=symbol,
                    timeframe=timeframe,
                    config=config,
                    account_balance=account_balance,
                    open_method=open_method,
                    close_method=close_method,
                    candles_required=candles_required
                )

                # Verificar si se abrió nueva posición
                new_positions = local_db.get_all_positions(symbol=symbol)
                if new_positions:
                    status_print("🎉 Nueva posición detectada. Saliendo del ciclo de análisis.")
                    break

                # Si no hay nueva posición, continuar análisis
                status_print("➡️ No hay nueva entrada. Continuando análisis...")

            except Exception as e:
                status_print(f"❌ Error en ciclo de análisis: {e}")
                time.sleep(30)

        if analysis_cycles >= max_cycles:
            status_print(f"⚠️ Límite de ciclos alcanzado ({max_cycles}). Reiniciando sistema...")


# === FUNCIONES DE UTILIDAD GLOBAL ===

def create_live_instance():
    '''
    Crear instancia de Live para uso externo.
    Factory function para inicialización limpia.
    '''
    return Live()

def set_verbose_mode(verbose: bool):
    '''
    Configurar modo verbose globalmente.

    Args:
        verbose: True para narración completa, False para modo silencioso
    '''
    global VERBOSE
    VERBOSE = verbose
    status_print(f"🔧 Modo verbose: {'Activado' if verbose else 'Desactivado'}")
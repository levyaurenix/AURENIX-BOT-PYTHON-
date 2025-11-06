# ── imports de tipos ───────────────────────────────────────────
from typing import Dict, Any, Tuple, Optional, Literal
from pandas import DataFrame
# ───────────────────────────────────────────────────────────────
# indicadores técnicos
import talib as ta

class Emas:
   
    # Constructor (recibe parámetros de la estrategia y símbolo decimal)
    def __init__(self, params: Dict[str, Any], decimal_symbol: int):
        self.DECIMAL_SYMBOL: int = decimal_symbol

        # Parámetros de EMAs y ATR
        self.EMA_FAST: int = params['ema_f']
        self.EMA_MEDIUM: int = params['ema_m']
        self.EMA_SLOW: int = params['ema_s']
        self.ATR_PERIOD: int = params['atr_period']
        self.SL_OFFSET_ATR: float = params['sl_offset']  # SL amplio

       # Filtros de entrada
        self.RSI_PERIOD: int = params['rsi_period']
        self.RSI_UPPER: float = params['rsi_upper']
        self.RSI_LOWER: float = params['rsi_lower']
        self.ATR_MUL_CONS: float = params['atr_mul_cons']
        self.ATR_MAX_DIST_FAST: float = params['atr_max_dist_fast']

       # --- ALIAS de compatibilidad (por si en métodos viejos se usa LIMIT_*) ---
        self.RSI_LIMIT_UPPER = self.RSI_UPPER
        self.RSI_LIMIT_LOWER = self.RSI_LOWER
       # opcional: alias en minúsculas si algún trozo antiguo los pide
        self.rsi_upper = self.RSI_UPPER
        self.rsi_lower = self.RSI_LOWER



    def order_entry(self, df: DataFrame) -> dict:
        
        min_required_bars = max(self.EMA_SLOW, self.RSI_PERIOD) + self.ATR_PERIOD
        log_message = 'NONE'

        if df.empty or len(df) < min_required_bars:
            log_message = f"❌ Análisis: Datos insuficientes ({len(df)})."
            # Estructura de retorno estricta
            return {'command': 'WAIT', 'sl_price': 0.0, 'entry_price': 0.0}

        close_prices = df['close'].values
        
        # Cálculo de Indicadores
        ema_fast_array = ta.EMA(close_prices, timeperiod=self.EMA_FAST)
        ema_medium_array = ta.EMA(close_prices, timeperiod=self.EMA_MEDIUM)
        ema_slow_array = ta.EMA(close_prices, timeperiod=self.EMA_SLOW)
        atr_array = ta.ATR(df['high'], df['low'], df['close'], timeperiod=self.ATR_PERIOD)
        rsi_array = ta.RSI(close_prices, timeperiod=self.RSI_PERIOD)

        current_atr = atr_array.iloc[-1]
        ema_fast = ema_fast_array[-1]
        ema_medium = ema_medium_array[-1]
        ema_slow = ema_slow_array[-1]
        current_rsi = rsi_array[-1]
        entry_price = close_prices[-1]
        
        command = 'NONE'
        sl_price = 0.0

        # FILTRO 1: ALINEACIÓN ESTRICTA Y SEPARACIÓN (Tendencia Fuerte)
        is_aligned_buy = (ema_fast > ema_medium) and (ema_medium > ema_slow)
        is_aligned_sell = (ema_fast < ema_medium) and (ema_medium < ema_slow)
        dist_medium_slow = abs(ema_medium - ema_slow)
        is_separated = (dist_medium_slow / current_atr) > self.ATR_MUL_CONS
        
        
        if is_aligned_buy or is_aligned_sell:
            
            if is_separated: # ✅ FILTRO 1 OK
                
                # FILTRO 2: DISTANCIA MÁXIMA DEL PRECIO A EMA RÁPIDA (Pullback)
                dist_price_fast_ema = abs(entry_price - ema_fast)
                is_price_near_fast_ema = (dist_price_fast_ema / current_atr) < self.ATR_MAX_DIST_FAST

                # FILTRO 3: ESTADO DEL MOMENTUM (RSI)
                is_rsi_valid_buy = current_rsi < self.RSI_LIMIT_UPPER 
                is_rsi_valid_sell = current_rsi > self.RSI_LIMIT_LOWER 
                
                # FILTRO 4: CONFIRMACIÓN DE LA BARRA DE ENTRADA (Cierre más allá de EMA Media)
                is_entry_confirmed = (is_aligned_buy and entry_price > ema_medium) or \
                                     (is_aligned_sell and entry_price < ema_medium)
                
                
                # LÓGICA DE DECISIÓN FINAL (Entra solo si los 4 filtros pasan)
                
                if is_entry_confirmed and is_price_near_fast_ema:
                    
                    if is_aligned_buy and is_rsi_valid_buy:
                        
                        sl_ema_55 = ema_slow - (current_atr * self.SL_OFFSET_ATR)
                        command = 'BUY'
                        sl_price = sl_ema_55
                        log_message = f"Análisis: ✅ COMPRA. SL: {sl_price:.5f}. (RSI: {current_rsi:.2f})"

                    elif is_aligned_sell and is_rsi_valid_sell:
                        
                        sl_ema_55 = ema_slow + (current_atr * self.SL_OFFSET_ATR)
                        command = 'SELL'
                        sl_price = sl_ema_55
                        log_message = f"Análisis: ✅ VENTA. SL: {sl_price:.5f}. (RSI: {current_rsi:.2f})"
                    
                    else:
                        # Bloqueo por fallos de RSI (Filtro 3)
                        log_message = "❌ Bloqueo: Alineación OK, pero RSI no válido."
                
                else:
                    # Bloqueo por fallos en Filtro 2 o 4
                    log_msg = "❌ Bloqueo: Alineación OK, pero falló en: "
                    if not is_price_near_fast_ema: log_msg += "Precio no cerca de EMA Rápida. "
                    if not is_entry_confirmed: log_msg += "Barra de entrada no confirma. "
                    log_message = log_msg
                    
            else:
                # Bloqueo por fallo en el Filtro 1 (Separación): Consolidación
                log_message = "Análisis: 🔴 Alineación OK, pero EMAs muy juntas (Consolidación)."
                
        else:
            # Bloqueo por fallo en el Filtro 1 (Alineación)
            log_message = "Análisis: 🔴 Las 3 EMAs no están estrictamente ordenadas."
            
        # RETORNO FINAL (¡Cumpliendo estrictamente el formato!)
        return {
            'command': command,
            'sl_mode': False,
            'sl_price': round(sl_price, self.DECIMAL_SYMBOL),
            'tp_mode': True,
            'tp_price': 0, 
            'entry_price': round(entry_price, self.DECIMAL_SYMBOL),
            'log_message': log_message
        }


    def tp_close(self, order_type: Literal['BUY', 'SELL'], data: DataFrame, entry_price: float) -> Tuple[str | None, str]:
        
        close_prices = data['close'].values
        signal = None
        log_message ='NONE'

        ema_medium_array = ta.EMA(close_prices, timeperiod=self.EMA_MEDIUM)
        ema_slow_array = ta.EMA(close_prices, timeperiod=self.EMA_SLOW)
        
        if len(ema_medium_array) < 1 or len(ema_slow_array) < 1:
            return None, "Datos insuficientes para cierre."
        
        last_ema_medium = ema_medium_array[-1]
        last_ema_slow = ema_slow_array[-1]
        current_close = close_prices[-1]

        
        # CONDICIÓN 1: RUPTURA DE LA EMA MEDIA (EMA 21) - TP Dinámico Principal (Salida más lenta)
        is_medium_break_exit = False
        
        if order_type == 'BUY':
            if current_close <= last_ema_medium:
                log_message = f"🔴 (Cierre BUY - Ruptura Media) Precio ({current_close:.5f}) debajo de EMA_MEDIUM."
                is_medium_break_exit = True
                
        elif order_type == 'SELL':
            if current_close >= last_ema_medium:
                log_message =f"🟢 (Cierre SELL - Ruptura Media) Precio ({current_close:.5f}) encima de EMA_MEDIUM."
                is_medium_break_exit = True

        
        # CONDICIÓN 2: RUPTURA DE TENDENCIA (Cierre por debajo/encima de EMA Slow)
        is_trend_broken = False
        
        if order_type == 'BUY':
            if current_close <= last_ema_slow:
                log_message = f"🛑 (Cierre BUY - Ruptura Trend) Precio ({current_close:.5f}) debajo de EMA_SLOW."
                is_trend_broken = True
                
        elif order_type == 'SELL':
            if current_close >= last_ema_slow:
                log_message =f"🛑 (Cierre SELL - Ruptura Trend) Precio ({current_close:.5f}) encima de EMA_SLOW."
                is_trend_broken = True
                
        if is_medium_break_exit or is_trend_broken:
            signal = 'CLOSE'
            
        return signal, log_message
    

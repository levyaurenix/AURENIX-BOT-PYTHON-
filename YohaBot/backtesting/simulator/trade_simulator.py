import pandas as pd
from typing import Callable
from datetime import timedelta
from pandas import DataFrame

from config.config_global import TIME_FRAME_MINUTES



class TradeSimulator:
    
    def __init__(self, symbol: str, instrument_config: dict, start_cash: float, ):
        """
        Inicializa el simulador de trading. Mantiene el estado de la cuenta 
        y registra las operaciones cerradas.
        """
        self.symbol = symbol

        self.leverage = instrument_config['leverage']
        self.decimal_symbol = instrument_config['decimal_symbol']
        self.point_size = instrument_config['point_size']
        self.spread_pips = instrument_config['spread_pips']
        self.contract_size = instrument_config['contract_size']
        self.commission_rate = instrument_config['commission']
        self.slippage_rate = instrument_config['slippage_bps']
        
        self.pip_value = self.contract_size * self.point_size
        self.spread_value = self.spread_pips * self.point_size
        
        
        self.start_cash = start_cash # Capital inicial para métricas
        
        self.balance = start_cash
        self.equity = start_cash
        self.free_margin = start_cash
        self.margin = 0

        # Pocision actual
        self.position = None 

        # Lista de diccionarios para el historial de trades
        self.closed_trades = [] 



    def open_position(self, order_entry: dict, current_price: float, lot_size: float, current_time: pd.Timestamp) -> bool:
        """Abre una posición y registra la información del trade."""
        
        side = order_entry['command']
        sl_mode = order_entry['sl_mode']
        sl_price = round(order_entry['sl_price'], self.decimal_symbol)
        tp_mode = order_entry['tp_mode']
        tp_price = round(order_entry['tp_price'], self.decimal_symbol)

        lot_size = round(lot_size, 2)

        # CÁLCULO Y VERIFICACIÓN DE MARGEN
        margin_required = round(self.calculate_margin(lot_size, current_price), 2)
        
        if self.balance < margin_required:
            print(f"❌ ABORTAR APERTURA: Margen ({margin_required:.2f}) excede balance disponible ({self.balance:.2f}).")
            return False

        standard_lot_size = 1.0 
        commission = (lot_size / standard_lot_size) * self.commission_rate
        
        commission_with_margin = commission + margin_required

        if self.balance - commission_with_margin <= 0.0:
            print(f"❌ ABORTAR APERTURA: Capital insuficiente después de comisión ({self.balance:.2f}).")
            return False

        self.balance -= commission
        self.balance -= margin_required
        self.free_margin = self.equity - margin_required
        self.margin += margin_required

        entry_price_with_slippage = round(self.apply_slippage(current_price, side), self.decimal_symbol)
        self.balance = round(self.balance, 2)

        self.position = {
            'side': side, 
            'lot_size': lot_size, 
            'sl_mode': sl_mode,
            'sl_price': sl_price,
            'tp_mode': tp_mode,
            'tp_price': tp_price,
            'commission': round(commission, 2),
            'entry_price': entry_price_with_slippage,
            'entry_time': current_time,
            'margin': margin_required
        }

        
        """ # PRINT DE EJECUCIÓN DE APERTURA REAL
        print("\n==================================================")
        print(f"🟢 TRADE OPEN | Time: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f" > Tipo: {side} | Lote: {lot_size:.4f} | Margen: {margin_required:.2f}")
        print(f" > Precio Entrada: {entry_price_with_slippage} | SL: {sl_price}")
        print(f" > CAPITAL ACTUAL: {self.balance:.2f}")
        print("==================================================") """
        
        return True


    def close_position(self, exit_price: float, close_type: str, current_time: pd.Timestamp) -> None:
        """Cierra la posición activa, actualiza el capital y registra el trade."""

        side = self.position['side'] 
        entry = round(self.position['entry_price'], self.decimal_symbol)
        lot_size = round(self.position['lot_size'], 2)
        margin_used = round(self.position['margin'], 2)
        commission = round(self.position['commission'], 2)
        entry_time = self.position['entry_time']

        # 1. Aplicar slippage al precio de cierre.
        exit_price_with_slippage = round(self.apply_slippage(exit_price, 'SELL' if side == 'BUY' else 'BUY'), self.decimal_symbol)

        # 2. CALCULAR PIPS
        pips_signed = self.calculate_pips(entry, exit_price_with_slippage, side) 

        # 3. PnL Bruto
        gross_profit = pips_signed * self.pip_value * lot_size

        # 4. PnL Neto: Descontar la comisión y redondear
        net_profit = round(gross_profit - commission, 2)
        pips_moved_recorded = round(pips_signed, 2)

        # 5. ACTUALIZACIÓN DEL BALANCE Y MARGEN

        # Reembolso del Margen Usado 
        self.balance += margin_used 

        # Aplicar PnL Neto (Sumará si es ganancia, restará si es pérdida)
        self.balance += net_profit

        # Redondear el Balance final para mantener la precisión decimal
        self.balance = round(self.balance, 2)

        # Limpieza de variables de posición
        self.margin -= margin_used
        self.equity = self.balance       
        self.free_margin = self.balance
        # 6. REGISTRO DE DATOS COMPLETOS EN EL HISTORIAL
        self.closed_trades.append({
            'symbol': self.symbol,
            'side': side,
            'lot_size': lot_size,
            'entry_time': entry_time,
            'entry_price': entry,
            'exit_time': current_time,
            'exit_price': exit_price_with_slippage,
            'net_profit': net_profit,         
            'pips_moved': pips_moved_recorded, 
            'commission': commission,
            'margin_used': margin_used,
            'close_type': close_type,
        })

        """ # 7. PRINT Y FINALIZACIÓN
        print("\n==================================================")
        print(f"⭐ TRADE CLOSED ({close_type}) | Time: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f" > Tipo Cierre: {side} | Ganancia: {net_profit:.2f}")
        print(f" > Margen Devuelto: {margin_used:.2f}")
        print(f" > SALDO TOTAL ACTUAL: {self.balance:.2f}")
        print("==================================================") """

        self.position = None


    def get_equity(self, current_price: float) -> float:
        """Calcula el capital actual (balance + PnL no realizado)."""

        # 1. Calcular Equidad
        if not self.position:
            # Sin posición abierta: Equidad = Balance
            self.equity = self.balance
        else:
            side = self.position['side']
            entry = self.position['entry_price']
            lot_size = self.position['lot_size']
            
            # PnL no realizado
            if side == 'BUY':
                pnl_unrealized = (current_price - entry) * lot_size * self.contract_size
            else: # SELL
                pnl_unrealized = (entry - current_price) * lot_size * self.contract_size
                
            self.equity = round(self.balance + pnl_unrealized, 2)
            
        # 2. Calcular Margen Libre
        self.free_margin = round(self.equity - self.margin, 2)

        return self.equity


    def calculate_margin(self, lot_size: float, price: float) -> float:
        """Calcula el margen requerido para una posición (size * price / leverage)."""

        return (lot_size * price * self.contract_size) / self.leverage


    def apply_slippage(self, price: float, side: str) -> float:
        """Aplica Spread (Bid/Ask) y Slippage (porcentaje) al precio de ejecución."""
        
        # spread_amount ahora es self.spread_value
        spread_amount = self.spread_value 
        
        # 1. Ajuste del precio por Spread (Bid/Ask)
        if side == 'BUY':
            # Abrir BUY usa ASK: precio + spread_amount
            execution_price_spread = price + spread_amount
        else: # SELL
            # Abrir SELL usa BID: precio base
            execution_price_spread = price 
            
        # 2. Aplicación del Slippage al precio de ejecución
        # self.slippage_rate es la tasa porcentual
        slippage_deviation = execution_price_spread * self.slippage_rate
        
        # 3. Aplicación final (siempre desfavorable)
        if side == 'BUY': 
            # Slippage desfavorable: precio más alto
            final_price = execution_price_spread + slippage_deviation 
        elif side == 'SELL': 
            # Slippage desfavorable: precio más bajo
            final_price = execution_price_spread - slippage_deviation
        else:
            final_price = price

        return final_price

    
    def calculate_pips(self, entry: float, exit: float, side: str) -> float:
        """
        Calcula el PnL en pips con el signo correcto (+ si gana, - si pierde).
        El cálculo usa self.point_size (el tamaño de un pip en el precio).
        """
        if side == 'BUY':
            # Diferencia de precio / Tamaño del punto
            pips = (exit - entry) / self.point_size 
        else: # SELL
            # Diferencia de precio / Tamaño del punto
            pips = (entry - exit) / self.point_size

        return pips


    def check_sl(self, current_low: float, current_high: float, current_time: pd.Timestamp,  side: str, sl_price: float, sl_mode: bool) -> bool:
       
        """Verifica si el precio de SL fue alcanzado por la barra actual."""
        closed = False

        if not sl_mode: # false es modo fijo

            if side == 'BUY' and current_low <= sl_price:
                self.close_position(current_low, 'STOP_LOSS', current_time)
                closed = True
            elif side == 'SELL' and current_high >= sl_price:
                self.close_position(current_high, 'STOP_LOSS', current_time)
                closed = True
            
        return closed
    
    
    def check_tp(self, close_method: Callable, df_slice: DataFrame ,current_low: float, current_high: float, current_time: pd.Timestamp, side: str, tp_price: float, tp_mode: bool) -> bool:
       
        """Verifica si el precio de SL fue alcanzado por la barra actual."""
        
        closed = False

        if tp_mode:

            signal, log_message = close_method(side, df_slice, self.position['entry_price'])
            
            if   signal == 'CLOSE':
                self.close_position(current_high, 'TP/DYNAMIC', current_time)
                closed = True
            

        else:
            
            if side == 'BUY' and current_high >= tp_price:
                self.close_position(current_high, 'TP/FIXED', current_time)
                closed = True
            elif side == 'SELL' and current_low <= tp_price:
                self.close_position(current_low, 'TP/FIXED', current_time)
                closed = True
            

            
        return closed
    

    def get_trades_history(self) -> pd.DataFrame:
        """Retorna el historial completo de operaciones cerradas como un DataFrame."""
        return pd.DataFrame(self.closed_trades)
    
    def calculate_total_days_analyzed(self, i: int, timeframe: str, trading_hours_per_day: int = 23, trading_days_per_week: int = 5) -> str:
        """
        Calcula y formatea la duración total de 'i' velas, considerando 
        las horas y días de trading reales del activo.
        """

       
        
        minutes_per_bar = TIME_FRAME_MINUTES.get(timeframe.upper())
        
        if minutes_per_bar is None:
            return f"Error: Temporalidad '{timeframe}' no válida."
        
        # 1. Minutos de Trading Total
        total_minutes = i * minutes_per_bar
        
        # 2. Minutos por Período de Trading
        minutes_per_trading_day = trading_hours_per_day * 60
        minutes_per_trading_week = minutes_per_trading_day * trading_days_per_week
        
        # 3. Desglose en Meses, Semanas, Días, Horas, Minutos
        
        parts = []
        
        # Meses (usando 4 semanas exactas)
        # 4 semanas * minutos por semana
        minutes_per_trading_month = minutes_per_trading_week * 4 
        
        # CÁLCULOS
        
        # Meses de trading completos
        months = total_minutes // minutes_per_trading_month
        total_minutes %= minutes_per_trading_month # Minutos restantes
        if months > 0:
            parts.append(f"{months}mo")
            
        # Semanas de trading completas
        weeks = total_minutes // minutes_per_trading_week
        total_minutes %= minutes_per_trading_week # Minutos restantes
        if weeks > 0:
            parts.append(f"{weeks}w")
            
        # Días de trading completos
        days = total_minutes // minutes_per_trading_day
        total_minutes %= minutes_per_trading_day # Minutos restantes
        if days > 0:
            parts.append(f"{days}d")
            
        # Horas restantes
        hours = total_minutes // 60
        total_minutes %= 60 # Minutos restantes finales
        if hours > 0:
            parts.append(f"{hours}h")
            
        # Minutos restantes finales
        if total_minutes > 0:
            parts.append(f"{total_minutes}m")
            
        if not parts:
            return "0m"
        
        return " ".join(parts)

"""     def calculate_total_days_analyzed(self, i: int, timeframe: str) -> str:
        
        # 1. Mapeo de duración de la vela a MINUTOS
        TIME_FRAME_MINUTES = {
            # Minutos
            "M1": 1, 
            "M2": 2, 
            "M3": 3, 
            "M4": 4, 
            "M5": 5, 
            "M6": 6, 
            "M10": 10,
            "M12": 12, 
            "M15": 15, 
            "M20": 20, 
            "M30": 30,

            # Horas (H * 60)
            "H1": 60, 
            "H2": 120, 
            "H3": 180, 
            "H4": 240, 
            "H6": 360, 
            "H8": 480, 
            "H12": 720,

            # Días, Semanas, Meses (Tiempo de calendario)
            "D1": 1440,      # 1 día de calendario
            "W1": 10080,     # 7 días de calendario
            "MN1": 43200,    # 30 días promedio de calendario (43800 para 30.41 días)
        }
    
        minutes_per_bar = TIME_FRAME_MINUTES.get(timeframe.upper())
        
        if minutes_per_bar is None:
            return f"Error: Temporalidad '{timeframe}' no válida."
        
        # Cálculo de la duración total
        total_minutes = i * minutes_per_bar
        duration = timedelta(minutes=total_minutes)
        
        # Conversión y Desglose
        total_days = duration.days
        total_seconds = duration.seconds
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        # 2. Formateo y Desglose en Meses, Días, Horas, Minutos
        
        parts = []
        
        # Meses (usando una base de 30 días para aproximar)
        if total_days >= 30:
            months = total_days // 30
            total_days %= 30 # Días restantes
            parts.append(f"{months}mo")
            
        if total_days > 0:
            parts.append(f"{total_days}d")
            
        if hours > 0:
            parts.append(f"{hours}h")
            
        if minutes > 0:
            parts.append(f"{minutes}m")
            
        # Retorna un valor, incluso si es solo 0 minutos
        if not parts:
            return "0m"
        
        return " ".join(parts)
     """


        


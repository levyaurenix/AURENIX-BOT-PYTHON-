from typing import Dict, Any
from strategy.strategies.emas import Emas

# --- CONFIGURACIÓN ÚNICA DEL BOT PARA BACKTESTING ---
BOT_CONFIG_BACKTEST: Dict[str, Any] = {
    'symbol': 'GOLD',
    'timeframe': 'M6',
    'balance': 200,
    'lot_mode': True,
    'lot_dynamic': lambda balance: round(balance / 10000, 2),
    'lot_fixed': 0.05,
    'strategy': {
        'name': 'EMA_Trend_Slow',
        'candles_required': 200, 
        'params': {
            'ema_f': 8,              
            'ema_m': 21,             
            'ema_s': 144,             
            'atr_period': 14,        
            'sl_offset': 2,        
            'rsi_period': 14,        
            'rsi_upper': 60.0,       
            'rsi_lower': 40.0,      
            'atr_mul_cons': 1.5,    
            'atr_max_dist_fast': 0.8,
            'atr_max_expansion': 4.0, 
            'ts_offset': 0.5,
            'ts_activation_atr': 1.0       
    
        },
        'strategy_class': Emas,
        'open_method': 'order_entry',
        'close_method': 'tp_close',
    }
}



""" BOT_CONFIG_BACKTEST: Dict[str, Any] = {
    'symbol': 'GOLD#',
    'timeframe': 'M6',
    'balance': 200,
    'lot_mode': True,
    'lot_dynamic': lambda balance: round(balance / 10000, 2),
    'lot_fixed': 0.05,
    'strategy': {
        'name': 'EMA_Trend_Slow',
        'candles_required': 200, 
        'params': {
            'ema_f': 8,              
            'ema_m': 21,             
            'ema_s': 144,             
            'atr_period': 14,        
            'sl_offset': 0.2,        
            'rsi_period': 14,        
            'rsi_upper': 60.0,       
            'rsi_lower': 40.0,       
            'atr_mul_cons': 1.5,     
            'atr_max_dist_fast': 0.8
        },
        'strategy_class': Emas,
        'open_method': 'order_entry',
        'close_method': 'tp_close',
    }
} """
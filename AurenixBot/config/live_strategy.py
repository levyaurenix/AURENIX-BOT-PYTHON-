from typing import List, Dict, Any
from strategy.strategies.emas import Emas

BOTS_CONFIG_LIVE: List[Dict[str, Any]] = [
    # --- BOT 1 ---
   {

    'symbol': 'GOLD',
    'timeframe': 'M6',
    'balance': 100,
    'lot_mode': True,
    'lot_dynamic': lambda balance: round(balance / 10000, 2),
    'lot_fixed': 0.05,
    'strategy': {
        'name': 'EMA_Trend_Slow',
        'candles_required': 200, # Minimo de velas requerido para el analisis
        'params': {
            'ema_f': 8,              
            'ema_m': 21,             
            'ema_s': 144,             
            'atr_period': 14,        
            'sl_offset': 2.5,        # SL amplio
            'rsi_period': 14,        
            'rsi_upper': 60.0,       # Máx. RSI para BUY
            'rsi_lower': 40.0,       # Mín. RSI para SELL
            'atr_mul_cons': 1.5,     # Mín. Separación EMAs
            'atr_max_dist_fast': 0.8
        },
        'strategy_class': Emas,
        'open_method': 'order_entry',
        'close_method': 'tp_close',
}
}
]
  

    # # #  # --- BOT 2 ---"""
    # # #          'symbol': 'GOLD',
    # # #     'timeframe': 'M6',
    # # #     'lot_mode': True,
    # # #     'lot_dynamic': lambda balance: round(balance / 10000, 2),
    # # #     'lot_fixed': 0.05,
    # # #     'magic_number': 1001,  # Todas las ordenes ejecutas por este bot
    # # #     'strategy': {
    # # #         'candles_required': 200,
    # # #         'name': 'EMA_Trend_Slow',
    # # #         'params': {
    # # #         'ema_f': 8,              
    # # #         'ema_m': 21,             
    # # #         'ema_s': 144,             
    # # #         'atr_period': 14,        
    # # #         'sl_offset': 0.2,        
    # # #         'rsi_period': 14,        
    # # #         'rsi_upper': 60.0,       
    # # #         'rsi_lower': 40.0,       
    # # #         'atr_mul_cons': 1.5,     
    # # #         'atr_max_dist_fast': 0.8
    # # #         },
    # # #         'strategy_class': Emas,       
    # # #         'open_method': 'order_entry',
    # # #         'close_method': 'tp_close',
    # # #     }
    # # # }, #
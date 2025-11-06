from typing import Dict, Any

from connection.terminal_connection import teminal_connection
from connection.broker_connection import broker_connection


def connection_and_get_dates(dates_login: dict, symbol: str) -> Dict[str, Any]:

        if not teminal_connection.initialize_terminal():
            return None
        
        if not teminal_connection.check_trade_permissions():
            teminal_connection.shutdown_terminal()
            return None
        
        # 1. INTENTO DE LOGIN Y OBTENCIÓN DE INFO
        success, account_balance, account_info, symbol_info = broker_connection.login_to_broker(dates_login['login'], dates_login['password'], dates_login['server'], symbol)

        # 2. OBTENCIÓN DE INFORMACIÓN DEL SÍMBOLO
        if not success:
            teminal_connection.shutdown_terminal()
            return None
        

        # 3. CONSTRUCCIÓN DE LA CONFIGURACIÓN DEL INSTRUMENTO
        instrument_config = {
            'account_balance': account_balance,
            'leverage': account_info.leverage, 
            'decimal_symbol': symbol_info.digits, 
            'point_size': symbol_info.point, 
            'spread_pips': symbol_info.spread,
            'contract_size': symbol_info.trade_contract_size,
            'commission': account_info.commission_blocked, 
            'slippage_bps': 0.0001, 
        }

        print("✅ conexión  completada")


        return instrument_config
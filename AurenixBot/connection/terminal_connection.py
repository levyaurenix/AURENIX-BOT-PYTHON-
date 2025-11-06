import MetaTrader5 as mt5

class TerminalConnection:
  
    # INIZIALIZACION DE LA TERMINAL
    def initialize_terminal(self):
        """
        Inicializa el terminal MetaTrader 5.
        Retorna True si tiene éxito, False en caso contrario.
        """
        if not mt5.initialize():
            # Comprobar si ya está inicializado
            if mt5.terminal_info() is not None:
                print("✅ El terminal MT5 ya está inicializado.")
                return True
            
            print(f"❌ Fallo al inicializar el terminal MT5. Error: {mt5.last_error()}")
            return False
        
        terminal_path = mt5.terminal_info().path
        print(f"✅ Conexión al terminal MT5 establecida en: {terminal_path}")
        return True

    # CIERRE DE TERMINAL
    def shutdown_terminal(self):
        """
        Cierra la conexión con el terminal MetaTrader 5
        """

        mt5.shutdown()
        print("🔌 Conexión con el terminal MT5 cerrada.")


    # CHECKEO DE PERMISO DE TRADING AUTOMATICO
    def check_trade_permissions(self):
        """
        Comprueba si el trading automatizado está habilitado en la configuración del terminal MT5.
        """

        if mt5.terminal_info().trade_allowed:
            print("✅ El trading automatizado está PERMITIDO.")
            return True
        else:
            print("⚠️ El trading automatizado NO está permitido en la configuración del terminal.")
            return False
        
teminal_connection = TerminalConnection()

from typing import Callable

from config.config_global import CHECK_FREQUENCY_CYCLES

from live.config.schedules import cycles
from connection.broker_connection import broker_connection
from live.config.db_positions import local_db




class Monitor:

  def monitor(self, order_data: dict, symbol: str, timeframe: str, tp_close: Callable, candles_required: int ):
    """
    Monitorea la posición activa en un bucle continuo sincronizado con el 
    cierre de cada vela. Su objetivo es detectar la condición de cierre 
    por estrategia
    
    Args:
        order_data (dict): Diccionario con la información de la posición (incluye 'ticket', 'type', 'magic').
        symbol (str): Símbolo del activo.
        timeframe (str): Marco de tiempo.
    """
    position_object = order_data['position_object']
    
    # 2. Extraer el tipo de orden (BUY/SELL) del comentario
    # Ejemplo: 'BOT_TREND_SELL' -> ['BOT', 'TREND', 'SELL'] -> 'SELL'
    comment = position_object.comment.upper()
    
    # Dividimos la cadena por el guion bajo (_) y tomamos el último elemento
    order_type = comment.split('_')[-1] 
    
    # Comprobación de seguridad (solo para que sea 'BUY' o 'SELL')
    if order_type not in ['BUY', 'SELL']:
        print(f"❌ (MONITOR ERROR) Tipo de orden no válido extraído: {order_type}")
        return

    ticket = order_data.get('ticket')

    if not order_type:
        print("❌ (MONITOR ERROR) No se pudo obtener el tipo de orden. Finalizando monitor.")
        return
    
    # Inicializar el contador de ciclos para la verificación de 15 min
    cycle_count = 0 

    print(f"*** Monitor de Estrategia INICIADO para TICKET {ticket} ({symbol}/{timeframe}) ***")
    

    try:
        # El ciclo while corre indefinidamente hasta que se detecte la condición de cierre
        while True:
            # 1. ESPERAR Y OBTENER DATOS (Sincronizado al cierre de vela)
            data = cycles.candle_closing_sync(symbol, timeframe, candles_required)
            cycle_count += 1

            if data is None:
                continue
            
            # =========================================================
            # A. VERIFICACIÓN PERIÓDICA DE POSICIÓN ABIERTA (Cada N ciclos)
            # =========================================================
            if cycle_count >= CHECK_FREQUENCY_CYCLES:
                print(f"⏳ Verificando TICKET {ticket} (Ciclo {cycle_count})...")
                
                # Si la posición ya no existe (cerrada por SL o TP externo)
                if not broker_connection.is_position_open(ticket):
                    print(f"❌ TICKET {ticket} no está abierta. Cierre por SL/TP detectado. Monitor FINALIZADO.")
                    break # Salir del bucle
                    
                cycle_count = 0 # Reiniciar el contador
            
            # =========================================================
            # B. ANÁLISIS DE CRUCE (Take Profit / Cierre por Estrategia)
            # =========================================================

            # Llama a la función que verifica el cruce inverso EMA 8/21
            signal, log_message = tp_close(
                order_type=order_type,
                data=data
            )

            if log_message != 'NONE':
                print (log_message)
            
            # 3. Control de funciones basado en el resultado
            if signal == 'CLOSE':                
                # Cierre de posición
                broker_connection.close_single_position(ticket)

                # Eliminar pocision local
                local_db.delete_position(ticket)

                # Finaliza la ejecución del monitor
                print(f"*** Condición de Cierre detectada para TICKET {ticket}. Monitor FINALIZADO. ***")
                break
                
            else:
                # Si es None, simplemente se reinicia el ciclo y espera el próximo cierre de vela
                print(" --> Aun no llegamos a la meta. Esperando próximo cierre de vela.")
        
    except Exception as e:
        print(f"❌ Error crítico del monitor ({symbol}) y ticket({ticket}): {e}")
        return

monitor = Monitor()  
    
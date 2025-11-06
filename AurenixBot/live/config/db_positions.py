import json
from typing import List, Dict, Any, Optional

from config.config_global import POSITIONS



# --- ESTRUCTURA DE DATOS --- #

# Tipo de datos para un solo registro de posición (ticket + comentario)
PositionRecord = Dict[str, Any] 

# Tipo de datos para el JSON completo (símbolo: lista de registros)
DatabaseStructure = Dict[str, List[PositionRecord]] 


class LocalDatabase:
    """
    Clase para manejar las interacciones CRUD (Crear, Obtener, Eliminar)
    con el archivo JSON local, almacenando posiciones agrupadas por símbolo.
    """

    def __init__(self):
        """Inicializa la base de datos local."""
        self.file_path = POSITIONS
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Asegura que el archivo JSON exista y contenga un diccionario vacío si está vacío."""
        try:
            with open(self.file_path, 'r') as f:
                # Intenta cargar para ver si es JSON válido
                json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Si no existe o está vacío/mal formateado, lo crea con un diccionario vacío
            with open(self.file_path, 'w') as f:
                json.dump({}, f)

    def _read_data(self) -> DatabaseStructure:
        """Lee y retorna todos los datos (diccionario de listas) del archivo JSON."""
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except:
            return {}

    def _write_data(self, data: DatabaseStructure):
        """Escribe la estructura completa de datos al archivo JSON."""
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)
    
# ======================================================================
# OPERACIONES CRUD
# ======================================================================

# CREAR POSICIÓN 
    def create_position(self, symbol: str, position_data: PositionRecord) -> bool:
        """
        Agrega una nueva posición al grupo del símbolo especificado.
        Guarda: ticket y comment.
        
        Args:
            symbol (str): El símbolo bajo el cual agrupar la posición.
            position_data (Dict): Un diccionario que debe contener 'ticket' y 'comment'.
            
        Returns:
            bool: True si la posición fue agregada, False si ya existía o faltan campos.
        """
        required_keys = {'ticket', 'comment'}
        if not required_keys.issubset(position_data.keys()):
            print(f"❌ Error: El diccionario de posición debe tener las claves {required_keys}.")
            return False

        data = self._read_data()
        ticket_to_add = position_data['ticket']
        
        # 1. Normalizar el símbolo a mayúsculas para consistencia
        symbol = symbol.upper()

        # 2. Verificar si el ticket ya existe en la lista de ese símbolo (si existe el símbolo)
        if symbol in data and any(item.get('ticket') == ticket_to_add for item in data[symbol]):
            print(f"⚠️ Advertencia: Posición con ticket {ticket_to_add} ya existe para {symbol}.")
            return False

        # 3. Construir el nuevo registro (solo ticket y comment)
        new_record = {
            'ticket': ticket_to_add,
            'comment': position_data['comment']
        }
        
        # 4. Inicializar la lista si el símbolo no existe y agregar el registro
        if symbol not in data:
            data[symbol] = []
            
        data[symbol].append(new_record)
        
        self._write_data(data)
        return True

# OBTENER TODAS LAS POSICIONES (POR SÍMBOLO)
    def get_all_positions(self, symbol: Optional[str] = None) -> DatabaseStructure | List[PositionRecord]:
        """
        Retorna todas las posiciones. Si se especifica un símbolo, retorna solo las de ese símbolo.
        
        Args:
            symbol (str, optional): Símbolo específico a consultar. Si es None, retorna toda la DB.
            
        Returns:
            DatabaseStructure | List[PositionRecord]: Diccionario completo o lista de posiciones del símbolo.
        """
        data = self._read_data()
        
        if symbol:
            symbol = symbol.upper()
            return data.get(symbol, [])
            
        return data

# OBTENER POSICIÓN POR TICKET (Busca en todos los símbolos)
    def get_position_by_ticket(self, ticket: int) -> Optional[PositionRecord]:
        """
        Busca y retorna un registro de posición específico por su número de ticket, 
        recorriendo todos los símbolos.
        
        Retorna el registro completo (ticket, comment) con la clave 'symbol' agregada, o None.
        """
        data = self._read_data()
        
        for symbol, positions_list in data.items():
            for record in positions_list:
                if record.get('ticket') == ticket:
                    # Incluimos el símbolo para saber a qué grupo pertenece
                    record['symbol'] = symbol 
                    return record
        return None

# ELIMINAR POSICIÓN POR TICKET
    def delete_position(self, ticket: int) -> bool:
        """
        Elimina un registro de posición de la base de datos usando su ticket.
        Recorre todos los símbolos para encontrar y eliminar el ticket.
        
        Retorna True si la posición fue eliminada, False si no se encontró.
        """
        data = self._read_data()
        deleted = False
        
        # Iteramos sobre los símbolos y sus listas, necesitando modificar 'data' directamente
        for symbol in list(data.keys()): # Usamos list() para poder modificar el diccionario si eliminamos un símbolo
            
            initial_count = len(data[symbol])
            
            # Filtramos la lista, manteniendo solo los tickets que NO coinciden
            data[symbol] = [item for item in data[symbol] if item.get('ticket') != ticket]
            
            if len(data[symbol]) < initial_count:
                deleted = True
                
            # Si la lista queda vacía, eliminamos el símbolo del diccionario principal
            if not data[symbol]:
                del data[symbol]

        if deleted:
            self._write_data(data)
            return True
        else:
            print(f"⚠️ Advertencia: Ticket {ticket} no encontrado para eliminar.")
            return False
        
local_db = LocalDatabase()
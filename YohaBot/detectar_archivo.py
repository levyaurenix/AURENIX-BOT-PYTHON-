#!/usr/bin/env python3
"""
Detector de Archivos YohaBot - Desde Raíz
=========================================

Ejecuta desde: raíz YohaBot (donde está main.py)
Detecta archivos en: database/, backtesting/, live/, etc.

Usage: python detectar_archivo.py (desde raíz)

Autor: Sistema AURENIX - Raíz YohaBot
"""

import os
from pathlib import Path

def detectar_archivos_yohabot():
    """
    Detecta archivos desde raíz YohaBot (donde está main.py)
    """
    print("🔍 DETECTANDO ARCHIVOS YOHABOT - DESDE RAÍZ")
    print("=" * 50)
    print("(Ejecutándose desde raíz donde está main.py)")
    print()
    
    # Directorio actual: raíz YohaBot
    base_dir = Path(".")
    
    print(f"📁 Directorio YohaBot: {base_dir.resolve()}")
    print()
    
    # 1. Verificar que estamos en lugar correcto
    if not (base_dir / "main.py").exists():
        print("⚠️  ADVERTENCIA: No se encuentra main.py")
        print("   Asegúrate de ejecutar desde la raíz de YohaBot")
        print()
    
    # 2. Explorar estructura YohaBot
    print("📂 ESTRUCTURA YOHABOT:")
    carpetas_proyecto = []
    
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.') and item.name != '__pycache__':
            carpetas_proyecto.append(item.name)
            print(f"  📂 {item.name}/")
    
    # 3. Buscar archivos de datos en subcarpetas
    print(f"\n📊 ARCHIVOS DE DATOS:")
    extensiones = ['.csv', '.parquet', '.feather', '.xlsx', '.pkl', '.json']
    archivos_encontrados = []
    
    # Buscar en carpetas específicas
    carpetas_datos = ['database', 'data', 'backtesting', 'live', 'simulator', 'strategy']
    
    for carpeta in carpetas_datos:
        carpeta_path = base_dir / carpeta
        if carpeta_path.exists():
            archivos_carpeta = []
            
            for ext in extensiones:
                archivos = list(carpeta_path.glob(f"*{ext}"))
                archivos_carpeta.extend(archivos)
            
            if archivos_carpeta:
                print(f"\n  📂 {carpeta}/:")
                for archivo in archivos_carpeta:
                    try:
                        tamaño = archivo.stat().st_size / (1024*1024)  # MB
                        ruta_relativa = f"{carpeta}/{archivo.name}"
                        print(f"    📄 {archivo.name} ({tamaño:.1f}MB)")
                        archivos_encontrados.append(ruta_relativa)
                    except:
                        print(f"    📄 {archivo.name} (tamaño no disponible)")
                        archivos_encontrados.append(f"{carpeta}/{archivo.name}")
    
    # 4. Buscar también en raíz
    print(f"\n  📂 raíz/:")
    archivos_raiz = []
    for ext in extensiones:
        archivos = list(base_dir.glob(f"*{ext}"))
        archivos_raiz.extend(archivos)
    
    if archivos_raiz:
        for archivo in archivos_raiz:
            try:
                tamaño = archivo.stat().st_size / (1024*1024)  # MB
                print(f"    📄 {archivo.name} ({tamaño:.1f}MB)")
                archivos_encontrados.append(archivo.name)
            except:
                print(f"    📄 {archivo.name}")
                archivos_encontrados.append(archivo.name)
    
    # 5. Mostrar scripts Python principales
    print(f"\n🐍 SCRIPTS PYTHON PRINCIPALES:")
    py_files = [f for f in base_dir.glob("*.py") if not f.name.startswith('_')]
    for py_file in py_files[:10]:
        print(f"  🐍 {py_file.name}")
    
    # 6. Generar comandos para Monte Carlo
    print(f"\n" + "=" * 60)
    print("🎯 COMANDOS PARA MONTE CARLO (desde raíz YohaBot):")
    print("=" * 60)
    
    if archivos_encontrados:
        print(f"\n📊 ARCHIVOS DISPONIBLES ({len(archivos_encontrados)} encontrados):")
        
        # Mostrar primeros 5 archivos
        for i, archivo in enumerate(archivos_encontrados[:5], 1):
            print(f"  {i}. {archivo}")
        
        if len(archivos_encontrados) > 5:
            print(f"  ... y {len(archivos_encontrados) - 5} archivos más")
        
        # Comando recomendado
        archivo_recomendado = archivos_encontrados[0]
        print(f"\n💻 COMANDO RECOMENDADO:")
        print(f'python monte_carlo_raiz.py "{archivo_recomendado}"')
        
        # Comandos alternativos
        if len(archivos_encontrados) > 1:
            print(f"\n💻 OTROS COMANDOS DISPONIBLES:")
            for archivo in archivos_encontrados[1:4]:  # Mostrar 3 más
                print(f'python monte_carlo_raiz.py "{archivo}"')
                
    else:
        print("❌ No se encontraron archivos de datos")
        print("💡 Verifica que tengas datos en:")
        print("   - database/")
        print("   - backtesting/")
        print("   - data/")
        print("   - live/")
    
    # 7. Información de integración
    print(f"\n" + "=" * 60)
    print("🔗 INFORMACIÓN DE INTEGRACIÓN:")
    print("=" * 60)
    print("📁 Ejecutar desde: raíz YohaBot (donde está main.py)")
    print("📊 Datos detectados en:")
    
    # Mostrar carpetas que tienen datos
    carpetas_con_datos = set()
    for archivo in archivos_encontrados:
        if '/' in archivo:
            carpeta = archivo.split('/')[0]
            carpetas_con_datos.add(carpeta)
    
    for carpeta in sorted(carpetas_con_datos):
        print(f"   - {carpeta}/")
    
    print("🐍 Scripts Python: mismo directorio raíz")
    print("🔄 Rutas: carpeta/archivo.csv")
    
    print(f"\n✅ DETECCIÓN COMPLETADA DESDE RAÍZ YOHABOT")
    print("💡 Usa las rutas mostradas arriba con monte_carlo_raiz.py")

if __name__ == "__main__":
    detectar_archivos_yohabot()
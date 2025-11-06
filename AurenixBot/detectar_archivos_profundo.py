#!/usr/bin/env python3
"""
Detector YohaBot - Explorador de Subdirectorios
===============================================

Explora subdirectorios como data_history/, positions_live/, stadistics_results/
para encontrar archivos de datos reales.

Usage: python detectar_archivos_profundo.py

Autor: Sistema AURENIX - Exploración profunda
"""

import os
from pathlib import Path

def explorar_subdirectorios():
    """
    Explora profundamente todos los subdirectorios de YohaBot
    """
    print("🔍 DETECTOR YOHABOT - EXPLORACIÓN PROFUNDA")
    print("=" * 55)
    print("Buscando archivos en subdirectorios...")
    print()
    
    base_dir = Path(".")
    archivos_encontrados = []
    
    # Subdirectorios específicos detectados
    subdirs_explorar = [
        "database/data_history",
        "database/positions_live", 
        "database/stadistics_results",
        "backtesting",
        "live",
        "config",
        "strategy"
    ]
    
    extensiones_datos = ['.csv', '.parquet', '.xlsx', '.xls', '.feather', '.pkl', '.json', '.txt', '.dat']
    
    print("📂 EXPLORANDO SUBDIRECTORIOS:")
    
    for subdir in subdirs_explorar:
        subdir_path = base_dir / subdir
        
        if subdir_path.exists():
            print(f"\n  📂 {subdir}/:")
            
            # Explorar archivos en este subdirectorio
            archivos_subdir = []
            
            try:
                for archivo in subdir_path.iterdir():
                    if archivo.is_file():
                        ext = archivo.suffix.lower()
                        if ext in extensiones_datos or ext == '':  # Incluir archivos sin extensión
                            try:
                                tamaño = archivo.stat().st_size
                                if tamaño > 1024:  # Solo archivos > 1KB
                                    tamaño_mb = tamaño / (1024*1024)
                                    ruta_completa = f"{subdir}/{archivo.name}"
                                    archivos_subdir.append((archivo.name, tamaño_mb, ruta_completa))
                            except:
                                # Si no puede obtener tamaño, incluir igual
                                ruta_completa = f"{subdir}/{archivo.name}"
                                archivos_subdir.append((archivo.name, 0, ruta_completa))
                
                # Mostrar archivos encontrados
                if archivos_subdir:
                    for nombre, tamaño, ruta in archivos_subdir:
                        if tamaño > 0:
                            print(f"    📄 {nombre} ({tamaño:.1f}MB)")
                        else:
                            print(f"    📄 {nombre}")
                        archivos_encontrados.append(ruta)
                else:
                    print(f"    (vacío)")
                    
            except PermissionError:
                print(f"    ❌ Sin permisos")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        else:
            print(f"\n  📂 {subdir}/: (no existe)")
    
    # Buscar también archivos sueltos en directorios principales
    print(f"\n📂 ARCHIVOS SUELTOS EN DIRECTORIOS PRINCIPALES:")
    
    directorios_principales = ["database", "backtesting", "live", "config", "strategy"]
    
    for dir_principal in directorios_principales:
        dir_path = base_dir / dir_principal
        if dir_path.exists():
            archivos_sueltos = []
            
            try:
                for archivo in dir_path.iterdir():
                    if archivo.is_file():
                        ext = archivo.suffix.lower()
                        if ext in extensiones_datos:
                            try:
                                tamaño = archivo.stat().st_size / (1024*1024)
                                ruta_completa = f"{dir_principal}/{archivo.name}"
                                archivos_sueltos.append((archivo.name, tamaño, ruta_completa))
                            except:
                                ruta_completa = f"{dir_principal}/{archivo.name}"
                                archivos_sueltos.append((archivo.name, 0, ruta_completa))
                
                if archivos_sueltos:
                    print(f"\n  📂 {dir_principal}/:")
                    for nombre, tamaño, ruta in archivos_sueltos:
                        if tamaño > 0:
                            print(f"    📄 {nombre} ({tamaño:.1f}MB)")
                        else:
                            print(f"    📄 {nombre}")
                        archivos_encontrados.append(ruta)
                        
            except Exception as e:
                print(f"  ❌ Error explorando {dir_principal}: {e}")
    
    # Resultados
    print(f"\n" + "=" * 60)
    print("📊 RESUMEN DE ARCHIVOS ENCONTRADOS")
    print("=" * 60)
    
    if archivos_encontrados:
        print(f"\n✅ {len(archivos_encontrados)} archivos de datos encontrados:")
        
        for i, archivo in enumerate(archivos_encontrados, 1):
            print(f"  {i}. {archivo}")
        
        # Comando recomendado
        print(f"\n💻 COMANDOS PARA MONTE CARLO:")
        print("=" * 40)
        
        # Mostrar varios comandos
        for archivo in archivos_encontrados[:5]:
            print(f'python monte_carlo_yohabot.py "{archivo}"')
        
        if len(archivos_encontrados) > 5:
            print(f"... y {len(archivos_encontrados) - 5} archivos más")
        
        # Recomendación específica
        archivo_recomendado = archivos_encontrados[0]
        print(f"\n🎯 COMANDO RECOMENDADO (usar primero):")
        print(f'python monte_carlo_yohabot.py "{archivo_recomendado}"')
        
    else:
        print("❌ No se encontraron archivos de datos")
        print("\n💡 ACCIONES SUGERIDAS:")
        print("1. Verifica que tienes datos históricos")
        print("2. Busca archivos .csv o .xlsx en tu sistema")
        print("3. Copia archivos de datos a database/data_history/")
        print("4. Verifica permisos de archivos")
    
    # Exploración manual sugerida
    print(f"\n" + "=" * 60)
    print("🔍 EXPLORACIÓN MANUAL SUGERIDA")
    print("=" * 60)
    print("Para ver contenido específico, ejecuta:")
    print("ls database/data_history/")
    print("ls database/positions_live/")
    print("ls database/stadistics_results/")
    print("ls backtesting/")
    print("ls live/")
    
    print(f"\n✅ EXPLORACIÓN COMPLETADA")

if __name__ == "__main__":
    explorar_subdirectorios()
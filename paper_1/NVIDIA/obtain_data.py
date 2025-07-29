# Ejecutar cada uno de los archivos .py almacenados en un directorio dado por argumento
# Para cada uno del nombre de los archivos (excluido el .py) obtener metricas de rendimiento y guardarlas en
# el directorio Data en formato .csv (si los archivos ya existen entonces borrarlos)
# finalmente ejecutar un script para graficar los datos almacenados en Data y almacenarlos en Results

import os
import threading
import time
import argparse
import subprocess
import csv
from pynvml import *

class GpuMonitor():
    def __init__(self, interval=0.1, min_w_usage=0):
        """
        Detecta las GPU's disponibles, asigna una lista de objetos gpu1, gpu2, gpuX

        Monitorea el uso de los recursos en intervalos regulares (por defecto 0.1 segundos)

        Args:
            interval (float): Intervalo entre mediciones
        """

        nvmlInit()

        # VARS
        self.interval = interval
        self.min_w_usage = min_w_usage
        self.gpus = [nvmlDeviceGetHandleByIndex(i) for i in range(nvmlDeviceGetCount())]
        self.vram_usage = [[] for _ in range(len(self.gpus))]
        self.power = [[] for _ in range(len(self.gpus))]
        self.util = [[] for _ in range(len(self.gpus))]
        self.running = False
        self.thread = None

    def start(self):
        """Inicia el monitoreo en un hilo separado"""
        if (self.running):
            print("El monitoreo ya ha empezado")
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    
    def stop(self):
        """Detiene el Hilo de monitoreo"""
        self.running = False
        if self.thread:
            self.thread.join()
        nvmlShutdown()
    
    def _monitor(self):
        # Comenzamos a registrar las metricas una vez el consumo de energia sea mayor que x mW
        start = False
        while self.running:
            if start:
                # Registramos las metricas para cada gpu
                for idx, gpu in enumerate(self.gpus):
                    self.vram_usage[idx].append(nvmlDeviceGetMemoryInfo(gpu).used)
                    self.power[idx].append(nvmlDeviceGetPowerUsage(gpu)/1000)
                    self.util[idx].append(nvmlDeviceGetUtilizationRates(gpu).gpu)
            else:
                start = any(nvmlDeviceGetPowerUsage(gpu) > self.min_w_usage for gpu in self.gpus)
            time.sleep(self.interval)

    def get_stats(self):
        """
        Obtiene estadísticas agregadas (promedio, máximo) de las métricas recopiladas en _monitor.
        """
        stats = {}
        for idx in range(len(self.gpus)):
            stats[f"gpu_{idx}_vram_usage_avg"] = sum(self.vram_usage[idx]) / len(self.vram_usage[idx]) if self.vram_usage[idx] else None
            stats[f"gpu_{idx}_power_avg"] = sum(self.power[idx]) / len(self.power[idx]) if self.power[idx] else None
        return stats

    def export_to_csv(self, filename_prefix="gpu_stats", output_path="Data"):
        os.makedirs(output_path, exist_ok=True)
        for idx in range(len(self.gpus)):
            filename = f"{output_path}/{filename_prefix}_gpu{idx}.csv"
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (s)", "VRAM Usage (Bytes)", "Power (W)", "Utilization (%)"])
                for i in range(len(self.vram_usage[idx])):
                               writer.writerow([
                                   i * self.interval,
                                   self.vram_usage[idx][i],
                                   self.power[idx][i],
                                   self.util[idx][i]
                                ])

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor de GPU utilizando NVML")
    parser.add_argument('--interval', type=float, default=0.1, help='Intervalo entre mediciones en segundos')
    parser.add_argument('--min_w_usage', type=int, default=0, help='Umbral mínimo de uso de energía en mW')
    parser.add_argument('--exec_dir', type=str, default='.', help='Directorio a utilizar para la medición')
    parser.add_argument('--output_dir', type=str, default='.', help='Directorio a utilizar para almacenar el output')
    return parser.parse_args()

def execute_and_monitor(exec_path, output_path="Data", interval=0.1, min_w_usage=0):
    exec_name = os.path.basename(exec_path)
    print(f"\n Ejecutando {exec_name}...")
    
    monitor = GpuMonitor(interval=interval, min_w_usage=min_w_usage)
    monitor.start()

    # Ejecutar el programa
    try:
        result = subprocess.run(["python", exec_path])
    except Exception as e:
        print(f"Error al ejecutar {exec_path}")
        monitor.stop()
        return

    if result.returncode != 0:
        print(f"La ejecucion de {exec_path} ha fallado")
        monitor.stop()
        return

    print(f"{exec_path} ejecutado correctamente!")

    monitor.stop()
    monitor.export_to_csv(exec_name, output_path)
    print(f"Datos guardados en: {output_path}/{exec_name}_gpux.csv")

if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.exec_dir):
        print(f"El directorio {args.exec_dir} no existe.")
        exit(1)

    ejecutables = [os.path.join(args.exec_dir, f)
                   for f in os.listdir(args.exec_dir)
                   if f.endswith('.py') and os.path.isfile(os.path.join(args.exec_dir, f))]

    if not ejecutables:
        print(f"No se encontraron archivos ejecutables en {args.exec_dir}")
        exit(0)

    for exe in ejecutables:
        execute_and_monitor(exec_path=exe, output_path=args.output_dir, interval=args.interval, min_w_usage=args.min_w_usage)


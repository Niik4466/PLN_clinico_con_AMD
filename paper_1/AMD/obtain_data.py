# Ejecutar cada uno de los archivos .py almacenados en un directorio dado por argumento
# Para cada uno del nombre de los archivos (excluido el .py) obtener metricas de rendimiento y guardarlas en
# el directorio Data en formato .csv (si los archivos ya existen entonces borrarlos)
# finalmente ejecutar un script para graficar los datos almacenados en Data y almacenarlos en Results

import os
import threading
import time
import csv
import argparse
import subprocess

class GpuMonitor():
    def __init__(self, interval=0.1, min_power_mw=0, max_cards=8):
        """
        Monitor de GPUs AMD leyendo métricas de /sys/class/drm/cardX/device/

        Args:
            interval (float): intervalo de lectura en segundos.
            min_power_mw (int): umbral mínimo de consumo (mW) para comenzar a registrar.
            max_cards (int): máximo número de GPUs (cards) a monitorear.
        """
        self.interval = interval
        self.min_power_mw = min_power_mw
        # Detectar cards físicamente existentes
        self.cards = [f"card{i}" for i in range(max_cards) if os.path.exists(f"/sys/class/drm/card{i}/device")]
        self.vram_usage = [[] for _ in self.cards]
        self.power = [[] for _ in self.cards]
        self.util = [[] for _ in self.cards]
        self.running = False
        self.thread = None

    def _read_int(self, path):
        try:
            with open(path, "r") as f:
                return int(f.read().strip()), None
        except FileNotFoundError:
            return None, f"No existe {path}"
        except PermissionError:
            return None, f"Permiso denegado {path}"
        except Exception as e:
            return None, f"Error leyendo {path}: {e}"

    def _read_metrics_for_card(self, card):
        base_path = f"/sys/class/drm/{card}/device"
        gpu_busy_path = f"{base_path}/gpu_busy_percent"
        vram_used_path = f"{base_path}/mem_info_vram_used"
        power_path = f"{base_path}/hwmon/hwmon4/power1_average"

        gpu_busy, err_busy = self._read_int(gpu_busy_path)
        vram_used, err_vram = self._read_int(vram_used_path)
        power, err_power = self._read_int(power_path)

        return gpu_busy, err_busy, vram_used, err_vram, power, err_power

    def start(self):
        if self.running:
            print("Monitoreo ya iniciado")
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor(self):
        start_recording = False
        while self.running:
            if start_recording:
                for idx, card in enumerate(self.cards):
                    gpu_busy, _, vram_used, _, power, _ = self._read_metrics_for_card(card)
                    self.util[idx].append(gpu_busy)
                    self.vram_usage[idx].append(vram_used / 1024 ** 2)  # Convertir a MB
                    self.power[idx].append(power / 1_000_000)  # Convertir a W
            else:
                # Activar grabación si alguna GPU pasa el umbral de potencia
                for card in self.cards:
                    power, _ = self._read_int(f"/sys/class/drm/{card}/device/power1_average")
                    if power and power > self.min_power_mw:
                        start_recording = True
                        break
            time.sleep(self.interval)

    def get_stats(self):
        stats = {}
        for idx in range(len(self.cards)):
            stats[f"gpu_{idx}_vram_usage_avg"] = (sum(self.vram_usage[idx]) / len(self.vram_usage[idx])
                                                  if self.vram_usage[idx] else None)
            stats[f"gpu_{idx}_power_avg"] = (sum(self.power[idx]) / len(self.power[idx])
                                             if self.power[idx] else None)
            stats[f"gpu_{idx}_util_avg"] = (sum(self.util[idx]) / len(self.util[idx])
                                            if self.util[idx] else None)
        return stats

    def export_to_csv(self, filename_prefix="gpu_stats", output_path="Data"):
        os.makedirs(output_path, exist_ok=True)
        for idx, card in enumerate(self.cards):
            filename = f"{output_path}/{filename_prefix}_{card}.csv"
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time (s)", "VRAM Usage (Bytes)", "Power (mW)", "Utilization (%)"])
                for i in range(len(self.vram_usage[idx])):
                    writer.writerow([
                        i * self.interval,
                        self.vram_usage[idx][i] if self.vram_usage[idx][i] is not None else "",
                        self.power[idx][i] if self.power[idx][i] is not None else "",
                        self.util[idx][i] if self.util[idx][i] is not None else ""
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

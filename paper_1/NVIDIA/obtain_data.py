"""
Monitor de GPU + ejecutor de scripts .py
Guarda métricas por GPU en Data/ y resultados de gráficas en Results/ (la parte de graficar puede ser otro script).
"""

import os
import threading
import time
import argparse
import torch
import subprocess
import logging
import csv
import sys
import signal
from statistics import mean
from typing import List, Optional

# pynvml puede no estar disponible en sistemas sin drivers NVIDIA
try:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetCount, \
        nvmlDeviceGetMemoryInfo, nvmlDeviceGetPowerUsage, nvmlDeviceGetUtilizationRates, NVMLError
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

import torch
from fvcore.nn import FlopCountAnalysis


class GpuMonitor:
    def __init__(self, interval: float = 0.1, min_w_usage: int = 0):
        """
        interval: segundos entre mediciones
        min_w_usage: umbral en mW para comenzar a registrar
        """
        self.interval = interval
        self.min_w_usage = min_w_usage
        self.running = False
        self.thread = None

        if not NVML_AVAILABLE:
            print("Aviso: pynvml no está disponible. No se monitorizará GPU.")
            self.gpus = []
            return

        try:
            nvmlInit()
            count = nvmlDeviceGetCount()
            self.gpus = [nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception as e:
            print("Error inicializando NVML:", e)
            self.gpus = []

        # almacenar series por GPU
        self.vram_usage: List[List[float]] = [[] for _ in range(len(self.gpus))]
        self.power: List[List[float]] = [[] for _ in range(len(self.gpus))]
        self.util: List[List[float]] = [[] for _ in range(len(self.gpus))]

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        if NVML_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass

    def _monitor(self):
        start = False
        while self.running:
            if not self.gpus:
                time.sleep(self.interval)
                continue
            try:
                if not start:
                    # nvmlDeviceGetPowerUsage devuelve mW
                    start = any(nvmlDeviceGetPowerUsage(g) > self.min_w_usage for g in self.gpus)
                else:
                    for idx, g in enumerate(self.gpus):
                        mem = nvmlDeviceGetMemoryInfo(g).used
                        pwr = nvmlDeviceGetPowerUsage(g) / 1000.0  # a W
                        util = nvmlDeviceGetUtilizationRates(g).gpu  # %
                        self.vram_usage[idx].append(float(mem))
                        self.power[idx].append(float(pwr))
                        self.util[idx].append(float(util))
            except Exception as e:
                # no abortar el hilo por un error momentáneo
                print("Warning: error leyendo NVML:", e)
            time.sleep(self.interval)

    def clear(self):
        self.vram_usage = [[] for _ in range(len(self.gpus))]
        self.power = [[] for _ in range(len(self.gpus))]
        self.util = [[] for _ in range(len(self.gpus))]

    def get_stats(self):
        stats = {}
        for i in range(len(self.gpus)):
            # Filter out 0 values
            vram_clean = [x for x in self.vram_usage[i] if x > 0]
            power_clean = [x for x in self.power[i] if x > 0]
            util_clean = [x for x in self.util[i] if x > 0]
            stats[f"gpu_{i}_vram_avg_bytes"] = mean(vram_clean) if vram_clean else None
            stats[f"gpu_{i}_power_avg_w"] = mean(power_clean) if power_clean else None
            stats[f"gpu_{i}_util_avg_pct"] = mean(util_clean) if util_clean else None
            stats[f"gpu_{i}_samples"] = len(self.vram_usage[i])
        return stats

    def export_to_csv(self, filename_prefix="gpu_stats", output_path="Data", units="MiB"):
        """
        units: "MiB" or "bytes"
        """
        os.makedirs(output_path, exist_ok=True)
        for idx in range(len(self.gpus)):
            filename = os.path.join(output_path, f"{filename_prefix}_gpu{idx}.csv")
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                unit_label = "VRAM (MiB)" if units == "MiB" else "VRAM (Bytes)"
                writer.writerow(["time_s", unit_label, "Power_W", "Util_pct"])
                for i in range(len(self.vram_usage[idx])):
                    vram_val = self.vram_usage[idx][i]
                    if units == "MiB":
                        vram_val = vram_val / (1024.0 * 1024.0)
                    writer.writerow([
                        round(i * self.interval, 6),
                        vram_val,
                        self.power[idx][i] if i < len(self.power[idx]) else None,
                        self.util[idx][i] if i < len(self.util[idx]) else None
                    ])

    def export_epoch_resume(self, filename_prefix="epoch_resume", output_path="Data", epoch: int = 0):
        """
        Guarda/concatena resumen por época. Columnas: Epoch, Time_s, mean VRAM (MiB) GPU0.., mean Power GPU0.., mean Util GPU0..
        """
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, f"{filename_prefix}.csv")
        file_exists = os.path.isfile(filename)

        # construir row con medidas (usar MiB para legibilidad)
        time_s = (len(self.vram_usage[0]) * self.interval) if self.vram_usage and self.vram_usage[0] else 0.0
        mean_vram = [(mean(l) / (1024.0 * 1024.0)) if l else None for l in self.vram_usage]
        mean_power = [mean(l) if l else None for l in self.power]
        mean_util = [mean(l) if l else None for l in self.util]

        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["Epoch", "Time_s"] + \
                    [f"Mean_VRAM_MiB_GPU{i}" for i in range(len(self.gpus))] + \
                    [f"Mean_Power_W_GPU{i}" for i in range(len(self.gpus))] + \
                    [f"Mean_Util_pct_GPU{i}" for i in range(len(self.gpus))]
                writer.writerow(header)

            row = [epoch, round(time_s, 6)] + \
                [round(v, 6) if v is not None else "" for v in mean_vram] + \
                [round(p, 6) if p is not None else "" for p in mean_power] + \
                [round(u, 6) if u is not None else "" for u in mean_util]

            writer.writerow(row)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=float, default=0.1)
    p.add_argument('--min_w_usage', type=int, default=0)
    p.add_argument('--exec_dir', type=str, default='.')
    p.add_argument('--output_dir', type=str, default='Data')
    p.add_argument('--timeout', type=float, default=None, help="Timeout (s) para cada script ejecutado")
    return p.parse_args()


def execute_and_monitor(exec_path, output_path="Data", interval=0.1, min_w_usage=0, timeout=None):
    exec_name = os.path.splitext(os.path.basename(exec_path))[0]
    print(f"\n== Ejecutando {exec_name} ({exec_path}) ==")

    # borrar CSVs previos del mismo prefijo (si quieres ese comportamiento)
    for fname in os.listdir(output_path) if os.path.isdir(output_path) else []:
        if fname.startswith(exec_name):
            try:
                os.remove(os.path.join(output_path, fname))
            except Exception:
                pass

    monitor = GpuMonitor(interval=interval, min_w_usage=min_w_usage)
    monitor.start()

    try:
        # usa el mismo intérprete de Python que corre este script
        proc = subprocess.run([sys.executable, exec_path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        print(f"Timeout al ejecutar {exec_path}")
        monitor.stop()
        return
    except Exception as e:
        print(f"Error lanzando {exec_path}: {e}")
        monitor.stop()
        return

    if proc.returncode != 0:
        print(f"Ejecución fallida ({proc.returncode}). Stderr:\n{proc.stderr}")
        monitor.stop()
        return

    print(f"{exec_path} ejecutado correctamente.")
    monitor.stop()

    # exportar CSV por GPU con prefijo del ejecutable
    monitor.export_to_csv(filename_prefix=exec_name, output_path=output_path, units="MiB")
    print(f"Datos guardados en {output_path}/ con prefijo {exec_name}_")


def contar_flops(model: torch.nn.Module, input_shape=(1, 3, 224, 224)) -> int:
    model.eval()
    device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device("cpu")
    inputs = torch.randn(input_shape).to(device)
    with torch.no_grad():
        flops = FlopCountAnalysis(model, inputs)
        # flops.total() devuelve un número (int/float)
    return int(flops.total())


def export_model_info(model: torch.nn.Module, output_dir: str = "Data", input_shape=(1, 3, 224, 224)):
    """
    Calcula los FLOPs de un modelo PyTorch y guarda la información en un archivo CSV.

    Args:
        model (torch.nn.Module): Modelo de PyTorch (ej. models.resnet50(pretrained=True)).
        output_dir (str): Directorio donde guardar el archivo CSV.
        input_shape (tuple): Forma de entrada para el cálculo de FLOPs.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(__name__)

    try:
        # Se asume que la función contar_flops(model, input_shape) ya está definida
        flops = contar_flops(model, input_shape=input_shape)
    except Exception as e:
        logger.warning(f"No se pudo calcular FLOPs: {e}")
        flops = None

    output_path = os.path.join(output_dir, "model_info.csv")

    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model_name', 'input_shape', 'flops', 'dtype'])
            writer.writerow([
                model.__class__.__name__,
                str(input_shape),
                flops if flops is not None else 'error',
                str(next(model.parameters()).dtype)
            ])
        logger.info(f"Información del modelo guardada en {output_path}")
    except Exception as e:
        logger.error(f"Error al guardar la información del modelo: {e}")



if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.exec_dir):
        print(f"El directorio {args.exec_dir} no existe.")
        sys.exit(1)

    ejecutables = [os.path.join(args.exec_dir, f)
                   for f in os.listdir(args.exec_dir)
                   if f.endswith('.py') and os.path.isfile(os.path.join(args.exec_dir, f))]

    if not ejecutables:
        print(f"No se encontraron archivos ejecutables en {args.exec_dir}")
        sys.exit(0)

    # CTRL-C friendly: solo para parar el script principal
    def _sigint_handler(signum, frame):
        print("Interrupción recibida. Saliendo.")
        sys.exit(1)
    signal.signal(signal.SIGINT, _sigint_handler)

    os.makedirs(args.output_dir, exist_ok=True)
    for exe in ejecutables:
        execute_and_monitor(exec_path=exe,
                            output_path=args.output_dir,
                            interval=args.interval,
                            min_w_usage=args.min_w_usage,
                            timeout=args.timeout)
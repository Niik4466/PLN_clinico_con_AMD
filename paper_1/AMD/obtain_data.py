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
import torch
from fvcore.nn import FlopCountAnalysis

# --- Importar GPUMonitor desde la raíz ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from GPUMonitor import GpuMonitor
except ImportError:
    print("Error: No se pudo importar GPUMonitor. Asegúrate de que la carpeta GPUMonitor existe en la raíz.")
    sys.exit(1)


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
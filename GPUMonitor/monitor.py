import threading
import time
import csv
import os
from statistics import mean
from typing import List, Optional

# --- Detección de librerías ---
AMDSMI_AVAILABLE = False
NVML_AVAILABLE = False
TORCH_AVAILABLE = False

# Intentar importar amdsmi
try:
    import amdsmi
    from amdsmi import (
        amdsmi_init,
        amdsmi_shut_down,
        amdsmi_get_processor_handles,
        amdsmi_get_gpu_vram_usage,
        amdsmi_get_power_info,
        amdsmi_get_gpu_activity,
    )
    AMDSMI_AVAILABLE = True
except ImportError:
    pass
except Exception:
    pass

# Intentar importar pynvml
try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetCount,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetUtilizationRates,
    )
    NVML_AVAILABLE = True
except ImportError:
    pass
except Exception:
    pass

# Intentar importar torch (fallback para VRAM en AMD)
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass
except Exception:
    pass


class GpuMonitor:
    def __init__(self, interval: float = 0.1, min_w_usage: int = 0):
        """
        Clase unificada para monitorizar GPUs AMD o NVIDIA.
        Detecta automáticamente qué librería está disponible.
        
        interval: segundos entre mediciones
        min_w_usage: umbral en mW para comenzar a registrar (útil para no grabar idle)
        """
        self.interval = interval
        self.min_w_usage = min_w_usage
        self.running = False
        self.thread = None
        self.backend = None  # 'AMD' o 'NVIDIA' o None
        self.gpus = []

        # Listas de datos
        # Comunes
        self.vram_usage: List[List[float]] = []
        self.power: List[List[float]] = []
        
        # Específicos AMD
        self.util_gfx: List[List[float]] = []
        self.util_umc: List[List[float]] = []
        self.util_mm: List[List[float]] = []
        
        # Específicos NVIDIA
        self.util_gpu: List[List[float]] = []

        self._init_backend()

    def _init_backend(self):
        # Prioridad: Si ambos están (raro), podríamos decidir. 
        # Aquí asumimos que si está amdsmi es AMD, si no, probamos NVML.
        
        if AMDSMI_AVAILABLE:
            try:
                amdsmi_init()
                self.gpus = amdsmi_get_processor_handles()
                if self.gpus:
                    self.backend = 'AMD'
                    print(f"[GPUMonitor] Backend AMD detectado. {len(self.gpus)} GPUs encontradas.")
            except Exception as e:
                print("[GPUMonitor] Error inicializando amdsmi:", e)

        if not self.backend and NVML_AVAILABLE:
            try:
                nvmlInit()
                count = nvmlDeviceGetCount()
                self.gpus = [nvmlDeviceGetHandleByIndex(i) for i in range(count)]
                if self.gpus:
                    self.backend = 'NVIDIA'
                    print(f"[GPUMonitor] Backend NVIDIA detectado. {len(self.gpus)} GPUs encontradas.")
            except Exception as e:
                print("[GPUMonitor] Error inicializando NVML:", e)

        if not self.backend:
            print("[GPUMonitor] Aviso: No se detectó backend compatible (ni AMD ni NVIDIA) o no hay GPUs.")
            
        # Inicializar listas según cantidad de GPUs
        num_gpus = len(self.gpus)
        self.vram_usage = [[] for _ in range(num_gpus)]
        self.power = [[] for _ in range(num_gpus)]
        
        if self.backend == 'AMD':
            self.util_gfx = [[] for _ in range(num_gpus)]
            self.util_umc = [[] for _ in range(num_gpus)]
            self.util_mm = [[] for _ in range(num_gpus)]
        elif self.backend == 'NVIDIA':
            self.util_gpu = [[] for _ in range(num_gpus)]

    def start(self):
        if self.running or not self.backend:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        """Para el thread de monitoreo pero NO cierra el backend.
        Esto permite reutilizar el monitor múltiples veces."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        # Nota: NO llamamos a nvmlShutdown/amdsmi_shut_down aquí
        # para permitir reutilizar el monitor. Usar shutdown() explícitamente.

    def shutdown(self):
        """Cierra el backend completamente. Llamar solo una vez al final del programa."""
        self.stop()  # Asegurar que el thread está detenido
        if self.backend == 'AMD':
            try:
                amdsmi_shut_down()
            except:
                pass
        elif self.backend == 'NVIDIA':
            try:
                nvmlShutdown()
            except:
                pass
        self.backend = None
        self.gpus = []

    def __del__(self):
        """Destructor: intenta cerrar el backend si no se hizo explícitamente."""
        if self.backend:
            self.shutdown()

    def _monitor(self):
        start_recording = False
        while self.running:
            if not self.gpus:
                time.sleep(self.interval)
                continue
            
            try:
                # Lógica de inicio por umbral de potencia
                if not start_recording:
                    current_power_mw = 0
                    if self.backend == 'AMD':
                        # amdsmi devuelve mW en average_socket_power? 
                        # En el código original: amdsmi_get_power_info(g)["average_socket_power"]
                        # Asumimos que es mW según el uso original.
                        current_power_mw = max([amdsmi_get_power_info(g)["average_socket_power"] for g in self.gpus])
                    elif self.backend == 'NVIDIA':
                        # nvmlDeviceGetPowerUsage devuelve mW
                        current_power_mw = max([nvmlDeviceGetPowerUsage(g) for g in self.gpus])
                    
                    if current_power_mw > self.min_w_usage:
                        start_recording = True
                
                if start_recording:
                    for idx, g in enumerate(self.gpus):
                        if self.backend == 'AMD':
                            # Intentar obtener VRAM de amdsmi
                            mem = 0
                            try:
                                vram_info = amdsmi_get_gpu_vram_usage(g)
                                mem = vram_info.get("vram_used", 0) or vram_info.get("used", 0)
                            except:
                                pass
                            
                            # Fallback a PyTorch si amdsmi reporta < 1MB (bug en ROCm 6.3.3)
                            if mem < 1024 * 1024 and TORCH_AVAILABLE:
                                try:
                                    mem = torch.cuda.memory_allocated(idx)
                                except:
                                    pass
                            
                            pwr = amdsmi_get_power_info(g)["average_socket_power"]
                            
                            self.vram_usage[idx].append(float(mem))
                            self.power[idx].append(float(pwr)) 
                            
                            util = amdsmi_get_gpu_activity(g)
                            self.util_gfx[idx].append(float(util['gfx_activity']))
                            self.util_umc[idx].append(float(util['umc_activity']))
                            val_mm = util['mm_activity']
                            self.util_mm[idx].append(float(0 if val_mm == "N/A" else val_mm))

                        elif self.backend == 'NVIDIA':
                            mem = nvmlDeviceGetMemoryInfo(g).used
                            pwr_mw = nvmlDeviceGetPowerUsage(g)
                            pwr_w = pwr_mw / 1000.0
                            
                            self.vram_usage[idx].append(float(mem))
                            self.power[idx].append(float(pwr_w))
                            
                            util = nvmlDeviceGetUtilizationRates(g).gpu
                            self.util_gpu[idx].append(float(util))

            except Exception as e:
                print(f"Warning: error leyendo {self.backend}: {e}")
            
            time.sleep(self.interval)

    def clear(self):
        num_gpus = len(self.gpus)
        self.vram_usage = [[] for _ in range(num_gpus)]
        self.power = [[] for _ in range(num_gpus)]
        if self.backend == 'AMD':
            self.util_gfx = [[] for _ in range(num_gpus)]
            self.util_umc = [[] for _ in range(num_gpus)]
            self.util_mm = [[] for _ in range(num_gpus)]
        elif self.backend == 'NVIDIA':
            self.util_gpu = [[] for _ in range(num_gpus)]

    def get_stats(self):
        stats = {}
        for i in range(len(self.gpus)):
            # VRAM y Power comunes
            vram_clean = [x for x in self.vram_usage[i] if x > 0] if self.vram_usage[i] else []
            power_clean = [x for x in self.power[i] if x > 0] if self.power[i] else []
            
            stats[f"gpu_{i}_vram_avg_bytes"] = mean(vram_clean) if vram_clean else None
            stats[f"gpu_{i}_power_avg_w"] = mean(power_clean) if power_clean else None
            stats[f"gpu_{i}_samples"] = len(self.vram_usage[i])

            if self.backend == 'AMD':
                stats[f"gpu_{i}_util_avg_pct_gfx"] = mean(self.util_gfx[i]) if self.util_gfx[i] else None
                stats[f"gpu_{i}_util_avg_pct_umc"] = mean(self.util_umc[i]) if self.util_umc[i] else None
                stats[f"gpu_{i}_util_avg_pct_mm"] = mean(self.util_mm[i]) if self.util_mm[i] else None
            elif self.backend == 'NVIDIA':
                util_clean = [x for x in self.util_gpu[i] if x > 0] if self.util_gpu[i] else []
                stats[f"gpu_{i}_util_avg_pct"] = mean(util_clean) if util_clean else None
        
        return stats

    def export_to_csv(self, filename_prefix="gpu_stats", output_path="Data", units="MiB"):
        os.makedirs(output_path, exist_ok=True)
        for idx in range(len(self.gpus)):
            filename = os.path.join(output_path, f"{filename_prefix}_gpu{idx}.csv")
            with open(filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                unit_label = "VRAM (MiB)" if units == "MiB" else "VRAM (Bytes)"
                
                # Headers dinámicos
                header = ["time_s", unit_label, "Power_W"]
                if self.backend == 'AMD':
                    header.extend(["Util_pct_gfx", "Util_pct_umc", "Util_pct_mm"])
                elif self.backend == 'NVIDIA':
                    header.append("Util_pct")
                
                writer.writerow(header)
                
                # Escribir filas
                num_samples = len(self.vram_usage[idx])
                for i in range(num_samples):
                    vram_val = self.vram_usage[idx][i]
                    if units == "MiB":
                        vram_val = vram_val / (1024.0 * 1024.0)
                    
                    row = [
                        round(i * self.interval, 6),
                        vram_val,
                        self.power[idx][i] if i < len(self.power[idx]) else None
                    ]
                    
                    if self.backend == 'AMD':
                        row.append(self.util_gfx[idx][i] if i < len(self.util_gfx[idx]) else None)
                        row.append(self.util_umc[idx][i] if i < len(self.util_umc[idx]) else None)
                        row.append(self.util_mm[idx][i] if i < len(self.util_mm[idx]) else None)
                    elif self.backend == 'NVIDIA':
                        row.append(self.util_gpu[idx][i] if i < len(self.util_gpu[idx]) else None)
                    
                    writer.writerow(row)

    def export_epoch_resume(self, filename_prefix="epoch_resume", output_path="Data", epoch: int = 0):
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, f"{filename_prefix}.csv")
        file_exists = os.path.isfile(filename)

        if not self.gpus:
            return

        time_s = (len(self.vram_usage[0]) * self.interval) if self.vram_usage and self.vram_usage[0] else 0.0
        
        # Calcular medias
        mean_vram = [(mean(l) / (1024.0 * 1024.0)) if l else None for l in self.vram_usage]
        mean_power = [mean(l) if l else None for l in self.power]
        
        mean_util_gfx = []
        mean_util_umc = []
        mean_util_mm = []
        mean_util_gpu = []

        if self.backend == 'AMD':
            mean_util_gfx = [mean(l) if l else None for l in self.util_gfx]
            mean_util_umc = [mean(l) if l else None for l in self.util_umc]
            mean_util_mm = [mean(l) if l else None for l in self.util_mm]
        elif self.backend == 'NVIDIA':
            mean_util_gpu = [mean(l) if l else None for l in self.util_gpu]

        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                header = ["Epoch", "Time_s"] + \
                    [f"Mean_VRAM_MiB_GPU{i}" for i in range(len(self.gpus))] + \
                    [f"Mean_Power_W_GPU{i}" for i in range(len(self.gpus))]
                
                if self.backend == 'AMD':
                    header += [f"Mean_Util_pct_gfx_GPU{i}" for i in range(len(self.gpus))]
                    header += [f"Mean_Util_pct_umc_GPU{i}" for i in range(len(self.gpus))]
                    header += [f"Mean_Util_pct_mm_GPU{i}" for i in range(len(self.gpus))]
                elif self.backend == 'NVIDIA':
                    header += [f"Mean_Util_pct_GPU{i}" for i in range(len(self.gpus))]
                
                writer.writerow(header)

            row = [epoch, round(time_s, 6)] + \
                [round(v, 6) if v is not None else "" for v in mean_vram] + \
                [round(p, 6) if p is not None else "" for p in mean_power]
            
            if self.backend == 'AMD':
                row += [round(u, 6) if u is not None else "" for u in mean_util_gfx]
                row += [round(u, 6) if u is not None else "" for u in mean_util_umc]
                row += [round(u, 6) if u is not None else "" for u in mean_util_mm]
            elif self.backend == 'NVIDIA':
                row += [round(u, 6) if u is not None else "" for u in mean_util_gpu]

            writer.writerow(row)

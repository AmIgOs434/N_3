"""
Конфигурация AI Clip Creator с системой уровней производительности
"""

import os
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional


class PerformanceLevel(IntEnum):
    """Уровни производительности (1 = минимум, 5 = максимум)"""
    MINIMAL = 1      # Слабый CPU, минимальное качество
    LOW = 2          # Средний CPU, базовое качество
    MEDIUM = 3       # Хороший CPU или слабый GPU, хорошее качество
    HIGH = 4         # Мощный CPU или средний GPU, отличное качество
    ULTRA = 5        # Топовый GPU, максимальное качество


@dataclass
class PerformanceConfig:
    """Конфигурация для конкретного уровня производительности"""
    
    # Whisper модель
    whisper_model: str
    
    # YOLO модель
    yolo_model: str
    yolo_confidence: float
    
    # Частота детекции
    detect_every_n_frames: int
    
    # Анализ
    enable_ai_analysis: bool
    enable_scene_detection: bool
    enable_audio_analysis: bool
    enable_visual_analysis: bool
    
    # Качество вывода
    output_bitrate: str
    output_fps: int
    
    # Сегментация
    max_segments: int
    min_segment_duration: float
    max_segment_duration: float
    
    # GPU
    use_gpu: bool
    fp16: bool  # Mixed precision для GPU
    
    # Оптимизация
    batch_size: int
    num_workers: int


# Предустановки для каждого уровня
PERFORMANCE_PRESETS = {
    PerformanceLevel.MINIMAL: PerformanceConfig(
        whisper_model="tiny",
        yolo_model="yolov8n.pt",
        yolo_confidence=0.4,
        detect_every_n_frames=10,
        enable_ai_analysis=False,
        enable_scene_detection=False,
        enable_audio_analysis=False,
        enable_visual_analysis=False,
        output_bitrate="4M",
        output_fps=24,
        max_segments=3,
        min_segment_duration=17.0,  # ХАРДКОД: 17 секунд
        max_segment_duration=60.0,  # ХАРДКОД: 60 секунд
        use_gpu=False,
        fp16=False,
        batch_size=1,
        num_workers=2,
    ),
    
    PerformanceLevel.LOW: PerformanceConfig(
        whisper_model="base",
        yolo_model="yolov8n.pt",
        yolo_confidence=0.4,
        detect_every_n_frames=5,
        enable_ai_analysis=False,
        enable_scene_detection=True,
        enable_audio_analysis=False,
        enable_visual_analysis=False,
        output_bitrate="6M",
        output_fps=30,
        max_segments=5,
        min_segment_duration=17.0,  # ХАРДКОД: 17 секунд
        max_segment_duration=60.0,  # ХАРДКОД: 60 секунд
        use_gpu=False,
        fp16=False,
        batch_size=1,
        num_workers=2,
    ),
    
    PerformanceLevel.MEDIUM: PerformanceConfig(
        whisper_model="small",
        yolo_model="yolov8s.pt",
        yolo_confidence=0.35,
        detect_every_n_frames=3,
        enable_ai_analysis=True,
        enable_scene_detection=True,
        enable_audio_analysis=True,
        enable_visual_analysis=False,
        output_bitrate="8M",
        output_fps=30,
        max_segments=8,
        min_segment_duration=17.0,  # ХАРДКОД: 17 секунд
        max_segment_duration=60.0,  # ХАРДКОД: 60 секунд
        use_gpu=True,
        fp16=True,
        batch_size=2,
        num_workers=4,
    ),
    
    PerformanceLevel.HIGH: PerformanceConfig(
        whisper_model="medium",
        yolo_model="yolov8m.pt",
        yolo_confidence=0.3,
        detect_every_n_frames=2,
        enable_ai_analysis=True,
        enable_scene_detection=True,
        enable_audio_analysis=True,
        enable_visual_analysis=True,
        output_bitrate="10M",
        output_fps=30,
        max_segments=10,
        min_segment_duration=17.0,  # ХАРДКОД: 17 секунд
        max_segment_duration=60.0,  # ХАРДКОД: 60 секунд
        use_gpu=True,
        fp16=True,
        batch_size=4,
        num_workers=4,
    ),
    
    PerformanceLevel.ULTRA: PerformanceConfig(
        whisper_model="large-v2",
        yolo_model="yolov8x.pt",
        yolo_confidence=0.25,
        detect_every_n_frames=1,
        enable_ai_analysis=True,
        enable_scene_detection=True,
        enable_audio_analysis=True,
        enable_visual_analysis=True,
        output_bitrate="15M",
        output_fps=30,
        max_segments=15,
        min_segment_duration=17.0,  # ХАРДКОД: 17 секунд
        max_segment_duration=60.0,  # ХАРДКОД: 60 секунд
        use_gpu=True,
        fp16=True,
        batch_size=8,
        num_workers=8,
    ),
}


def detect_environment() -> dict:
    """Определяет окружение запуска"""
    is_colab = 'COLAB_GPU' in os.environ or Path('/content').exists()
    
    has_gpu = False
    gpu_name = None
    gpu_memory_gb = 0
    
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except:
        pass
    
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    return {
        'is_colab': is_colab,
        'has_gpu': has_gpu,
        'gpu_name': gpu_name,
        'gpu_memory_gb': gpu_memory_gb,
        'cpu_count': cpu_count,
    }


def get_recommended_level() -> PerformanceLevel:
    """Рекомендует уровень производительности на основе окружения"""
    env = detect_environment()
    
    if not env['has_gpu']:
        # CPU only
        if env['cpu_count'] >= 8:
            return PerformanceLevel.LOW
        else:
            return PerformanceLevel.MINIMAL
    
    # GPU доступен
    gpu_memory = env['gpu_memory_gb']
    
    if gpu_memory >= 20:  # A100, V100
        return PerformanceLevel.ULTRA
    elif gpu_memory >= 14:  # T4, RTX 3080
        return PerformanceLevel.HIGH
    elif gpu_memory >= 8:  # GTX 1080, RTX 2060
        return PerformanceLevel.MEDIUM
    else:
        return PerformanceLevel.LOW


def get_config(level: Optional[PerformanceLevel] = None) -> PerformanceConfig:
    """
    Получить конфигурацию для указанного уровня
    Если level=None, выбирается автоматически
    """
    if level is None:
        level = get_recommended_level()
    
    config = PERFORMANCE_PRESETS[level]
    
    # Автоотключение GPU если недоступен
    env = detect_environment()
    if config.use_gpu and not env['has_gpu']:
        config.use_gpu = False
        config.fp16 = False
    
    return config


def print_environment_info():
    """Выводит информацию об окружении"""
    env = detect_environment()
    recommended = get_recommended_level()
    
    print("="*70)
    print("🖥️  ИНФОРМАЦИЯ ОБ ОКРУЖЕНИИ")
    print("="*70)
    
    print(f"\n📍 Среда: {'Google Colab' if env['is_colab'] else 'Локальная'}")
    print(f"🔢 CPU ядер: {env['cpu_count']}")
    
    if env['has_gpu']:
        print(f"✅ GPU: {env['gpu_name']}")
        print(f"💾 VRAM: {env['gpu_memory_gb']:.1f} GB")
    else:
        print("❌ GPU: Не обнаружен")
    
    print(f"\n⭐ Рекомендуемый уровень: {recommended} ({recommended.name})")
    
    print("\n📊 ДОСТУПНЫЕ УРОВНИ:")
    print("-"*70)
    
    for level in PerformanceLevel:
        config = PERFORMANCE_PRESETS[level]
        gpu_req = "🎮 GPU" if config.use_gpu else "🖥️  CPU"
        
        print(f"{level}. {level.name:10} - {gpu_req} - "
              f"Whisper:{config.whisper_model:6} - "
              f"YOLO:{config.yolo_model} - "
              f"AI:{'✅' if config.enable_ai_analysis else '❌'}")
    
    print("="*70)


if __name__ == "__main__":
    print_environment_info()
    
    print("\n🧪 ТЕСТ КОНФИГУРАЦИЙ:")
    print("-"*70)
    
    for level in PerformanceLevel:
        config = get_config(level)
        print(f"\nLevel {level} ({level.name}):")
        print(f"  Whisper: {config.whisper_model}")
        print(f"  YOLO: {config.yolo_model}")
        print(f"  GPU: {config.use_gpu}")
        print(f"  AI Analysis: {config.enable_ai_analysis}")

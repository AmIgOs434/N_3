"""
AI Clip Creator - УНИВЕРСАЛЬНАЯ ВЕРСИЯ
Работает в Google Colab и локально
С системой уровней производительности (1-5)
ИСПРАВЛЕНО: строгие лимиты 17-60 секунд, быстрый рендеринг
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time

import numpy as np
import cv2

from config import PerformanceConfig, PerformanceLevel, get_config, detect_environment

logger = logging.getLogger(__name__)


class TrackingMode(Enum):
    """Режимы трекинга камеры"""
    PERSON = "person"
    FACE = "face"
    SPEAKER = "speaker"
    STATIC_CENTER = "static_center"
    STATIC_WEBCAM = "static_webcam"
    HORIZONTAL = "horizontal"


class VideoMode(Enum):
    """Типы контента"""
    PODCAST = "podcast"
    STREAM = "stream"
    DYNAMIC = "dynamic"
    TALKING_HEAD = "talking_head"


@dataclass
class AudioMetrics:
    """Метрики аудио для сегмента"""
    avg_volume: float
    peak_volume: float
    speech_rate: float
    silence_ratio: float
    energy_variance: float


@dataclass
class VisualMetrics:
    """Визуальные метрики для сегмента"""
    motion_score: float
    face_coverage: float
    scene_changes: int
    brightness_variance: float
    color_variance: float
    visual_complexity: float


@dataclass
class EngagementScore:
    """Комплексная оценка вирусности сегмента"""
    total_score: float
    hook_score: float
    audio_score: float
    visual_score: float
    content_score: float
    pacing_score: float
    
    def __str__(self):
        return f"Total: {self.total_score:.1f} (Hook: {self.hook_score:.1f}, Content: {self.content_score:.1f})"


@dataclass
class VideoSegment:
    """Сегмент видео с метриками"""
    start: float
    end: float
    title: str
    score: float
    tags: List[str]
    
    audio_metrics: Optional[AudioMetrics] = None
    visual_metrics: Optional[VisualMetrics] = None
    engagement: Optional[EngagementScore] = None
    transcript_text: str = ""
    hook_text: str = ""
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ProcessingConfig:
    """Конфигурация обработки видео"""
    
    input_path: str
    output_dir: str
    mode: VideoMode = VideoMode.DYNAMIC
    tracking_mode: TrackingMode = TrackingMode.PERSON
    
    # Уровень производительности
    performance_level: PerformanceLevel = None
    
    # Качество вывода
    output_width: int = 1080
    output_height: int = 1920
    fps: int = 30
    output_bitrate: str = "8M"
    
    # GPU настройки
    use_gpu: bool = True
    force_cpu: bool = False
    fp16: bool = True
    
    # Детекция
    detect_every_n_frames: int = 3
    max_speed_px_per_sec: float = 180.0
    target_ema_alpha: float = 0.25
    
    # Модели
    yolo_model: str = "yolov8s.pt"
    yolo_confidence: float = 0.3
    
    # Субтитры
    whisper_model: str = "small"
    whisper_language: str = "ru"
    subtitle_font_size: int = 58
    subtitle_position: float = 0.82
    
    # Сегментация - СТРОГИЕ ЛИМИТЫ
    min_segment_duration: float = 17.0  # ХАРДКОД: минимум 17 секунд
    max_segment_duration: float = 60.0  # ХАРДКОД: максимум 60 секунд
    max_segments: int = 8
    
    # AI анализ
    use_ai_analysis: bool = True
    openai_api_key: Optional[str] = "sk-Yqz5qU3hmVLtKHDnmdtctNNvmcWxiKZK"
    openai_base_url: str = "https://api.proxyapi.ru/openai/v1"
    openai_model: str = "gpt-5-nano"
    
    # Дополнительные анализы
    enable_emotion_detection: bool = False
    enable_audio_analysis: bool = True
    enable_visual_saliency: bool = False
    enable_engagement_scoring: bool = True
    enable_scene_detection: bool = True
    
    # Пороги качества
    min_engagement_score: float = 40.0
    min_hook_score: float = 30.0
    
    temp_dir: str = None
    
    def apply_performance_preset(self, perf_config: PerformanceConfig):
        """Применить настройки производительности"""
        self.whisper_model = perf_config.whisper_model
        self.yolo_model = perf_config.yolo_model
        self.yolo_confidence = perf_config.yolo_confidence
        self.detect_every_n_frames = perf_config.detect_every_n_frames
        self.use_ai_analysis = perf_config.enable_ai_analysis
        self.enable_scene_detection = perf_config.enable_scene_detection
        self.enable_audio_analysis = perf_config.enable_audio_analysis
        self.enable_visual_saliency = perf_config.enable_visual_analysis
        self.output_bitrate = perf_config.output_bitrate
        self.fps = perf_config.output_fps
        self.max_segments = perf_config.max_segments
        # КРИТИЧНО: применяем строгие лимиты из конфига
        self.min_segment_duration = perf_config.min_segment_duration
        self.max_segment_duration = perf_config.max_segment_duration
        self.use_gpu = perf_config.use_gpu
        self.fp16 = perf_config.fp16


def detect_device():
    """Определить доступное устройство (CPU/GPU)"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ GPU detected: {gpu_name}")
            logger.info(f"  VRAM: {vram:.1f} GB")
            return device, gpu_name, vram
        else:
            logger.info("No GPU available, using CPU")
            return 'cpu', None, 0
            
    except ImportError:
        logger.warning("PyTorch not installed, using CPU")
        return 'cpu', None, 0


def setup_temp_directories(temp_dir: str = None):
    """Настройка временных директорий"""
    env = detect_environment()
    
    if temp_dir is None:
        if env['is_colab']:
            temp_dir = "/content/temp_ai_clip"
        else:
            if sys.platform == 'win32':
                import shutil
                drives = []
                for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
                    drive = f"{letter}:\\"
                    if os.path.exists(drive):
                        try:
                            stat = shutil.disk_usage(drive)
                            free_gb = stat.free / (1024**3)
                            drives.append((drive, free_gb))
                        except:
                            pass
                
                if drives:
                    drives.sort(key=lambda x: x[1], reverse=True)
                    best_drive = drives[0][0]
                    temp_dir = os.path.join(best_drive, "temp_ai_clip")
                    logger.info(f"Selected drive {best_drive} with {drives[0][1]:.1f}GB free space")
                else:
                    temp_dir = str(Path.home() / "temp_ai_clip")
            else:
                temp_dir = str(Path.home() / "temp_ai_clip")
    
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    
    tempfile.tempdir = str(temp_path)
    os.environ['TEMP'] = str(temp_path)
    os.environ['TMP'] = str(temp_path)
    os.environ['TMPDIR'] = str(temp_path)
    os.environ['FFMPEG_TMPDIR'] = str(temp_path)
    
    whisper_cache = temp_path / "whisper_cache"
    whisper_cache.mkdir(exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(whisper_cache)
    
    if sys.platform == 'win32':
        os.environ['HF_HOME'] = str(temp_path / "huggingface")
        os.environ['TORCH_HOME'] = str(temp_path / "torch")
        os.environ['TRANSFORMERS_CACHE'] = str(temp_path / "transformers")
    
    logger.info(f"Temporary directories set to: {temp_path}")
    logger.info(f"Available space: {get_disk_space(temp_path):.2f} GB")
    
    return temp_path


def get_disk_space(path: Path) -> float:
    """Получить свободное место на диске в GB"""
    import shutil
    stat = shutil.disk_usage(path)
    return stat.free / (1024**3)


class VideoInfo:
    """Информация о видео файле"""
    
    def __init__(self, path: str):
        self.path = Path(path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        
        self._info = None
        self._load_info()
    
    def _load_info(self):
        """Загрузка информации через ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(self.path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                encoding='utf-8',
                errors='replace'
            )
            
            if not result.stdout:
                raise RuntimeError("ffprobe returned empty output")
            
            self._info = json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    @property
    def duration(self) -> float:
        try:
            return float(self._info['format']['duration'])
        except:
            return 0.0
    
    @property
    def width(self) -> int:
        for stream in self._info.get('streams', []):
            if stream.get('codec_type') == 'video':
                return int(stream.get('width', 0))
        return 0
    
    @property
    def height(self) -> int:
        for stream in self._info.get('streams', []):
            if stream.get('codec_type') == 'video':
                return int(stream.get('height', 0))
        return 0
    
    @property
    def fps(self) -> float:
        for stream in self._info.get('streams', []):
            if stream.get('codec_type') == 'video':
                fps_str = stream.get('r_frame_rate', '30/1')
                try:
                    num, den = map(int, fps_str.split('/'))
                    return num / den if den != 0 else 30.0
                except:
                    return 30.0
        return 30.0
    
    @property
    def has_audio(self) -> bool:
        for stream in self._info.get('streams', []):
            if stream.get('codec_type') == 'audio':
                return True
        return False
    
    @property
    def total_frames(self) -> int:
        return int(self.duration * self.fps)


class VideoProcessor:
    """УНИВЕРСАЛЬНЫЙ процессор видео"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Применяем performance preset если указан
        if config.performance_level is not None:
            perf_config = get_config(config.performance_level)
            config.apply_performance_preset(perf_config)
            level_name = config.performance_level.name if hasattr(config.performance_level, 'name') else str(config.performance_level)
            logger.info(f"Applied performance preset: Level {config.performance_level} ({level_name})")
        
        # ПРОВЕРКА ЛИМИТОВ
        logger.info(f"⚠️  SEGMENT LIMITS: min={config.min_segment_duration}s, max={config.max_segment_duration}s")
        
        # Определяем устройство
        if config.force_cpu:
            self.device = 'cpu'
            self.gpu_name = None
            self.vram = 0
            logger.info("🔧 Force CPU mode")
        elif config.use_gpu:
            self.device, self.gpu_name, self.vram = detect_device()
        else:
            self.device = 'cpu'
            self.gpu_name = None
            self.vram = 0
            logger.info("Using CPU (GPU disabled in config)")
        
        self.temp_dir = setup_temp_directories(config.temp_dir)
        
        free_space = get_disk_space(self.temp_dir)
        if free_space < 5.0:
            logger.warning(f"Low disk space: {free_space:.2f}GB. Minimum 5GB recommended.")
        
        if not Path(config.input_path).exists():
            raise FileNotFoundError(f"Input video not found: {config.input_path}")
        
        # Быстрая проверка кодека без зависания
        logger.info("Quick codec compatibility check...")
        self.video_compatible = self._quick_codec_check(config.input_path)
        
        if not self.video_compatible:
            logger.warning("⚠️  Video codec may be incompatible (AV1/VP9)")
            logger.warning("⚠️  Visual analysis will be DISABLED")
            config.enable_scene_detection = False
            config.enable_visual_saliency = False
        else:
            logger.info("✅ Video codec compatible")
        
        self.video_info = VideoInfo(config.input_path)
        
        self._init_detectors()
        
        logger.info("=" * 60)
        logger.info("VIDEO PROCESSOR INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device.upper()}")
        if self.device == 'cuda':
            logger.info(f"GPU: {self.gpu_name}")
            logger.info(f"VRAM: {self.vram:.1f} GB")
        logger.info(f"Video: {self.video_info.width}x{self.video_info.height}, "
                   f"{self.video_info.duration:.2f}s, {self.video_info.fps:.2f}fps")
        logger.info(f"YOLO Model: {config.yolo_model}")
        logger.info(f"Whisper Model: {config.whisper_model}")
        logger.info(f"Temp Dir: {self.temp_dir} (Free: {free_space:.2f}GB)")
        logger.info(f"SEGMENT LIMITS: {config.min_segment_duration}-{config.max_segment_duration}s")
        logger.info("=" * 60)
    
    def _quick_codec_check(self, video_path: str, timeout: int = 5) -> bool:
        """Быстрая проверка кодека (5 секунд максимум)"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                cap.release()
                return False
            
            # Пробуем прочитать 3 кадра с таймаутом
            import time
            start_time = time.time()
            success_count = 0
            
            for i in range(3):
                if time.time() - start_time > timeout:
                    logger.warning("Codec check timeout")
                    break
                
                ret, frame = cap.read()
                if ret and frame is not None:
                    success_count += 1
            
            cap.release()
            
            # Если хотя бы 2 из 3 кадров прочитались - кодек OK
            return success_count >= 2
            
        except Exception as e:
            logger.error(f"Codec check error: {e}")
            return False
    
    def _init_detectors(self):
        """Инициализация детекторов"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        self.yolo_model = None
        
        logger.info("✓ Detectors initialized")
    
    def _load_yolo(self):
        """Загрузка YOLO модели"""
        if self.yolo_model is None:
            try:
                from ultralytics import YOLO
                
                logger.info(f"Loading YOLO {self.config.yolo_model} on {self.device}...")
                self.yolo_model = YOLO(self.config.yolo_model)
                
                if self.device == 'cuda':
                    self.yolo_model.to('cuda')
                    logger.info(f"✓ YOLO using GPU: {self.gpu_name}")
                else:
                    logger.info(f"✓ YOLO using CPU")
                    
            except Exception as e:
                logger.error(f"Failed to load YOLO: {e}")
                logger.info("Falling back to face detection only")
    
    async def process(self, progress_callback=None) -> List[str]:
        """Обработка видео"""
        try:
            logger.info("Starting video processing...")
            
            if progress_callback:
                await progress_callback("Анализ видео...", 0, 100)
            
            # Глубокий анализ
            segments = await self._analyze_and_segment(progress_callback)
            
            if not segments:
                logger.warning("No segments found with current settings")
                segments = self._simple_segmentation()
                
                if not segments:
                    raise RuntimeError("No suitable segments found")
            
            logger.info(f"Selected {len(segments)} segments")
            
            # ПРОВЕРКА ЛИМИТОВ ПЕРЕД РЕНДЕРИНГОМ
            logger.info("\n🔍 FINAL SEGMENT VALIDATION:")
            valid_segments = []
            for i, seg in enumerate(segments):
                duration = seg.duration
                logger.info(f"  Segment {i+1}: {duration:.1f}s ({seg.start:.1f}-{seg.end:.1f})")
                
                if duration < self.config.min_segment_duration:
                    logger.warning(f"    ❌ TOO SHORT (min {self.config.min_segment_duration}s) - SKIPPED")
                elif duration > self.config.max_segment_duration:
                    logger.warning(f"    ❌ TOO LONG (max {self.config.max_segment_duration}s) - TRIMMING")
                    # Обрезаем до максимума
                    seg.end = seg.start + self.config.max_segment_duration
                    valid_segments.append(seg)
                else:
                    logger.info(f"    ✅ OK")
                    valid_segments.append(seg)
            
            segments = valid_segments
            
            if not segments:
                raise RuntimeError("No valid segments after duration check")
            
            logger.info(f"\n✓ {len(segments)} valid segments ready for rendering\n")
            
            # Рендер
            output_files = []
            total_segments = len(segments)
            
            for i, segment in enumerate(segments):
                if progress_callback:
                    progress = int((i / total_segments) * 100)
                    await progress_callback(
                        f"Рендер {i+1}/{total_segments}: {segment.title}",
                        progress,
                        100
                    )
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing segment {i+1}/{total_segments}")
                logger.info(f"Title: {segment.title}")
                logger.info(f"Duration: {segment.duration:.1f}s ({segment.start:.1f}-{segment.end:.1f})")
                if segment.engagement:
                    logger.info(f"Engagement: {segment.engagement}")
                logger.info(f"{'='*60}\n")
                
                output_file = await self._process_segment(segment, i, progress_callback)
                output_files.append(output_file)
            
            if progress_callback:
                await progress_callback("✓ Обработка завершена!", 100, 100)
            
            logger.info("\n" + "="*60)
            logger.info(f"PROCESSING COMPLETE")
            logger.info(f"Created {len(output_files)} clips")
            logger.info("="*60 + "\n")
            
            return output_files
            
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            raise
    
    async def _analyze_and_segment(self, progress_callback=None) -> List[VideoSegment]:
        """Анализ видео и сегментация"""
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS STARTING")
        logger.info("="*60)
        
        # 1. Транскрипция (0-25%)
        if progress_callback:
            await progress_callback("Транскрипция (Whisper)...", 0, 100)
        
        transcript = await self._transcribe(progress_callback)
        logger.info(f"✓ Transcription: {len(transcript.get('words', []))} words")
        
        # 2. Детекция сцен (25-40%) - ТОЛЬКО если кодек совместим
        scene_changes = []
        if self.config.enable_scene_detection and self.video_compatible:
            if progress_callback:
                await progress_callback("Scene detection...", 25, 100)
            
            scene_changes = await self._detect_scenes_fast(progress_callback)
            logger.info(f"✓ Scene detection: {len(scene_changes)} changes")
        else:
            logger.info("✓ Scene detection: skipped (incompatible codec)")
        
        # 3. Аудио анализ (40-55%)
        audio_data = {'times': [], 'rms': [], 'zcr': [], 'spectral_centroid': [], 'tempo': 0, 'duration': 0}
        if self.config.enable_audio_analysis:
            if progress_callback:
                await progress_callback("Анализ аудио...", 40, 100)
            
            audio_data = await self._analyze_audio_metrics(progress_callback)
            logger.info(f"✓ Audio analysis complete")
        
        # 4. Визуальный анализ - ПРОПУСКАЕМ если кодек несовместим
        visual_data = {'times': [], 'motion': [], 'faces': [], 'brightness': [], 'saturation': []}
        if self.config.enable_visual_saliency and self.video_compatible:
            if progress_callback:
                await progress_callback("Анализ визуала...", 55, 100)
            
            visual_data = await self._analyze_visual_metrics(progress_callback)
            logger.info(f"✓ Visual analysis complete")
        else:
            logger.info(f"✓ Visual analysis skipped")
        
        # 5. Генерация кандидатов (70-80%)
        if progress_callback:
            await progress_callback("Поиск моментов...", 70, 100)
        
        candidate_segments = self._generate_candidate_segments(
            transcript, scene_changes, audio_data, visual_data
        )
        logger.info(f"✓ Generated {len(candidate_segments)} candidates")
        
        # 6. AI scoring (80-95%)
        scored_segments = candidate_segments
        if self.config.use_ai_analysis and self.config.openai_api_key:
            if progress_callback:
                await progress_callback("AI анализ вирусности...", 80, 100)
            
            scored_segments = await self._ai_scoring(
                candidate_segments, transcript, progress_callback
            )
            logger.info(f"✓ AI scoring complete")
        else:
            # Простой scoring без AI, но добавляем транскрипт
            words = transcript.get('words', [])
            for seg in scored_segments:
                seg.engagement = EngagementScore(
                    hook_score=50.0,
                    content_score=50.0,
                    audio_score=50.0,
                    visual_score=50.0,
                    pacing_score=50.0,
                    total_score=50.0
                )
                # Добавляем транскрипт для субтитров
                segment_words = [
                    w for w in words
                    if seg.start <= w['start'] <= seg.end
                ]
                seg.transcript_text = " ".join(w['word'] for w in segment_words)
        
        # 7. Фильтрация (95-100%)
        if progress_callback:
            await progress_callback("Отбор лучших...", 95, 100)
        
        final_segments = self._filter_and_rank_segments(scored_segments)
        
        logger.info("\n" + "="*60)
        logger.info(f"ANALYSIS COMPLETE: {len(final_segments)} segments")
        for i, seg in enumerate(final_segments, 1):
            score_str = f"Score: {seg.engagement.total_score:.1f}" if seg.engagement else ""
            duration_str = f"{seg.duration:.1f}s"
            logger.info(f"  {i}. {seg.title} - {duration_str} {score_str}")
        logger.info("="*60 + "\n")
        
        return final_segments
    
    async def _transcribe(self, progress_callback=None) -> Dict[str, Any]:
        """Транскрипция с Whisper"""
        logger.info(f"Starting transcription (Whisper {self.config.whisper_model} on {self.device})...")
        
        try:
            import stable_whisper
            
            model_cache_dir = self.temp_dir / "whisper_models"
            model_cache_dir.mkdir(exist_ok=True)
            os.environ['WHISPER_CACHE'] = str(model_cache_dir)
            
            logger.info(f"Loading Whisper model on {self.device}...")
            model = stable_whisper.load_model(
                self.config.whisper_model,
                device=self.device,
                download_root=str(model_cache_dir)
            )
            
            temp_audio = self.temp_dir / f"audio_{os.getpid()}.wav"
            
            logger.info("Extracting audio...")
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.config.input_path),
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                str(temp_audio)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
            
            logger.info("Running Whisper transcription...")
            start_time = time.time()
            
            result = model.transcribe(
                str(temp_audio),
                language=self.config.whisper_language,
                word_timestamps=True,
                verbose=False,
                vad=True,
                fp16=(self.device == 'cuda' and self.config.fp16)
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✓ Transcription complete in {elapsed:.1f}s")
            
            if temp_audio.exists():
                temp_audio.unlink()
            
            words = []
            for segment in result.segments:
                for word in segment.words:
                    words.append({
                        'word': word.word.strip(),
                        'start': float(word.start),
                        'end': float(word.end),
                        'confidence': float(getattr(word, 'confidence', 1.0))
                    })
            
            return {
                'text': result.text,
                'words': words,
                'language': result.language,
                'segments': result.segments
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return {'text': '', 'words': [], 'language': 'ru', 'segments': []}
    
    async def _detect_scenes_fast(self, progress_callback=None) -> List[float]:
        """Быстрая детекция сцен БЕЗ зависания"""
        logger.info("Fast scene detection (sample-based)...")
        
        cap = cv2.VideoCapture(str(self.config.input_path))
        
        if not cap.isOpened():
            logger.error("Failed to open video")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0 or fps == 0:
            logger.warning("Invalid video properties")
            cap.release()
            return []
        
        scene_changes = []
        prev_frame = None
        threshold = 27.0
        
        # УСКОРЕНИЕ: анализируем только каждый N-й кадр
        sample_rate = 10  # каждый 10-й кадр
        frame_idx = 0
        analyzed = 0
        max_analyze = min(500, total_frames // sample_rate)  # макс 500 проверок
        
        logger.info(f"Analyzing {max_analyze} sampled frames...")
        
        while analyzed < max_analyze:
            # Прыгаем к следующему сэмплу
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, gray)
                    mean_diff = diff.mean()
                    
                    if mean_diff > threshold:
                        timestamp = frame_idx / fps
                        scene_changes.append(timestamp)
                
                prev_frame = gray
                
            except Exception as e:
                logger.warning(f"Error at frame {frame_idx}: {e}")
            
            frame_idx += sample_rate
            analyzed += 1
            
            # Прогресс каждые 10%
            if analyzed % (max_analyze // 10) == 0:
                if progress_callback:
                    progress = int((analyzed / max_analyze) * 100)
                    await progress_callback(
                        f"Scene detection: {progress}% ({len(scene_changes)} scenes)",
                        25 + int(progress * 0.15),
                        100
                    )
        
        cap.release()
        logger.info(f"✓ Fast scene detection: {len(scene_changes)} scenes")
        
        return scene_changes
    
    async def _analyze_audio_metrics(self, progress_callback=None) -> Dict[str, Any]:
        """Анализ аудио метрик"""
        logger.info("Analyzing audio metrics...")
        
        try:
            import librosa
            
            temp_audio = self.temp_dir / f"audio_analysis_{os.getpid()}.wav"
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.config.input_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '22050', '-ac', '1',
                str(temp_audio)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
            
            y, sr = librosa.load(str(temp_audio), sr=22050)
            
            rms = librosa.feature.rms(y=y)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo)
            except:
                tempo = 120.0
            
            times = librosa.frames_to_time(range(len(rms)), sr=sr)
            
            audio_data = {
                'times': [float(t) for t in times],
                'rms': [float(r) for r in rms],
                'zcr': [float(z) for z in zcr],
                'spectral_centroid': [float(s) for s in spectral_centroid],
                'tempo': tempo,
                'duration': float(len(y) / sr)
            }
            
            if temp_audio.exists():
                temp_audio.unlink()
            
            logger.info(f"✓ Audio analysis: tempo={tempo:.1f} BPM")
            return audio_data
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return {'times': [], 'rms': [], 'zcr': [], 'spectral_centroid': [], 'tempo': 0, 'duration': 0}
    
    async def _analyze_visual_metrics(self, progress_callback=None) -> Dict[str, Any]:
        """Анализ визуальных метрик (быстрая версия)"""
        logger.info("Analyzing visual metrics (sampled)...")
        
        cap = cv2.VideoCapture(str(self.config.input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        visual_data = {
            'times': [],
            'motion': [],
            'faces': [],
            'brightness': [],
            'saturation': []
        }
        
        prev_gray = None
        sample_rate = 30  # каждый 30-й кадр
        max_samples = min(200, total_frames // sample_rate)
        
        for i in range(max_samples):
            frame_idx = i * sample_rate
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            visual_data['times'].append(timestamp)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion = diff.mean()
                visual_data['motion'].append(float(motion))
            else:
                visual_data['motion'].append(0.0)
            
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            visual_data['faces'].append(len(faces))
            
            visual_data['brightness'].append(float(gray.mean()))
            visual_data['saturation'].append(float(hsv[:,:,1].mean()))
            
            prev_gray = gray
        
        cap.release()
        logger.info(f"✓ Visual analysis: {len(visual_data['times'])} samples")
        return visual_data
    
    def _generate_candidate_segments(
        self,
        transcript: Dict,
        scenes: List[float],
        audio_data: Dict,
        visual_data: Dict
    ) -> List[VideoSegment]:
        """Генерация кандидатов с СОБЛЮДЕНИЕМ ЛИМИТОВ"""
        logger.info("Generating candidate segments...")
        
        MIN_DUR = self.config.min_segment_duration
        MAX_DUR = self.config.max_segment_duration
        
        logger.info(f"  Limits: {MIN_DUR}s - {MAX_DUR}s")
        
        candidates = []
        words = transcript.get('words', [])
        
        if not words:
            return self._simple_segmentation()
        
        # Сегменты вокруг сцен
        for i, scene_time in enumerate(scenes):
            # Центрируем вокруг сцены, но соблюдаем лимиты
            duration = min(MAX_DUR, 30.0)  # предпочитаем 30s если это < MAX
            start = max(0, scene_time - 5.0)
            end = min(self.video_info.duration, start + duration)
            
            # Корректируем если вышли за MAX
            if end - start > MAX_DUR:
                end = start + MAX_DUR
            
            if end - start >= MIN_DUR:
                candidates.append(VideoSegment(
                    start=start,
                    end=end,
                    title=f"Scene {i+1}",
                    score=0.5,
                    tags=["scene_change"]
                ))
        
        # Сегменты на основе пауз
        for i in range(len(words) - 1):
            pause = words[i+1]['start'] - words[i]['end']
            
            if pause > 1.5:
                # Начинаем за 10s до паузы
                start = max(0, words[i]['end'] - 10.0)
                # Берем 25-30 секунд или до следующей паузы
                target_dur = min(MAX_DUR, 30.0)
                end = min(self.video_info.duration, start + target_dur)
                
                # Проверка лимитов
                actual_dur = end - start
                if actual_dur < MIN_DUR:
                    # Пытаемся растянуть
                    end = min(self.video_info.duration, start + MIN_DUR)
                    actual_dur = end - start
                
                if actual_dur > MAX_DUR:
                    end = start + MAX_DUR
                    actual_dur = MAX_DUR
                
                if MIN_DUR <= actual_dur <= MAX_DUR:
                    candidates.append(VideoSegment(
                        start=start,
                        end=end,
                        title=f"Pause {i}",
                        score=0.5,
                        tags=["pause"]
                    ))
        
        # Если кандидатов мало - добавляем равномерную нарезку
        if len(candidates) < 3:
            logger.info("Few candidates, adding uniform segments")
            uniform_segments = self._simple_segmentation()
            candidates.extend(uniform_segments)
        
        candidates = self._merge_overlapping_segments(candidates)
        
        # ФИНАЛЬНАЯ ПРОВЕРКА всех кандидатов
        valid_candidates = []
        for seg in candidates:
            dur = seg.duration
            if dur < MIN_DUR:
                logger.debug(f"  Candidate too short: {dur:.1f}s < {MIN_DUR}s - skipped")
            elif dur > MAX_DUR:
                logger.debug(f"  Candidate too long: {dur:.1f}s > {MAX_DUR}s - trimming")
                seg.end = seg.start + MAX_DUR
                valid_candidates.append(seg)
            else:
                valid_candidates.append(seg)
        
        logger.info(f"Generated {len(valid_candidates)} valid candidates")
        return valid_candidates
    
    def _merge_overlapping_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Объединяет перекрывающиеся сегменты"""
        if not segments:
            return []
        
        sorted_segs = sorted(segments, key=lambda s: s.start)
        merged = [sorted_segs[0]]
        
        for seg in sorted_segs[1:]:
            last = merged[-1]
            
            if seg.start <= last.end:
                # При объединении соблюдаем MAX лимит
                new_end = min(
                    max(last.end, seg.end),
                    last.start + self.config.max_segment_duration
                )
                
                merged[-1] = VideoSegment(
                    start=last.start,
                    end=new_end,
                    title=last.title,
                    score=max(last.score, seg.score),
                    tags=list(set(last.tags + seg.tags))
                )
            else:
                merged.append(seg)
        
        return merged
    
    async def _ai_scoring(
        self,
        segments: List[VideoSegment],
        transcript: Dict,
        progress_callback=None
    ) -> List[VideoSegment]:
        """AI scoring с GPT"""
        logger.info(f"AI scoring for {len(segments)} segments...")
        
        if not self.config.openai_api_key:
            logger.warning("No OpenAI key - using simple scoring")
            for seg in segments:
                seg.engagement = EngagementScore(
                    hook_score=50.0,
                    content_score=50.0,
                    audio_score=50.0,
                    visual_score=50.0,
                    pacing_score=50.0,
                    total_score=50.0
                )
            return segments
        
        from openai import OpenAI
        client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url
        )
        
        words = transcript.get('words', [])
        full_text = transcript.get('text', '')
        
        scored_segments = []
        
        for i, segment in enumerate(segments):
            if progress_callback:
                progress = int((i / len(segments)) * 100)
                await progress_callback(
                    f"AI scoring {i+1}/{len(segments)}...",
                    80 + int(progress * 0.15),
                    100
                )
            
            segment_words = [
                w for w in words
                if segment.start <= w['start'] <= segment.end
            ]
            segment_text = " ".join(w['word'] for w in segment_words)
            
            hook_words = [
                w for w in segment_words
                if w['start'] - segment.start <= 3.0
            ]
            hook_text = " ".join(w['word'] for w in hook_words)
            
            prompt = f"""Ты эксперт по вирусным видео (TikTok/YouTube Shorts).

Оцени этот сегмент:

КОНТЕКСТ:
{full_text[:1500]}

СЕГМЕНТ ({segment.duration:.1f}s):
{segment_text}

ХУК (первые 3s):
{hook_text}

Оцени (0-100):
1. HOOK_SCORE: Цепляет ли начало?
2. CONTENT_SCORE: Качество контента
3. VIRAL_POTENTIAL: Вирусный потенциал
4. TITLE: Заголовок (до 40 символов)
5. TAGS: 3 тега

JSON:
{{
  "hook_score": 75,
  "content_score": 70,
  "viral_potential": 65,
  "title": "Заголовок",
  "tags": ["тег1", "тег2", "тег3"]
}}
"""
            
            try:
                response = client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content
                
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    
                    engagement = EngagementScore(
                        hook_score=float(data.get('hook_score', 50)),
                        content_score=float(data.get('content_score', 50)),
                        audio_score=60.0,
                        visual_score=60.0,
                        pacing_score=55.0,
                        total_score=0.0
                    )
                    
                    engagement.total_score = (
                        engagement.hook_score * 0.3 +
                        engagement.content_score * 0.25 +
                        float(data.get('viral_potential', 50)) * 0.25 +
                        engagement.audio_score * 0.1 +
                        engagement.visual_score * 0.1
                    )
                    
                    segment.title = data.get('title', segment.title)
                    segment.tags = data.get('tags', segment.tags)
                    segment.engagement = engagement
                    segment.transcript_text = segment_text  # ДОБАВЛЕНО для субтитров
                    segment.hook_text = hook_text
                    
                    logger.info(f"  {i+1}. {segment.title} - {engagement.total_score:.1f}")
                    
            except Exception as e:
                logger.error(f"AI error for segment {i+1}: {e}")
                segment.engagement = EngagementScore(
                    hook_score=50.0,
                    content_score=50.0,
                    audio_score=50.0,
                    visual_score=50.0,
                    pacing_score=50.0,
                    total_score=50.0
                )
                segment.transcript_text = segment_text  # ДОБАВЛЕНО для субтитров даже при ошибке
            
            scored_segments.append(segment)
        
        return scored_segments
    
    def _filter_and_rank_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Фильтрация и ранжирование С ПРОВЕРКОЙ ДЛИТЕЛЬНОСТИ"""
        logger.info(f"Filtering {len(segments)} segments...")
        
        MIN_DUR = self.config.min_segment_duration
        MAX_DUR = self.config.max_segment_duration
        
        filtered = [
            seg for seg in segments
            if seg.engagement and
               seg.engagement.total_score >= self.config.min_engagement_score and
               seg.engagement.hook_score >= self.config.min_hook_score and
               MIN_DUR <= seg.duration <= MAX_DUR  # КРИТИЧНО!
        ]
        
        logger.info(f"After filter: {len(filtered)} segments")
        
        if len(filtered) < 2 and len(segments) > 0:
            logger.info("Too few filtered segments, relaxing duration limits slightly")
            # Берем сегменты с небольшим отклонением от лимитов
            filtered = [
                s for s in segments
                if s.engagement and
                   (MIN_DUR * 0.9) <= s.duration <= (MAX_DUR * 1.1)
            ]
            
            if len(filtered) < 2:
                logger.info("Still too few, taking top by score")
                filtered = sorted(
                    [s for s in segments if s.engagement],
                    key=lambda s: s.engagement.total_score,
                    reverse=True
                )[:self.config.max_segments]
        
        ranked = sorted(filtered, key=lambda s: s.engagement.total_score if s.engagement else 0, reverse=True)
        top_segments = ranked[:self.config.max_segments]
        final = sorted(top_segments, key=lambda s: s.start)
        
        return final
    
    def _simple_segmentation(self) -> List[VideoSegment]:
        """Fallback равномерная нарезка С СОБЛЮДЕНИЕМ ЛИМИТОВ"""
        logger.info("Using simple segmentation")
        
        MIN_DUR = self.config.min_segment_duration
        MAX_DUR = self.config.max_segment_duration
        
        segments = []
        current_time = 0.0
        segment_idx = 0
        
        # Предпочитаемая длина - что-то посередине между MIN и MAX
        preferred_duration = min(MAX_DUR, max(MIN_DUR, (MIN_DUR + MAX_DUR) / 2))
        
        logger.info(f"  Simple segmentation: {MIN_DUR}s - {MAX_DUR}s, preferred={preferred_duration:.1f}s")
        
        while current_time < self.video_info.duration and segment_idx < self.config.max_segments:
            end_time = min(
                current_time + preferred_duration,
                self.video_info.duration
            )
            
            duration = end_time - current_time
            
            # Проверяем лимиты
            if duration >= MIN_DUR:
                # Обрезаем если слишком длинный
                if duration > MAX_DUR:
                    end_time = current_time + MAX_DUR
                    duration = MAX_DUR
                
                segments.append(VideoSegment(
                    start=current_time,
                    end=end_time,
                    title=f"Segment {segment_idx + 1}",
                    score=0.5,
                    tags=[],
                    engagement=EngagementScore(
                        hook_score=50.0,
                        content_score=50.0,
                        audio_score=50.0,
                        visual_score=50.0,
                        pacing_score=50.0,
                        total_score=50.0
                    )
                ))
                segment_idx += 1
            
            current_time = end_time
        
        logger.info(f"Created {len(segments)} simple segments")
        return segments
    
    async def _process_segment(
        self,
        segment: VideoSegment,
        index: int,
        progress_callback=None
    ) -> str:
        """Обработка сегмента С ТРЕКИНГОМ И СУБТИТРАМИ"""
        logger.info(f"Processing segment {index+1}...")
        
        safe_title = "".join(c for c in segment.title if c.isalnum() or c in (' ', '-', '_'))[:50]
        output_filename = f"{index+1:02d}_{safe_title}_{int(segment.start)}-{int(segment.end)}.mp4"
        output_path = Path(self.config.output_dir) / output_filename
        
        # Проверяем режим трекинга
        if self.config.tracking_mode == TrackingMode.STATIC_CENTER:
            # Простой центральный crop без трекинга
            logger.info("  Static center crop...")
            return await self._render_static_crop(segment, output_path)
        
        # УМНЫЙ ТРЕКИНГ + СУБТИТРЫ
        logger.info("  Smart tracking + subtitles rendering...")
        
        try:
            # 1. Извлекаем сегмент
            temp_segment = await self._extract_segment(segment)
            
            # 2. Применяем трекинг
            if progress_callback:
                await progress_callback(f"Трекинг объекта {index+1}...", 0, 100)
            
            tracked_video = await self._apply_smart_tracking(temp_segment, segment)
            
            # 3. Добавляем субтитры
            if progress_callback:
                await progress_callback(f"Добавление субтитров {index+1}...", 50, 100)
            
            final_video = await self._add_subtitles(tracked_video, segment)
            
            # 4. Перемещаем в output
            final_video.rename(output_path)
            
            # Очистка
            if temp_segment.exists():
                temp_segment.unlink()
            
            logger.info(f"✓ Segment complete: {output_path.name}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Segment processing error: {e}", exc_info=True)
            # Fallback: простой crop
            logger.warning("Falling back to simple crop...")
            return await self._render_static_crop(segment, output_path)
    
    async def _render_static_crop(self, segment: VideoSegment, output_path: Path) -> str:
        """Простой статичный crop по центру"""
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(segment.start),
            '-t', str(segment.duration),
            '-i', str(self.config.input_path),
            '-vf', f'crop={self.config.output_width}:{self.config.output_height}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-b:v', self.config.output_bitrate,
            '-c:a', 'aac',
            '-b:a', '128k',
            str(output_path)
        ]
        
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=300
        )
        
        return str(output_path)
    
    async def _apply_smart_tracking(self, video_path: Path, segment: VideoSegment) -> Path:
        """Применить умный трекинг к видео"""
        output_path = self.temp_dir / f"tracked_{os.getpid()}_{int(segment.start)}.mp4"
        
        # Загружаем YOLO если не загружена
        self._load_yolo()
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.warning("Cannot open video for tracking, using simple crop")
            video_path.rename(output_path)
            return output_path
        
        # Параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Инициализация трекера
        from tracking import SmoothTracker, HaarFaceDetector
        
        tracker = SmoothTracker(
            frame_width=width,
            frame_height=height,
            target_width=self.config.output_width,
            target_height=self.config.output_height,
            max_speed=self.config.max_speed_px_per_sec,
            ema_alpha=self.config.target_ema_alpha
        )
        
        # Детектор
        detector = None
        if self.yolo_model and self.config.tracking_mode == TrackingMode.PERSON:
            # Используем уже загруженную YOLO модель
            detector = self.yolo_model
            use_yolo = True
            logger.info("Using YOLO person detector")
        else:
            detector = HaarFaceDetector()
            use_yolo = False
            logger.info("Using Haar face detector")
        
        # Подготовка writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.config.output_width, self.config.output_height)
        )
        
        frame_idx = 0
        
        logger.info(f"Processing {total_frames} frames with tracking...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Детекция каждый N-й кадр
            detection = None
            if frame_idx % self.config.detect_every_n_frames == 0:
                try:
                    if use_yolo:
                        # YOLO детекция
                        results = detector(frame, verbose=False, classes=[0])  # class 0 = person
                        
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            boxes = results[0].boxes
                            best_result = None
                            best_score = 0
                            
                            for box in boxes:
                                conf = float(box.conf[0])
                                if conf < self.config.yolo_confidence:
                                    continue
                                
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                width_box = x2 - x1
                                height_box = y2 - y1
                                area = width_box * height_box
                                
                                score = conf * np.sqrt(area)
                                
                                if score > best_score:
                                    best_score = score
                                    center_x = int((x1 + x2) / 2)
                                    center_y = int((y1 + y2) / 2)
                                    
                                    from tracking import TrackingResult
                                    detection = TrackingResult(
                                        bbox=(int(x1), int(y1), int(width_box), int(height_box)),
                                        confidence=conf,
                                        center=(center_x, center_y),
                                        area=area
                                    )
                    else:
                        # Haar детекция
                        detection = detector.detect(frame)
                        
                except Exception as e:
                    logger.debug(f"Detection error at frame {frame_idx}: {e}")
            
            # Обновить позицию трекера
            center_x, center_y = tracker.update(detection, fps)
            
            # Получить bbox для кропа
            x1, y1, x2, y2 = tracker.get_crop_bbox()
            
            # Crop кадра
            cropped = frame[y1:y2, x1:x2]
            
            # Resize если нужно (на случай краев)
            if cropped.shape[:2] != (self.config.output_height, self.config.output_width):
                cropped = cv2.resize(
                    cropped,
                    (self.config.output_width, self.config.output_height)
                )
            
            writer.write(cropped)
            frame_idx += 1
            
            # Прогресс каждые 5%
            if frame_idx % (max(1, total_frames // 20)) == 0:
                progress = int((frame_idx / total_frames) * 100)
                logger.debug(f"  Tracking progress: {progress}%")
        
        cap.release()
        writer.release()
        
        # Добавляем аудио обратно
        output_with_audio = self.temp_dir / f"tracked_audio_{os.getpid()}_{int(segment.start)}.mp4"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(output_path),
            '-i', str(video_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-shortest',
            str(output_with_audio)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
        
        # Cleanup
        if output_path.exists():
            output_path.unlink()
        if video_path.exists():
            video_path.unlink()
        
        logger.info("✓ Smart tracking applied")
        return output_with_audio
    
    async def _add_subtitles(self, video_path: Path, segment: VideoSegment) -> Path:
        """Добавить субтитры к видео"""
        output_path = self.temp_dir / f"subtitled_{os.getpid()}_{int(segment.start)}.mp4"
        
        # Если нет транскрипта - возвращаем как есть
        if not hasattr(segment, 'transcript_text') or not segment.transcript_text:
            logger.info("No transcript, skipping subtitles")
            video_path.rename(output_path)
            return output_path
        
        try:
            # Создаем SRT файл
            srt_path = self.temp_dir / f"subs_{os.getpid()}_{int(segment.start)}.srt"
            
            # Получаем слова для этого сегмента из транскрипции
            # (в реальности нужен доступ к полной транскрипции, используем текст)
            
            # Простая временная разметка (равномерно распределяем слова)
            words = segment.transcript_text.split()
            if not words:
                logger.info("No words in transcript, skipping subtitles")
                video_path.rename(output_path)
                return output_path
            
            duration = segment.duration
            time_per_word = duration / len(words)
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                # Группируем по 3-5 слов на субтитр
                words_per_group = 4
                for i in range(0, len(words), words_per_group):
                    group = words[i:i+words_per_group]
                    start_time = i * time_per_word
                    end_time = min((i + len(group)) * time_per_word, duration)
                    
                    f.write(f"{i//words_per_group + 1}\n")
                    f.write(f"{self._format_srt_time(start_time)} --> {self._format_srt_time(end_time)}\n")
                    f.write(f"{' '.join(group)}\n\n")
            
            # Применяем субтитры через ffmpeg
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-vf', (
                    f"subtitles={str(srt_path)}:"
                    f"force_style='FontName=Arial,FontSize={self.config.subtitle_font_size},"
                    f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2,"
                    f"Alignment=2,MarginV={int(self.config.output_height * (1 - self.config.subtitle_position))}'"
                ),
                '-c:a', 'copy',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
            
            # Cleanup
            if srt_path.exists():
                srt_path.unlink()
            if video_path.exists():
                video_path.unlink()
            
            logger.info("✓ Subtitles added")
            return output_path
            
        except Exception as e:
            logger.error(f"Subtitle error: {e}")
            # Fallback: без субтитров
            if not output_path.exists():
                video_path.rename(output_path)
            return output_path
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Форматировать время для SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    async def _extract_segment(self, segment: VideoSegment) -> Path:
        """Извлечение сегмента (УСТАРЕЛО - используем прямой рендеринг)"""
        temp_file = self.temp_dir / f"segment_{os.getpid()}_{int(segment.start)}.mp4"
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(segment.start),
            '-t', str(segment.duration),
            '-i', str(self.config.input_path),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            str(temp_file)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
        return temp_file
    
    async def _create_vertical(self, input_path: Path, segment: VideoSegment) -> Path:
        """Вертикальное видео (УСТАРЕЛО - используем прямой рендеринг)"""
        output_path = self.temp_dir / f"vertical_{os.getpid()}_{int(segment.start)}.mp4"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-vf', f'crop={self.config.output_width}:{self.config.output_height}',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-b:v', self.config.output_bitrate,
            '-c:a', 'copy',
            str(output_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, encoding='utf-8', errors='replace')
        
        if input_path.exists():
            input_path.unlink()
        
        return output_path


if __name__ == "__main__":
    import asyncio
    from config import print_environment_info
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print_environment_info()

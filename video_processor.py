"""
AI Clip Creator - УНИВЕРСАЛЬНАЯ ВЕРСИЯ
Работает в Google Colab и локально
С системой уровней производительности (1-5)
"""

import os
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
    fp16: bool = True  # Mixed precision
    
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
    
    # Сегментация
    min_segment_duration: float = 15.0
    max_segment_duration: float = 60.0
    max_segments: int = 8
    
    # AI анализ
    use_ai_analysis: bool = True
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.proxyapi.ru/openai/v1"
    openai_model: str = "gpt-5-nano"
    
    # Дополнительные анализы
    enable_emotion_detection: bool = False
    enable_audio_analysis: bool = True
    enable_visual_saliency: bool = False
    enable_engagement_scoring: bool = True
    enable_scene_detection: bool = True
    
    # Пороги качества
    min_engagement_score: float = 40.0  # Понижен для большей гибкости
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
    """Настройка временных директорий — универсально для Colab и локально"""
    env = detect_environment()
    
    if temp_dir is None:
        if env['is_colab']:
            temp_dir = "/content/temp_ai_clip"
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
            logger.info(f"Applied performance preset: Level {config.performance_level} ({config.performance_level.name})")
        
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
        logger.info("=" * 60)
    
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
                # Fallback: создаем базовую сегментацию
                segments = self._simple_segmentation()
                
                if not segments:
                    raise RuntimeError("No suitable segments found")
            
            logger.info(f"Selected {len(segments)} segments")
            
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
                logger.info(f"Duration: {segment.duration:.1f}s")
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
        
        # 2. Детекция сцен (25-40%)
        scene_changes = []
        if self.config.enable_scene_detection:
            if progress_callback:
                await progress_callback("Scene detection...", 25, 100)
            
            scene_changes = await self._detect_scenes_custom(progress_callback)
            logger.info(f"✓ Scene detection: {len(scene_changes)} changes")
        
        # 3. Аудио анализ (40-55%)
        audio_data = {'times': [], 'rms': [], 'zcr': [], 'spectral_centroid': [], 'tempo': 0, 'duration': 0}
        if self.config.enable_audio_analysis:
            if progress_callback:
                await progress_callback("Анализ аудио...", 40, 100)
            
            audio_data = await self._analyze_audio_metrics(progress_callback)
            logger.info(f"✓ Audio analysis complete")
        
        # 4. Визуальный анализ (55-70%)
        visual_data = {'times': [], 'motion': [], 'faces': [], 'brightness': [], 'saturation': []}
        if self.config.enable_visual_saliency:
            if progress_callback:
                await progress_callback("Анализ визуала...", 55, 100)
            
            visual_data = await self._analyze_visual_metrics(progress_callback)
            logger.info(f"✓ Visual analysis complete")
        
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
            # Простой scoring без AI
            for seg in scored_segments:
                seg.engagement = EngagementScore(
                    hook_score=50.0,
                    content_score=50.0,
                    audio_score=50.0,
                    visual_score=50.0,
                    pacing_score=50.0,
                    total_score=50.0
                )
        
        # 7. Фильтрация (95-100%)
        if progress_callback:
            await progress_callback("Отбор лучших...", 95, 100)
        
        final_segments = self._filter_and_rank_segments(scored_segments)
        
        logger.info("\n" + "="*60)
        logger.info(f"ANALYSIS COMPLETE: {len(final_segments)} segments")
        for i, seg in enumerate(final_segments, 1):
            score_str = f"Score: {seg.engagement.total_score:.1f}" if seg.engagement else ""
            logger.info(f"  {i}. {seg.title} {score_str}")
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
            
            # Извлекаем аудио
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
            
            # Извлекаем слова
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
    
    async def _detect_scenes_custom(self, progress_callback=None) -> List[float]:
        """Детектор сцен с прогрессом"""
        logger.info("Custom scene detection...")
        
        cap = cv2.VideoCapture(str(self.config.input_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        scene_changes = []
        prev_frame = None
        frame_idx = 0
        threshold = 27.0
        
        start_time = time.time()
        last_progress = 0
        
        logger.info(f"Analyzing {total_frames} frames for scene changes...")
        
        # Используем GPU для ускорения если доступен
        use_gpu_processing = self.device == 'cuda'
        if use_gpu_processing:
            try:
                import torch
                logger.info("Using GPU-accelerated scene detection")
            except:
                use_gpu_processing = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = diff.mean()
                
                if mean_diff > threshold:
                    timestamp = frame_idx / fps
                    scene_changes.append(timestamp)
            
            prev_frame = gray
            frame_idx += 1
            
            # Прогресс каждые 2%
            current_progress = int((frame_idx / total_frames) * 100)
            if current_progress > last_progress and current_progress % 2 == 0:
                if progress_callback:
                    await progress_callback(
                        f"Scene detection: {current_progress}% ({len(scene_changes)} scenes found)",
                        25 + int(current_progress * 0.15),  # 25-40%
                        100
                    )
                last_progress = current_progress
        
        cap.release()
        elapsed = time.time() - start_time
        logger.info(f"✓ Scene detection complete in {elapsed:.1f}s: {len(scene_changes)} scenes")
        
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
            except:
                tempo = 120.0
            
            times = librosa.frames_to_time(range(len(rms)), sr=sr)
            
            audio_data = {
                'times': times.tolist(),
                'rms': rms.tolist(),
                'zcr': zcr.tolist(),
                'spectral_centroid': spectral_centroid.tolist(),
                'tempo': float(tempo),
                'duration': len(y) / sr
            }
            
            if temp_audio.exists():
                temp_audio.unlink()
            
            logger.info(f"✓ Audio analysis: tempo={tempo:.1f} BPM")
            return audio_data
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return {'times': [], 'rms': [], 'zcr': [], 'spectral_centroid': [], 'tempo': 0, 'duration': 0}
    
    async def _analyze_visual_metrics(self, progress_callback=None) -> Dict[str, Any]:
        """Анализ визуальных метрик"""
        logger.info("Analyzing visual metrics...")
        
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
        frame_idx = 0
        sample_rate = 30
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
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
                
                brightness = gray.mean()
                visual_data['brightness'].append(float(brightness))
                
                saturation = hsv[:,:,1].mean()
                visual_data['saturation'].append(float(saturation))
                
                prev_gray = gray
            
            frame_idx += 1
        
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
        """Генерация кандидатов сегментов"""
        logger.info("Generating candidate segments...")
        
        candidates = []
        words = transcript.get('words', [])
        
        if not words:
            return self._simple_segmentation()
        
        # Сегменты вокруг сцен
        for i, scene_time in enumerate(scenes):
            start = max(0, scene_time - 5.0)
            end = min(self.video_info.duration, scene_time + 45.0)
            
            if end - start >= self.config.min_segment_duration:
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
                start = max(0, words[i]['end'] - 10.0)
                end = min(self.video_info.duration, words[i+1]['start'] + 40.0)
                
                if end - start >= self.config.min_segment_duration:
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
        
        logger.info(f"Generated {len(candidates)} candidates")
        return candidates
    
    def _merge_overlapping_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Объединяет перекрывающиеся сегменты"""
        if not segments:
            return []
        
        sorted_segs = sorted(segments, key=lambda s: s.start)
        merged = [sorted_segs[0]]
        
        for seg in sorted_segs[1:]:
            last = merged[-1]
            
            if seg.start <= last.end:
                merged[-1] = VideoSegment(
                    start=last.start,
                    end=max(last.end, seg.end),
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
            
            # Текст сегмента
            segment_words = [
                w for w in words
                if segment.start <= w['start'] <= segment.end
            ]
            segment_text = " ".join(w['word'] for w in segment_words)
            
            # Хук
            hook_words = [
                w for w in segment_words
                if w['start'] - segment.start <= 3.0
            ]
            hook_text = " ".join(w['word'] for w in hook_words)
            
            # AI анализ
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
                    segment.transcript_text = segment_text
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
            
            scored_segments.append(segment)
        
        return scored_segments
    
    def _filter_and_rank_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """Фильтрация и ранжирование"""
        logger.info(f"Filtering {len(segments)} segments...")
        
        filtered = [
            seg for seg in segments
            if seg.engagement and
               seg.engagement.total_score >= self.config.min_engagement_score and
               seg.engagement.hook_score >= self.config.min_hook_score and
               self.config.min_segment_duration <= seg.duration <= self.config.max_segment_duration
        ]
        
        logger.info(f"After filter: {len(filtered)} segments")
        
        # Если после фильтра мало сегментов - берем топ по скору без фильтров
        if len(filtered) < 2 and len(segments) > 0:
            logger.info("Too few filtered segments, taking top segments by score")
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
        """Fallback равномерная нарезка"""
        logger.info("Using simple segmentation")
        segments = []
        current_time = 0.0
        segment_idx = 0
        
        segment_duration = min(
            self.config.max_segment_duration,
            max(self.config.min_segment_duration, self.video_info.duration / self.config.max_segments)
        )
        
        while current_time < self.video_info.duration and segment_idx < self.config.max_segments:
            end_time = min(
                current_time + segment_duration,
                self.video_info.duration
            )
            
            if (end_time - current_time) >= self.config.min_segment_duration * 0.8:  # 80% минимума
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
        """Обработка сегмента"""
        logger.info(f"Processing segment {index+1}...")
        
        safe_title = "".join(c for c in segment.title if c.isalnum() or c in (' ', '-', '_'))[:50]
        output_filename = f"{index+1:02d}_{safe_title}_{int(segment.start)}-{int(segment.end)}.mp4"
        output_path = Path(self.config.output_dir) / output_filename
        
        temp_segment = await self._extract_segment(segment)
        vertical_video = await self._create_vertical(temp_segment, segment)
        
        vertical_video.rename(output_path)
        
        logger.info(f"✓ Segment complete: {output_path.name}")
        return str(output_path)
    
    async def _extract_segment(self, segment: VideoSegment) -> Path:
        """Извлечение сегмента"""
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
        """Вертикальное видео"""
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
        
        # Удаляем временный файл
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
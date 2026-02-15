"""
Модуль умного трекинга объектов в видео
Поддержка различных алгоритмов: YOLO, Haar Cascade, оптический поток
"""

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TrackerType(Enum):
    """Типы трекеров"""
    YOLO = "yolo"
    HAAR = "haar"
    DNN = "dnn"
    OPTICAL_FLOW = "optical_flow"


@dataclass
class TrackingResult:
    """Результат трекинга"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    center: Tuple[int, int]
    area: float


class SmoothTracker:
    """
    Умный трекер с плавным движением камеры
    
    Использует:
    - EMA (Exponential Moving Average) для сглаживания
    - Ограничение скорости и ускорения
    - Предсказание положения при потере объекта
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        target_width: int,
        target_height: int,
        max_speed: float = 260.0,
        max_acceleration: float = 1400.0,
        ema_alpha: float = 0.35,
        lost_hold_time: float = 3.0,
        deadzone: int = 10
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_width = target_width
        self.target_height = target_height
        
        # Параметры сглаживания
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.ema_alpha = ema_alpha
        self.lost_hold_time = lost_hold_time
        self.deadzone = deadzone
        
        # Текущее состояние
        self.current_x = frame_width // 2
        self.current_y = frame_height // 2
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.target_x = None
        self.target_y = None
        
        # Счетчики
        self.frames_since_detection = 0
        self.total_frames = 0
        
        logger.info(f"SmoothTracker initialized: {frame_width}x{frame_height} -> {target_width}x{target_height}")
    
    def update(
        self,
        detection: Optional[TrackingResult],
        fps: float = 30.0
    ) -> Tuple[int, int]:
        """
        Обновить позицию трекера
        
        Args:
            detection: Результат детекции (None если объект не найден)
            fps: FPS видео для расчета времени
        
        Returns:
            (x, y): Новая позиция центра кропа
        """
        dt = 1.0 / fps
        self.total_frames += 1
        
        if detection is not None:
            # Объект найден
            self.frames_since_detection = 0
            target_x, target_y = detection.center
            
            # Применяем EMA для сглаживания целевой позиции
            if self.target_x is None:
                self.target_x = target_x
                self.target_y = target_y
            else:
                self.target_x = (1 - self.ema_alpha) * self.target_x + self.ema_alpha * target_x
                self.target_y = (1 - self.ema_alpha) * self.target_y + self.ema_alpha * target_y
        
        else:
            # Объект потерян
            self.frames_since_detection += 1
            
            # Если слишком долго нет детекции - возврат к центру
            if self.frames_since_detection > (self.lost_hold_time * fps):
                self.target_x = self.frame_width // 2
                self.target_y = self.frame_height // 2
        
        # Если есть цель - двигаемся к ней
        if self.target_x is not None:
            self._move_towards_target(dt)
        
        return int(self.current_x), int(self.current_y)
    
    def _move_towards_target(self, dt: float):
        """Плавное движение к цели с ограничением скорости"""
        # Вычисляем ошибку
        error_x = self.target_x - self.current_x
        error_y = self.target_y - self.current_y
        
        # Deadzone - не двигаемся если близко
        if abs(error_x) < self.deadzone and abs(error_y) < self.deadzone:
            self.velocity_x = 0
            self.velocity_y = 0
            return
        
        # Желаемая скорость (пропорциональная ошибке)
        desired_vx = np.clip(error_x * 3.0, -self.max_speed, self.max_speed)
        desired_vy = np.clip(error_y * 3.0, -self.max_speed, self.max_speed)
        
        # Ускорение для достижения желаемой скорости
        accel_x = desired_vx - self.velocity_x
        accel_y = desired_vy - self.velocity_y
        
        # Ограничение ускорения
        max_accel_dt = self.max_acceleration * dt
        accel_x = np.clip(accel_x, -max_accel_dt, max_accel_dt)
        accel_y = np.clip(accel_y, -max_accel_dt, max_accel_dt)
        
        # Обновляем скорость
        self.velocity_x += accel_x
        self.velocity_y += accel_y
        
        # Обновляем позицию
        self.current_x += self.velocity_x * dt
        self.current_y += self.velocity_y * dt
        
        # Ограничиваем позицию границами кадра
        half_w = self.target_width // 2
        half_h = self.target_height // 2
        self.current_x = np.clip(self.current_x, half_w, self.frame_width - half_w)
        self.current_y = np.clip(self.current_y, half_h, self.frame_height - half_h)
    
    def get_crop_bbox(self) -> Tuple[int, int, int, int]:
        """Получить bbox для кропа"""
        half_w = self.target_width // 2
        half_h = self.target_height // 2
        
        x1 = int(self.current_x - half_w)
        y1 = int(self.current_y - half_h)
        x2 = int(self.current_x + half_w)
        y2 = int(self.current_y + half_h)
        
        return (x1, y1, x2, y2)


class YOLOPersonDetector:
    """Детектор людей на основе YOLO"""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.confidence = confidence
            logger.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> Optional[TrackingResult]:
        """
        Детекция человека в кадре
        
        Returns:
            TrackingResult для самого крупного/уверенного человека или None
        """
        results = self.model(frame, verbose=False, classes=[0])  # class 0 = person
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        boxes = results[0].boxes
        best_result = None
        best_score = 0
        
        for box in boxes:
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue
            
            # Извлекаем координаты
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # Скор = уверенность * sqrt(площадь)
            # Приоритет большим объектам
            score = conf * np.sqrt(area)
            
            if score > best_score:
                best_score = score
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                best_result = TrackingResult(
                    bbox=(int(x1), int(y1), int(width), int(height)),
                    confidence=conf,
                    center=(center_x, center_y),
                    area=area
                )
        
        return best_result


class HaarFaceDetector:
    """Детектор лиц на основе Haar Cascade (быстрый)"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("Haar Cascade face detector loaded")
    
    def detect(self, frame: np.ndarray) -> Optional[TrackingResult]:
        """
        Детекция лица в кадре
        
        Returns:
            TrackingResult для самого крупного лица или None
        """
        # Конвертируем в grayscale для Haar Cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        if len(faces) == 0:
            return None
        
        # Выбираем самое крупное лицо
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        return TrackingResult(
            bbox=(int(x), int(y), int(w), int(h)),
            confidence=1.0,  # Haar не дает уверенность
            center=(int(x + w/2), int(y + h/2)),
            area=float(w * h)
        )


class DNNFaceDetector:
    """Детектор лиц на основе DNN (более точный, но медленнее)"""
    
    def __init__(self, confidence: float = 0.7):
        # Загружаем предобученную модель
        model_file = "opencv_face_detector_uint8.pb"
        config_file = "opencv_face_detector.pbtxt"
        
        try:
            self.net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            self.confidence = confidence
            logger.info("DNN face detector loaded")
        except Exception as e:
            logger.warning(f"Failed to load DNN detector: {e}")
            logger.info("Falling back to Haar Cascade")
            self.net = None
            self.haar_detector = HaarFaceDetector()
    
    def detect(self, frame: np.ndarray) -> Optional[TrackingResult]:
        """Детекция лица"""
        if self.net is None:
            return self.haar_detector.detect(frame)
        
        h, w = frame.shape[:2]
        
        # Подготовка входа для сети
        blob = cv2.dnn.blobFromImage(
            frame, 
            1.0, 
            (300, 300), 
            [104, 117, 123], 
            False, 
            False
        )
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        best_result = None
        best_score = 0
        
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            
            if confidence < self.confidence:
                continue
            
            # Координаты bbox
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            score = confidence * np.sqrt(area)
            
            if score > best_score:
                best_score = score
                best_result = TrackingResult(
                    bbox=(x1, y1, width, height),
                    confidence=confidence,
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                    area=area
                )
        
        return best_result


def create_detector(detector_type: TrackerType, **kwargs):
    """Фабрика для создания детекторов"""
    if detector_type == TrackerType.YOLO:
        return YOLOPersonDetector(**kwargs)
    elif detector_type == TrackerType.HAAR:
        return HaarFaceDetector()
    elif detector_type == TrackerType.DNN:
        return DNNFaceDetector(**kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


if __name__ == "__main__":
    # Тест трекера
    logging.basicConfig(level=logging.INFO)
    
    tracker = SmoothTracker(
        frame_width=1920,
        frame_height=1080,
        target_width=608,
        target_height=1080
    )
    
    # Симуляция детекции
    for i in range(100):
        if i % 10 == 0:
            # Каждые 10 кадров "детектируем" объект
            detection = TrackingResult(
                bbox=(500 + i*5, 400, 200, 200),
                confidence=0.9,
                center=(600 + i*5, 500),
                area=40000
            )
        else:
            detection = None
        
        x, y = tracker.update(detection)
        print(f"Frame {i}: position=({x}, {y}), velocity=({tracker.velocity_x:.1f}, {tracker.velocity_y:.1f})")

"""
Модуль генерации субтитров с умной разбивкой на строки
и анимацией подсветки текущего слова
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class Word:
    """Слово с временными метками"""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SubtitleGroup:
    """Группа слов для отображения как одна подпись"""
    words: List[Word]
    lines: List[List[int]]  # Индексы слов в каждой строке
    start: float
    end: float
    
    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def get_text(self) -> str:
        """Получить весь текст группы"""
        return " ".join(w.text for w in self.words)
    
    def get_line_text(self, line_idx: int) -> str:
        """Получить текст конкретной строки"""
        if line_idx >= len(self.lines):
            return ""
        word_indices = self.lines[line_idx]
        return " ".join(self.words[i].text for i in word_indices)


class SubtitleEngine:
    """
    Движок для генерации субтитров
    
    Возможности:
    - Умная разбивка на строки без обрезания слов
    - Группировка слов в фразы по паузам
    - Анимация подсветки текущего слова
    - Настраиваемый стиль (шрифт, цвет, позиция)
    """
    
    def __init__(
        self,
        font_path: str = "arial.ttf",
        font_size: int = 56,
        text_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
        highlight_color: Tuple[int, int, int, int] = (255, 215, 0, 255),
        stroke_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        stroke_width: int = 3,
        position_y: float = 0.82,  # От верха (0.0-1.0)
        max_width: float = 0.85,  # От ширины кадра
        max_chars_per_line: int = 26,
        max_lines: int = 2,
        max_words_per_group: int = 10,
        max_duration_per_group: float = 4.0,
        pause_threshold: float = 0.55,
        hold_time: float = 0.35  # Задержка перед исчезновением
    ):
        self.font_path = font_path
        self.font_size = font_size
        self.text_color = text_color
        self.highlight_color = highlight_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.position_y = position_y
        self.max_width = max_width
        
        # Параметры группировки
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
        self.max_words_per_group = max_words_per_group
        self.max_duration_per_group = max_duration_per_group
        self.pause_threshold = pause_threshold
        self.hold_time = hold_time
        
        # Загрузка шрифта
        try:
            self.font = ImageFont.truetype(font_path, font_size)
            logger.info(f"Font loaded: {font_path}, size={font_size}")
        except Exception as e:
            logger.warning(f"Failed to load font: {e}, using default")
            self.font = ImageFont.load_default()
    
    def create_groups(self, words: List[Word]) -> List[SubtitleGroup]:
        """
        Группировка слов в субтитры
        
        Правила:
        - Разрыв при длинной паузе между словами
        - Разрыв при превышении max_words_per_group
        - Разрыв при превышении max_duration_per_group
        """
        if not words:
            return []
        
        groups = []
        current_words = []
        
        for i, word in enumerate(words):
            # Проверка паузы с предыдущим словом
            if current_words:
                prev_word = current_words[-1]
                pause = word.start - prev_word.end
                
                # Проверка условий разрыва
                duration = word.end - current_words[0].start
                should_break = (
                    pause >= self.pause_threshold or
                    len(current_words) >= self.max_words_per_group or
                    duration >= self.max_duration_per_group
                )
                
                if should_break:
                    # Создаем группу из накопленных слов
                    groups.append(self._create_group(current_words))
                    current_words = []
            
            current_words.append(word)
        
        # Последняя группа
        if current_words:
            groups.append(self._create_group(current_words))
        
        logger.info(f"Created {len(groups)} subtitle groups from {len(words)} words")
        return groups
    
    def _create_group(self, words: List[Word]) -> SubtitleGroup:
        """Создать группу с разбивкой на строки"""
        lines = self._split_into_lines(words)
        
        return SubtitleGroup(
            words=words,
            lines=lines,
            start=words[0].start,
            end=words[-1].end
        )
    
    def _split_into_lines(self, words: List[Word]) -> List[List[int]]:
        """
        Умная разбивка слов на строки
        
        Правила:
        - Не разрывать слова
        - Не превышать max_chars_per_line
        - Не превышать max_lines
        """
        lines = []
        current_line = []
        current_length = 0
        
        for i, word in enumerate(words):
            word_len = len(word.text)
            
            # Добавить пробел если не первое слово в строке
            add_length = word_len if not current_line else word_len + 1
            
            # Проверка переполнения строки
            if current_line and (current_length + add_length > self.max_chars_per_line):
                # Сохранить текущую строку
                lines.append(current_line)
                
                # Проверка лимита строк
                if len(lines) >= self.max_lines:
                    return lines
                
                # Начать новую строку
                current_line = [i]
                current_length = word_len
            else:
                current_line.append(i)
                current_length += add_length
        
        # Последняя строка
        if current_line and len(lines) < self.max_lines:
            lines.append(current_line)
        
        return lines
    
    def render_subtitle(
        self,
        frame: np.ndarray,
        groups: List[SubtitleGroup],
        time: float
    ) -> np.ndarray:
        """
        Рендер субтитров на кадр
        
        Args:
            frame: Кадр видео (numpy array BGR)
            groups: Группы субтитров
            time: Текущее время в секундах
        
        Returns:
            Кадр с субтитрами
        """
        # Найти активную группу
        active_group = None
        for group in groups:
            if group.start <= time < (group.end + self.hold_time):
                active_group = group
                break
        
        if not active_group:
            return frame
        
        # Конвертировать в PIL для рисования
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        height, width = frame.shape[:2]
        max_text_width = int(width * self.max_width)
        
        # Найти текущее слово для подсветки
        highlight_word_idx = self._find_current_word(active_group, time)
        
        # Рендер каждой строки
        y_position = int(height * self.position_y)
        
        for line_idx, word_indices in enumerate(active_group.lines):
            line_y = y_position + line_idx * (self.font_size + 10)
            
            # Собрать текст строки
            line_text = " ".join(active_group.words[i].text for i in word_indices)
            
            # Измерить размер текста
            bbox = draw.textbbox((0, 0), line_text, font=self.font)
            text_width = bbox[2] - bbox[0]
            
            # Центрировать текст
            x_position = (width - text_width) // 2
            
            # Рисовать слова по отдельности для подсветки
            current_x = x_position
            for word_idx in word_indices:
                word_text = active_group.words[word_idx].text + " "
                
                # Определить цвет
                color = self.highlight_color if word_idx == highlight_word_idx else self.text_color
                
                # Рисовать с обводкой
                self._draw_text_with_stroke(
                    draw,
                    (current_x, line_y),
                    word_text,
                    font=self.font,
                    fill=color,
                    stroke_fill=self.stroke_color,
                    stroke_width=self.stroke_width
                )
                
                # Обновить позицию
                word_bbox = draw.textbbox((0, 0), word_text, font=self.font)
                current_x += (word_bbox[2] - word_bbox[0])
        
        # Конвертировать обратно в OpenCV
        return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    def _find_current_word(self, group: SubtitleGroup, time: float) -> int:
        """Найти индекс текущего слова для подсветки"""
        for i, word in enumerate(group.words):
            if word.start <= time < word.end:
                return i
        
        # Если время после всех слов - подсветить последнее
        if time >= group.words[-1].end:
            return len(group.words) - 1
        
        return 0
    
    def _draw_text_with_stroke(
        self,
        draw: ImageDraw.Draw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: Tuple[int, int, int, int],
        stroke_fill: Tuple[int, int, int, int],
        stroke_width: int
    ):
        """Рисовать текст с обводкой"""
        x, y = position
        
        # Рисовать обводку
        for offset_x in range(-stroke_width, stroke_width + 1):
            for offset_y in range(-stroke_width, stroke_width + 1):
                if offset_x != 0 or offset_y != 0:
                    draw.text(
                        (x + offset_x, y + offset_y),
                        text,
                        font=font,
                        fill=stroke_fill
                    )
        
        # Рисовать основной текст
        draw.text((x, y), text, font=font, fill=fill)
    
    def export_srt(self, groups: List[SubtitleGroup], output_path: str):
        """Экспорт в SRT формат"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, group in enumerate(groups, 1):
                start_time = self._format_srt_time(group.start)
                end_time = self._format_srt_time(group.end)
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                
                # Писать по строкам
                for line in group.lines:
                    line_text = " ".join(group.words[idx].text for idx in line)
                    f.write(f"{line_text}\n")
                
                f.write("\n")
        
        logger.info(f"SRT exported: {output_path}")
    
    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Форматировать время для SRT (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# Импортируем cv2 для конвертации
import cv2


def test_subtitle_engine():
    """Тест движка субтитров"""
    logging.basicConfig(level=logging.INFO)
    
    # Создание тестовых слов
    words = [
        Word("Привет", 0.0, 0.5),
        Word("это", 0.6, 0.9),
        Word("тестовые", 1.0, 1.5),
        Word("субтитры", 1.6, 2.2),
        Word("для", 2.8, 3.1),
        Word("проверки", 3.2, 3.8),
        Word("работы", 4.0, 4.5),
        Word("системы", 4.6, 5.2),
    ]
    
    engine = SubtitleEngine()
    groups = engine.create_groups(words)
    
    print(f"\nСоздано групп: {len(groups)}")
    for i, group in enumerate(groups):
        print(f"\nГруппа {i+1}: {group.start:.2f}s - {group.end:.2f}s")
        print(f"Слов: {len(group.words)}")
        print(f"Строк: {len(group.lines)}")
        for line_idx, line in enumerate(group.lines):
            print(f"  Строка {line_idx+1}: {group.get_line_text(line_idx)}")
    
    # Экспорт в SRT
    engine.export_srt(groups, "test_subtitles.srt")


if __name__ == "__main__":
    test_subtitle_engine()

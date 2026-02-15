"""
Модуль для работы с YouTube
- Загрузка видео (ИСПРАВЛЕНО для всех версий yt-dlp)
- OAuth авторизация
- Публикация Shorts
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger(__name__)

SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.readonly'
]


def download_youtube_video(url: str, output_dir: str) -> str:
    """
    Загрузка видео с YouTube через yt-dlp (УНИВЕРСАЛЬНАЯ ВЕРСИЯ)
    
    Args:
        url: YouTube URL
        output_dir: Директория для сохранения
    
    Returns:
        Путь к скачанному файлу
    """
    logger.info(f"Downloading YouTube video: {url}")
    
    output_template = os.path.join(output_dir, '%(title).120s.%(id)s.%(ext)s')
    
    # ИСПРАВЛЕНО: убраны несовместимые флаги
    cmd = [
        'yt-dlp',
        '-f', 'bestvideo+bestaudio/best',
        '--merge-output-format', 'mp4',
        '--no-warnings',
        '--no-playlist',
        '--restrict-filenames',  # Убирает кириллицу, пробелы, спецсимволы
        '--windows-filenames',   # Совместимость с Windows
        '-o', output_template,
        url
    ]
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Явная кодировка UTF-8
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(output_dir_path),
            encoding='utf-8',
            errors='replace'
        )
        
        # Ищем все mp4 файлы и берём самый свежий
        mp4_files = list(output_dir_path.glob('*.mp4'))
        
        if not mp4_files:
            logger.error("No MP4 file found after download")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError("No MP4 file found after download")
        
        # Берём самый новый файл
        latest_file = max(mp4_files, key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Downloaded: {latest_file}")
        return str(latest_file)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"yt-dlp failed with return code {e.returncode}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        
        # Проверяем, может файл всё равно скачался
        mp4_files = list(output_dir_path.glob('*.mp4'))
        if mp4_files:
            latest_file = max(mp4_files, key=lambda p: p.stat().st_mtime)
            logger.warning(f"Using existing file: {latest_file}")
            return str(latest_file)
        
        raise RuntimeError(f"Failed to download video (code {e.returncode}): {e.stderr}")
        
    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        raise RuntimeError(f"Failed to download video: {e}")


def get_youtube_metadata(url: str) -> Dict[str, Any]:
    """
    Получить метаданные видео без скачивания
    
    Returns:
        dict с title, duration, thumbnail, etc.
    """
    try:
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--skip-download',
            '--no-warnings',
            url
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace'
        )
        
        if not result.stdout:
            raise RuntimeError("yt-dlp returned empty output")
        
        data = json.loads(result.stdout)
        
        return {
            'title': data.get('title', ''),
            'duration': data.get('duration', 0),
            'thumbnail': data.get('thumbnail', ''),
            'description': data.get('description', ''),
            'uploader': data.get('uploader', ''),
            'view_count': data.get('view_count', 0)
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get metadata: {e}")
        logger.error(f"stderr: {e.stderr}")
        return {}
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {}
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return {}


class YouTubeAuth:
    """Управление OAuth авторизацией YouTube"""
    
    def __init__(self, tokens_dir: str, clients_dir: str):
        self.tokens_dir = Path(tokens_dir)
        self.clients_dir = Path(clients_dir)
        self.tokens_dir.mkdir(parents=True, exist_ok=True)
        self.clients_dir.mkdir(parents=True, exist_ok=True)
    
    def add_account(
        self,
        client_secret_file: str,
        label: Optional[str] = None,
        port: int = 8080
    ) -> Dict[str, Any]:
        """
        Добавить аккаунт через OAuth
        
        Args:
            client_secret_file: Путь к client_secret.json
            label: Метка для аккаунта
            port: Порт для OAuth callback
        
        Returns:
            Информация об аккаунте
        """
        logger.info(f"Starting OAuth flow with {client_secret_file}")
        
        # Запуск OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file(
            client_secret_file,
            SCOPES
        )
        
        creds = flow.run_local_server(port=port)
        
        # Получение информации о канале
        youtube = build('youtube', 'v3', credentials=creds)
        
        try:
            response = youtube.channels().list(
                part='snippet,statistics',
                mine=True
            ).execute()
            
            if not response.get('items'):
                raise RuntimeError("No channel found for this account")
            
            channel = response['items'][0]
            channel_id = channel['id']
            channel_title = channel['snippet']['title']
            subscribers = channel['statistics'].get('subscriberCount', 0)
            
        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            channel_id = 'unknown'
            channel_title = label or 'Unknown Channel'
            subscribers = 0
        
        # Сохранение токена
        account_id = channel_id if channel_id != 'unknown' else f"acc_{os.urandom(4).hex()}"
        token_path = self.tokens_dir / f"{account_id}.json"
        
        with open(token_path, 'w', encoding='utf-8') as f:
            f.write(creds.to_json())
        
        logger.info(f"Account added: {channel_title} ({channel_id})")
        
        return {
            'id': account_id,
            'channel_id': channel_id,
            'channel_title': channel_title,
            'subscribers': subscribers,
            'label': label or channel_title,
            'token_path': str(token_path)
        }
    
    def load_credentials(self, account_id: str) -> Credentials:
        """Загрузить credentials для аккаунта"""
        token_path = self.tokens_dir / f"{account_id}.json"
        
        if not token_path.exists():
            raise FileNotFoundError(f"Token not found: {token_path}")
        
        with open(token_path, 'r', encoding='utf-8') as f:
            token_data = json.load(f)
        
        return Credentials.from_authorized_user_info(token_data, SCOPES)
    
    def get_channel_info(self, account_id: str) -> Dict[str, Any]:
        """Получить информацию о канале"""
        creds = self.load_credentials(account_id)
        youtube = build('youtube', 'v3', credentials=creds)
        
        response = youtube.channels().list(
            part='snippet,statistics',
            mine=True
        ).execute()
        
        if not response.get('items'):
            return {'ok': False, 'error': 'No channel found'}
        
        channel = response['items'][0]
        
        return {
            'ok': True,
            'channel_id': channel['id'],
            'title': channel['snippet']['title'],
            'description': channel['snippet']['description'],
            'subscriber_count': channel['statistics'].get('subscriberCount', 0),
            'video_count': channel['statistics'].get('videoCount', 0)
        }


class YouTubeUploader:
    """Загрузка видео на YouTube"""
    
    def __init__(self, auth: YouTubeAuth):
        self.auth = auth
    
    def upload_short(
        self,
        account_id: str,
        video_path: str,
        title: str,
        description: str = "",
        tags: list = None,
        privacy_status: str = "private",
        made_for_kids: bool = False,
        publish_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Загрузить Short на YouTube
        
        Args:
            account_id: ID аккаунта
            video_path: Путь к видео файлу
            title: Название (макс 100 символов)
            description: Описание
            tags: Теги
            privacy_status: private/unlisted/public
            made_for_kids: Для детей?
            publish_at: RFC3339 timestamp для отложенной публикации
        
        Returns:
            Результат загрузки с video_id
        """
        logger.info(f"Uploading video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Проверяем что #Shorts в описании
        if '#shorts' not in description.lower() and '#short' not in description.lower():
            description = description + "\n\n#Shorts"
        
        # Загружаем credentials
        creds = self.auth.load_credentials(account_id)
        youtube = build('youtube', 'v3', credentials=creds)
        
        # Подготовка метаданных
        body = {
            'snippet': {
                'title': title[:100],  # YouTube limit
                'description': description,
                'tags': tags or [],
                'categoryId': '22'  # People & Blogs
            },
            'status': {
                'privacyStatus': privacy_status,
                'selfDeclaredMadeForKids': made_for_kids
            }
        }
        
        # Отложенная публикация
        if publish_at:
            body['status']['publishAt'] = publish_at
            body['status']['privacyStatus'] = 'private'
        
        # Создаем Media upload
        media = MediaFileUpload(
            video_path,
            mimetype='video/mp4',
            resumable=True,
            chunksize=10 * 1024 * 1024  # 10MB chunks
        )
        
        # Инициализация запроса
        request = youtube.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media
        )
        
        # Загрузка с прогрессом
        response = None
        while response is None:
            status, response = request.next_chunk()
            
            if status:
                progress = int(status.progress() * 100)
                logger.info(f"Upload progress: {progress}%")
        
        video_id = response['id']
        logger.info(f"Video uploaded successfully: {video_id}")
        
        return {
            'ok': True,
            'video_id': video_id,
            'url': f"https://www.youtube.com/watch?v={video_id}",
            'response': response
        }
    
    def get_video_status(
        self,
        account_id: str,
        video_id: str
    ) -> Dict[str, Any]:
        """Проверить статус обработки видео"""
        creds = self.auth.load_credentials(account_id)
        youtube = build('youtube', 'v3', credentials=creds)
        
        response = youtube.videos().list(
            part='status,processingDetails',
            id=video_id
        ).execute()
        
        if not response.get('items'):
            return {'ok': False, 'error': 'Video not found'}
        
        video = response['items'][0]
        
        return {
            'ok': True,
            'video_id': video_id,
            'upload_status': video['status'].get('uploadStatus'),
            'privacy_status': video['status'].get('privacyStatus'),
            'processing_status': video.get('processingDetails', {}).get('processingStatus')
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("YouTube module ready") 

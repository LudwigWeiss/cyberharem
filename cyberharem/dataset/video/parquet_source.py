from typing import Dict, Optional
from pathlib import Path
from waifuc.source import LocalSource
from waifuc.action import CCIPAction

from ...parquet_download import ParquetDownloader

class ParquetVideoSource:
    def __init__(self, parquet_dir: str, download_dir: Optional[str] = None):
        self.parquet_dir = Path(parquet_dir)
        self.download_dir = Path(download_dir) if download_dir else self.parquet_dir / 'downloads'
        self.downloader = ParquetDownloader(
            dir_parquets=str(self.parquet_dir),
            dir_download=str(self.download_dir)
        )
    
    def create_source(self, frame_extract_fps: int = 1) -> LocalSource:
        """
        Download videos and extract frames to create a LocalSource
        """
        # Download all videos
        video_paths = self.downloader.download_all()
        
        # Create temporary directory for frames
        with TemporaryDirectory() as td:
            # Extract frames from videos (you'll need to implement this)
            frame_paths = self._extract_frames(
                video_paths, 
                output_dir=td,
                fps=frame_extract_fps
            )
            
            # Create LocalSource with extracted frames
            source = LocalSource(td, shuffle=True)
            
            # Add CCIP action for character detection
            source = source.attach(CCIPAction())
            
            return source
    
    def _extract_frames(self, video_paths: Dict[str, str], 
                       output_dir: str, fps: int) -> List[str]:
        """
        Extract frames from videos using ffmpeg
        Returns: List of paths to extracted frames
        """
        frame_paths = []
        for video_id, video_path in tqdm(video_paths.items(), desc="Extracting frames"):
            # Use ffmpeg to extract frames
            output_pattern = os.path.join(output_dir, f"{video_id}_%04d.jpg")
            os.system(f'ffmpeg -i "{video_path}" -vf fps={fps} "{output_pattern}"')
            
            # Collect frame paths
            frame_paths.extend(glob.glob(os.path.join(output_dir, f"{video_id}_*.jpg")))
        
        return frame_paths 
import os
import time
import requests
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import json
import csv
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class VideoMetadata:
    identifier: str
    url_link: str
    scene_start_time: Optional[str]
    scene_end_time: Optional[str]
    frame_number: Optional[float]
    key_frame_number: Optional[float]
    anime_tags: Optional[str]
    user_tags: Optional[str]
    text_description: Optional[str]
    aesthetic_score: Optional[float]
    dynamic_score: Optional[float]
    rating: Optional[str]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    taxonomy: Dict[str, str]

class ParquetDownloader:
    def __init__(self, dir_parquets: str, dir_download: str, dir_metadata: Optional[str] = None):
        self.dir_parquets = dir_parquets
        self.dir_download = dir_download
        self.dir_metadata = dir_metadata or os.path.join(os.path.dirname(dir_download), 'metadata')
        
        # Ensure directories exist
        try:
            os.makedirs(self.dir_parquets, exist_ok=True)
            os.makedirs(self.dir_download, exist_ok=True)
            os.makedirs(self.dir_metadata, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create directories: {e}")
            raise
    
    def extract_metadata(self, df_row) -> VideoMetadata:
        """Extract metadata from a single parquet row"""
        try:
            taxonomy = {
                key.replace('Taxonomy_', ''): str(df_row.get(key, ''))
                for key in df_row.keys()
                if key.startswith('Taxonomy_')
            }
            
            return VideoMetadata(
                identifier=str(df_row.get('identifier', '')),
                url_link=str(df_row.get('url_link', '')),
                scene_start_time=str(df_row.get('scene_start_time')) if df_row.get('scene_start_time') else None,
                scene_end_time=str(df_row.get('scene_end_time')) if df_row.get('scene_end_time') else None,
                frame_number=float(df_row.get('frame_number')) if df_row.get('frame_number') else None,
                key_frame_number=float(df_row.get('key_frame_number')) if df_row.get('key_frame_number') else None,
                anime_tags=str(df_row.get('anime_tags')) if df_row.get('anime_tags') else None,
                user_tags=str(df_row.get('user_tags')) if df_row.get('user_tags') else None,
                text_description=str(df_row.get('text_description')) if df_row.get('text_description') else None,
                aesthetic_score=float(df_row.get('aesthetic_score')) if df_row.get('aesthetic_score') else None,
                dynamic_score=float(df_row.get('dynamic_score')) if df_row.get('dynamic_score') else None,
                rating=str(df_row.get('rating')) if df_row.get('rating') else None,
                width=int(df_row.get('width')) if df_row.get('width') else None,
                height=int(df_row.get('height')) if df_row.get('height') else None,
                fps=float(df_row.get('fps')) if df_row.get('fps') else None,
                taxonomy=taxonomy
            )
        except Exception as e:
            logging.error(f"Error extracting metadata: {e}")
            raise
    
    def save_single_metadata(self, video_id: str, metadata: VideoMetadata) -> Tuple[str, str]:
        """
        Save metadata for a single video
        Returns: Tuple of (json_path, csv_path)
        """
        # Use same name as video file but with different extensions
        base_name = video_id
        json_path = os.path.join(self.dir_metadata, f"{base_name}.json")
        csv_path = os.path.join(self.dir_metadata, f"{base_name}.csv")
        
        # Save as JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                metadata_dict = {
                    'identifier': metadata.identifier,
                    'url_link': metadata.url_link,
                    'scene_start_time': metadata.scene_start_time,
                    'scene_end_time': metadata.scene_end_time,
                    'frame_number': metadata.frame_number,
                    'key_frame_number': metadata.key_frame_number,
                    'anime_tags': metadata.anime_tags,
                    'user_tags': metadata.user_tags,
                    'text_description': metadata.text_description,
                    'aesthetic_score': metadata.aesthetic_score,
                    'dynamic_score': metadata.dynamic_score,
                    'rating': metadata.rating,
                    'width': metadata.width,
                    'height': metadata.height,
                    'fps': metadata.fps,
                    'taxonomy': metadata.taxonomy
                }
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved JSON metadata to {json_path}")
        except Exception as e:
            logging.error(f"Failed to save JSON metadata for {video_id}: {e}")
            
        # Save as CSV
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                header = ['field', 'value']
                writer.writerow(header)
                
                # Write basic metadata
                for field, value in metadata_dict.items():
                    if field != 'taxonomy':
                        writer.writerow([field, value])
                
                # Write taxonomy data
                for tax_key, tax_value in metadata.taxonomy.items():
                    writer.writerow([f'taxonomy_{tax_key}', tax_value])
                    
            logging.info(f"Saved CSV metadata to {csv_path}")
        except Exception as e:
            logging.error(f"Failed to save CSV metadata for {video_id}: {e}")
            
        return json_path, csv_path

    def download_single_parquet(self, parquet_file: str) -> Tuple[Dict[str, str], Dict[str, VideoMetadata]]:
        """Download videos from a single parquet file"""
        try:
            parquet_path = os.path.join(self.dir_parquets, parquet_file)
            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
            df = pd.read_parquet(parquet_path)
            logging.info(f"Processing {parquet_file} -->")
            
            downloaded_files = {}
            metadata_map = {}
            
            for _, row in df.iterrows():
                if not row.get('url_link') or not row.get('identifier'):
                    continue
                    
                try:
                    id_video = row['identifier'].split(':')[0]
                    file_extension = urlparse(row['url_link']).path.split(".")[-1]
                    
                    file_download = os.path.join(self.dir_download, f"{id_video}.{file_extension}")
                    
                    if not os.path.exists(file_download):
                        response = requests.get(row['url_link'], stream=True)
                        response.raise_for_status()
                        
                        with open(file_download, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        time.sleep(0.35)
                    
                    # Extract and save metadata immediately after download
                    metadata = self.extract_metadata(row)
                    self.save_single_metadata(id_video, metadata)
                    
                    downloaded_files[id_video] = file_download
                    metadata_map[id_video] = metadata
                    
                except Exception as e:
                    logging.error(f"Error processing row {row.get('identifier', 'unknown')}: {e}")
                    continue
            
            return downloaded_files, metadata_map
            
        except Exception as e:
            logging.error(f"Error processing parquet file {parquet_file}: {e}")
            raise

    def download_all(self) -> Tuple[Dict[str, str], Dict[str, VideoMetadata]]:
        """Download all videos from all parquet files"""
        try:
            parquets = [p for p in os.listdir(self.dir_parquets) if p.endswith('.parquet')]
            if not parquets:
                logging.warning("No parquet files found in directory")
                return {}, {}
            
            all_downloads = {}
            all_metadata = {}
            
            for parquet in parquets:
                try:
                    downloads, metadata = self.download_single_parquet(parquet)
                    all_downloads.update(downloads)
                    all_metadata.update(metadata)
                except Exception as e:
                    logging.error(f"Error processing {parquet}: {e}")
                    continue
            
            return all_downloads, all_metadata
            
        except Exception as e:
            logging.error(f"Error in download_all: {e}")
            raise

def main():
    try:
        cwd = Path(__file__).parent
        parser = argparse.ArgumentParser()
        parser.add_argument('--dir_parquets', type=str, default=str(cwd/'parquet'), 
                          help='Path to parquet files')
        parser.add_argument('--dir_download', type=str, default=str(cwd/'download'), 
                          help='Path to download the videos')
        parser.add_argument('--dir_metadata', type=str, default=None,
                          help='Path to save metadata files (default: ./metadata)')
        args = parser.parse_args()
        
        downloader = ParquetDownloader(args.dir_parquets, args.dir_download, args.dir_metadata)
        downloads, metadata = downloader.download_all()
        
        # Print summary
        print(f"\nProcessed {len(downloads)} videos")
        for id_video, file_download in tqdm(downloads.items(), desc="Downloaded files"):
            try:
                meta = metadata[id_video]
                print(f"\nVideo {id_video}:")
                print(f"  Video: {file_download}")
                print(f"  Metadata: {os.path.join(downloader.dir_metadata, f'{id_video}.json')}")
                print(f"  Resolution: {meta.width}x{meta.height}")
                print(f"  FPS: {meta.fps}")
                print(f"  Rating: {meta.rating}")
                print(f"  Aesthetic Score: {meta.aesthetic_score}")
                if meta.anime_tags:
                    print(f"  Anime Tags: {meta.anime_tags}")
                if meta.taxonomy:
                    print("  Taxonomy:")
                    for key, value in meta.taxonomy.items():
                        if value:
                            print(f"    {key}: {value}")
            except Exception as e:
                logging.error(f"Error printing metadata for {id_video}: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise

if __name__=="__main__":
    main()
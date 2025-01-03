import subprocess
import os
from pathlib import Path
from typing import List, Optional, Dict
import logging
import pandas as pd
import math

class TPURunner:
    def __init__(self, working_dir: str = None):
        self.working_dir = working_dir or os.getcwd()
        
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command across all TPUs using podrun"""
        full_command = f"python3.10 podrun -i -- {command}"
        logging.info(f"Running command: {full_command}")
        
        result = subprocess.run(
            full_command,
            shell=True,
            check=check,
            text=True,
            capture_output=True
        )
        return result
    
    def get_tpu_count(self) -> int:
        """Get number of available TPUs"""
        result = self.run_command("echo $TPU_NUM_DEVICES", check=False)
        try:
            return int(result.stdout.strip())
        except:
            return 1  # Default to 1 if can't determine
    
    def distribute_parquets(self, parquet_dir: str):
        """Distribute parquet files across TPUs"""
        # Get TPU count
        num_tpus = self.get_tpu_count()
        
        # Create directories on each TPU
        self.run_command("mkdir -p data/parquet data/download data/metadata")
        
        # List all parquet files
        parquet_files = list(Path(parquet_dir).glob("*.parquet"))
        
        # Calculate files per TPU
        files_per_tpu = math.ceil(len(parquet_files) / num_tpus)
        
        # Create distribution command
        commands = []
        for tpu_idx in range(num_tpus):
            start_idx = tpu_idx * files_per_tpu
            end_idx = min((tpu_idx + 1) * files_per_tpu, len(parquet_files))
            
            # Get files for this TPU
            tpu_files = parquet_files[start_idx:end_idx]
            if not tpu_files:
                continue
                
            # Create command to copy specific files
            file_list = " ".join(str(f) for f in tpu_files)
            commands.append(f"""
if [ "$TPU_PROCESS_INDEX" = "{tpu_idx}" ]; then
    cp {file_list} data/parquet/
fi
""")
        
        # Run distribution command
        distribution_command = "\n".join(commands)
        self.run_command(distribution_command)
        logging.info(f"Distributed parquet files across {num_tpus} TPUs")
    
    def process_parquets(self, 
                        source_repo: str,
                        ch_ids: List[int],
                        name: str,
                        limit: Optional[int] = 1000) -> subprocess.CompletedProcess:
        """Process parquets across TPUs"""
        command = f"""python -c "
import os
from pathlib import Path
from cyberharem.dataset.video.parquet_source import ParquetVideoSource
from cyberharem.dataset.video.crawler import crawl_base_to_huggingface
from parquet_download import ParquetDownloader

# Get TPU index
tpu_idx = int(os.environ.get('TPU_PROCESS_INDEX', 0))

# Initialize downloader for this TPU
downloader = ParquetDownloader(
    dir_parquets='data/parquet',
    dir_download=f'data/download/tpu_{tpu_idx}',
    dir_metadata=f'data/metadata/tpu_{tpu_idx}'
)

# Process files assigned to this TPU
downloads, metadata = downloader.download_all()

# Create source and process
source = ParquetVideoSource(
    parquet_dir='data/parquet',
    download_dir=f'data/download/tpu_{tpu_idx}'
)

# Run crawler
crawl_base_to_huggingface(
    source_repository='{source_repo}',
    ch_id={ch_ids},
    name='{name}',
    parquet_source=source,
    limit={limit}
)
"
"""
        return self.run_command(command)
    
    def collect_results(self) -> Dict:
        """Collect results from all TPUs"""
        command = """python -c "
import os
import json
from pathlib import Path
from glob import glob

# Get all metadata directories
metadata_dirs = glob('data/metadata/tpu_*')
results = {}

for meta_dir in metadata_dirs:
    # Get TPU index from directory name
    tpu_idx = meta_dir.split('_')[-1]
    
    # Collect all JSON files
    json_files = glob(f'{meta_dir}/*.json')
    
    results[f'tpu_{tpu_idx}'] = {
        'files_processed': len(json_files),
        'metadata_dir': meta_dir,
        'download_dir': f'data/download/tpu_{tpu_idx}'
    }

print(json.dumps(results))
"
"""
        result = self.run_command(command)
        try:
            return json.loads(result.stdout)
        except:
            return {} 
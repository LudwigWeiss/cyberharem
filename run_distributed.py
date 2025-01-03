import logging
from pathlib import Path
from cyberharem.utils.tpu_runner import TPURunner

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize runner
    runner = TPURunner()
    
    # Get TPU count
    num_tpus = runner.get_tpu_count()
    logging.info(f"Running with {num_tpus} TPUs")
    
    # Distribute parquet files
    runner.distribute_parquets("data/parquet")
    
    # Run processing
    result = runner.process_parquets(
        source_repo="your/source/repo",
        ch_ids=[1, 2, 3],
        name="character_name",
        limit=1000
    )
    
    # Print processing results
    print("Processing Results:")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        logging.error("Processing failed!")
        return
    
    # Collect and print results
    results = runner.collect_results()
    print("\nTPU Processing Summary:")
    for tpu_id, data in results.items():
        print(f"\n{tpu_id}:")
        print(f"  Files Processed: {data['files_processed']}")
        print(f"  Metadata Directory: {data['metadata_dir']}")
        print(f"  Download Directory: {data['download_dir']}")
    
    logging.info("Processing completed successfully!")

if __name__ == "__main__":
    main() 
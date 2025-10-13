"""
Bus Ridership Prediction - Complete ML Pipeline
Author: Øyvind S.
Date: 2025-09-21

This script orchestrates the complete machine learning pipeline
for bus ridership prediction in Trondheim.
"""
# Import libraries
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BusRidershipPipeline:
    """Complete pipeline for bus ridership prediction model training"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize pipeline with project structure"""
        self.project_root = project_root or Path.cwd()
        self.setup_paths()
        self.pipeline_steps = self.define_pipeline()
        
    def setup_paths(self):
        """Define all project paths"""
        # Main directories
        self.data_dir = self.project_root / 'Data'
        self.atb_data_dir = self.data_dir / 'atb_data' / '2024'
        self.weather_dir = self.data_dir / 'weather'
        self.processed_dir = self.data_dir / 'processed'
        self.src_dir = self.project_root / 'src'
        
        # Create directories if they don't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.weather_dir.mkdir(parents=True, exist_ok=True)
        
        # Expected files for validation
        self.expected_files = {
            'monthly_csvs': [self.atb_data_dir / f'{i:02d} 2024.csv' for i in range(1, 13)],
            'events_config': self.project_root / 'events_config.json',
            'weather_data': self.weather_dir / 'weather_data_2024.csv',
            'event_features': self.processed_dir / 'event_features_2024.csv',
            'master_dataset_pkl': self.processed_dir / 'master_feature_dataset_2024.pkl',
            'advanced_features_pkl': self.processed_dir / 'master_feature_dataset_2024_with_advanced_features.pkl',
            'final_training_set': self.processed_dir / 'final_training_set_advanced.pkl',
            'final_model': self.project_root / 'models' / 'final_bus_ridership_model.joblib'
        }
        
    def define_pipeline(self) -> List[Dict]:
        """Define the complete pipeline steps"""
        return [
            #{
            #    'name': 'Weather Data Collection',
            #    'script': 'retrieve_hourly_weather.py',
            #    'description': 'Fetch meteorological data from frost.met.no API',
            #    'input_files': [],
            #    'output_files': ['weather_data'],
            #    'required_inputs': ['events_config']  # Just needs to exist
            #},
            {
                'name': 'Event Feature Building',
                'script': 'event_feature_builder.py',
                'description': 'Create binary event indicators from configuration',
                'input_files': ['events_config'],
                'output_files': ['event_features'],
                'required_inputs': ['events_config']
            },
            {
                'name': 'Master Dataset Building',
                'script': 'master_dataset_builder.py',
                'description': 'Integrate ridership, weather, and event data',
                'input_files': ['monthly_csvs', 'weather_data', 'event_features'],
                'output_files': ['master_dataset_pkl'],
                'required_inputs': ['weather_data', 'event_features']
            },
            {
                'name': 'Feature Engineering',
                'script': 'feature_engineered_RF.py',
                'description': 'Create advanced features (lag, rolling stats)',
                'input_files': ['master_dataset_pkl'],
                'output_files': ['advanced_features_pkl'],
                'required_inputs': ['master_dataset_pkl']
            },
            {
                'name': 'Hyperparameter Tuning',
                'script': 'hyperparameter_tuning_RF.py',
                'description': 'Optimize Random Forest parameters',
                'input_files': ['advanced_features_pkl'],
                'output_files': ['final_model'],
                'required_inputs': ['advanced_features_pkl']
            },
            {
                'name': 'Final Model Training',
                'script': 'train_final_model.py',
                'description': 'Train final optimized model',
                'input_files': ['advanced_features_pkl'],
                'output_files': ['final_model', 'final_training_set'],
                'required_inputs': ['advanced_features_pkl']
            },
            {
                'name': 'Business Impact Analysis',
                'script': 'business_impact_analysis.py',
                'description': 'Generate business value assessment',
                'input_files': ['final_training_set', 'final_model'],
                'output_files': [],
                'required_inputs': ['final_training_set', 'final_model']
            }
        ]
    
    def validate_inputs(self, step: Dict) -> bool:
        """Validate that required input files exist for a step"""
        missing_files = []
        
        for input_key in step['required_inputs']:
            if input_key == 'monthly_csvs':
                missing_monthly = [f for f in self.expected_files['monthly_csvs'] if not f.exists()]
                if missing_monthly:
                    missing_files.extend(missing_monthly)
            else:
                file_path = self.expected_files.get(input_key)
                if file_path and not file_path.exists():
                    missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing required files for {step['name']}:")
            for file in missing_files:
                logger.error(f"  - {file}")
            return False
        
        return True
    
    def run_script(self, script_name: str) -> bool:
        """Execute a Python script and return success status"""
        script_path = self.src_dir / script_name
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        process = None
        try:
            logger.info(f"Executing: {script_name}")
            if sys.platform == "win32":
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=self.project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

            stdout, stderr = process.communicate(timeout=7200)  # 2 hour timeout

            if process.returncode == 0:
                logger.info(f"SUCCESS: {script_name} completed successfully")
                if stdout:
                    logger.debug(f"Output: {stdout}")
                return True
            else:
                logger.error(f"FAILED: {script_name} failed with return code {process.returncode}")
                if stderr:
                    logger.error(f"Error: {stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"TIMEOUT: {script_name} timed out after 1 hour")
            if process:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)])
                else:
                    process.kill()
            return False
        except Exception as e:
            logger.error(f"ERROR: {script_name} failed with exception: {str(e)}")
            return False
    
    def validate_outputs(self, step: Dict) -> bool:
        """Validate that expected output files were created"""
        missing_outputs = []
        
        for output_key in step['output_files']:
            file_path = self.expected_files.get(output_key)
            if file_path and not file_path.exists():
                missing_outputs.append(file_path)
        
        if missing_outputs:
            logger.error(f"Missing expected outputs for {step['name']}:")
            for file in missing_outputs:
                logger.error(f"  - {file}")
            return False
        
        return True
    
    def run_step(self, step: Dict) -> bool:
        """Execute a single pipeline step with validation"""
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP: {step['name']}")
        logger.info(f"Description: {step['description']}")
        logger.info(f"{'='*60}")
        
        # Validate inputs
        if not self.validate_inputs(step):
            logger.error(f"Input validation failed for {step['name']}")
            return False
        
        # Run the script
        success = self.run_script(step['script'])
        
        if not success:
            return False
        
        # Validate outputs
        if not self.validate_outputs(step):
            logger.error(f"Output validation failed for {step['name']}")
            return False
        
        logger.info(f"SUCCESS: {step['name']} completed successfully")
        return True
    
    def run_full_pipeline(self, start_from_step: int = 0, stop_at_step: Optional[int] = None):
        """Execute the complete pipeline"""
        logger.info("="*80)
        logger.info("STARTING BUS RIDERSHIP PREDICTION PIPELINE")
        logger.info(f"Project Root: {self.project_root}")
        logger.info("="*80)
        
        start_time = time.time()
        successful_steps = 0
        total_steps = len(self.pipeline_steps)
        
        if stop_at_step is None:
            stop_at_step = total_steps
        
        for i, step in enumerate(self.pipeline_steps[start_from_step:stop_at_step], start_from_step):
            logger.info(f"\nPipeline Progress: Step {i+1}/{total_steps}")
            
            if self.run_step(step):
                successful_steps += 1
            else:
                logger.error(f"Pipeline failed at step {i+1}: {step['name']}")
                logger.error("Stopping pipeline execution")
                break
        
        # Final summary
        total_time = time.time() - start_time
        logger.info("\n" + "="*80)
        if successful_steps == (stop_at_step - start_from_step):
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        else:
            logger.error(f"ERROR: PIPELINE FAILED after {successful_steps} steps")
        
        logger.info(f"Execution time: {total_time:.2f} seconds")
        logger.info(f"Steps completed: {successful_steps}/{stop_at_step - start_from_step}")
        logger.info("="*80)
        
        return successful_steps == (stop_at_step - start_from_step)
    
    def check_status(self):
        """Check which files exist and pipeline readiness"""
        logger.info("PIPELINE STATUS CHECK")
        logger.info("="*40)
        
        for name, file_path in self.expected_files.items():
            if isinstance(file_path, list):  # monthly_csvs
                existing = sum(1 for f in file_path if f.exists())
                total = len(file_path)
                status = "Existing" if existing == total else "Missing"
                logger.info(f"{status} {name}: {existing}/{total} files")
            else:
                status = "Existing" if file_path.exists() else "Missing"
                logger.info(f"{status} {name}: {file_path}")
        
        logger.info("="*40)

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Bus Ridership ML Pipeline')
    parser.add_argument('--check-status', action='store_true', help='Check pipeline status')
    parser.add_argument('--start-from', type=int, default=0, help='Start from step number (0-based)')
    parser.add_argument('--stop-at', type=int, help='Stop at step number (0-based)')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    project_root = Path(args.project_root) if args.project_root else Path.cwd()
    pipeline = BusRidershipPipeline(project_root)
    
    try:
        if args.check_status:
            pipeline.check_status()
        else:
            pipeline.run_full_pipeline(
                start_from_step=args.start_from,
                stop_at_step=args.stop_at
            )
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
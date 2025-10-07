"""
Manages job lifecycle and tracks progress of ebook-to-audio conversions.
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages the lifecycle of conversion jobs.
    """
    
    def __init__(self):
        self.jobs_dir = Path("outputs/jobs")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.active_jobs: Dict[str, Dict] = {}
    
    # === Job Creation Methods ===
    async def create_job(self, ebook_file: str, selected_models: List[str], parameters: Dict, output_format: str) -> str:
        """
        Create a new conversion job.
        
        Args:
            ebook_file: Path to the ebook file
            selected_models: List of selected TTS models
            parameters: Conversion parameters
            output_format: Output audio format (mp3/wav)
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job_info = {
            "id": job_id,
            "ebook_file": str(ebook_file),
            "selected_models": selected_models,
            "parameters": parameters,
            "output_format": output_format,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "chapters": [],
            "output_files": [],
            "processed_fragments": {},
            "failed_fragments": {},
            "max_retries": 3
        }
        
        self.active_jobs[job_id] = job_info
        self._save_job(job_id, job_info)
        
        logger.info(f"Created job {job_id}")
        return job_id
    
    # === Job Status Update Methods ===
    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and additional info."""
        if job_id not in self.active_jobs:
            return
        
        self.active_jobs[job_id]["status"] = status
        self.active_jobs[job_id]["updated_at"] = datetime.now().isoformat()
        
        for key, value in kwargs.items():
            self.active_jobs[job_id][key] = value
        
        self._save_job(job_id, self.active_jobs[job_id])
    
    async def update_progress(self, job_id: str, progress: float):
        """Update job progress."""
        if job_id not in self.active_jobs:
            return
        
        self.active_jobs[job_id]["progress"] = progress
        self.active_jobs[job_id]["updated_at"] = datetime.now().isoformat()
        self._save_job(job_id, self.active_jobs[job_id])
    
    async def complete_job(self, job_id: str, output_files: List[str]):
        """Mark job as completed."""
        if job_id not in self.active_jobs:
            return
        
        self.active_jobs[job_id]["status"] = "completed"
        self.active_jobs[job_id]["output_files"] = output_files
        self.active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        self.active_jobs[job_id]["updated_at"] = datetime.now().isoformat()
        self._save_job(job_id, self.active_jobs[job_id])
    
    async def fail_job(self, job_id: str, error_message: str):
        """Mark job as failed."""
        if job_id not in self.active_jobs:
            return
        
        self.active_jobs[job_id]["status"] = "failed"
        self.active_jobs[job_id]["error"] = error_message
        self.active_jobs[job_id]["failed_at"] = datetime.now().isoformat()
        self.active_jobs[job_id]["updated_at"] = datetime.now().isoformat()
        self._save_job(job_id, self.active_jobs[job_id])
    
    # === Chapter Management Methods ===
    def add_chapter(self, job_id: str, chapter_info: Dict):
        """Add chapter information to job."""
        if job_id not in self.active_jobs:
            return
        
        self.active_jobs[job_id]["chapters"].append(chapter_info)
        self._save_job(job_id, self.active_jobs[job_id])
    
    async def update_job_chapters(self, job_id: str, chapters: List[Dict]):
        """Update job with chapter information."""
        if job_id not in self.active_jobs:
            return

        for chapter in chapters:
            if "text" in chapter:
                text = chapter["text"]
                fragments = []
                for i in range(0, len(text), 200):
                    fragments.append(text[i:i+200])
                chapter["fragments"] = fragments
        
        self.active_jobs[job_id]["chapters"] = chapters
        self.active_jobs[job_id]["updated_at"] = datetime.now().isoformat()
        self._save_job(job_id, self.active_jobs[job_id])
    
    def add_output_file(self, job_id: str, file_path: str):
        """Add output file to job."""
        if job_id not in self.active_jobs:
            return
        
        if file_path not in self.active_jobs[job_id]["output_files"]:
            self.active_jobs[job_id]["output_files"].append(file_path)
            self._save_job(job_id, self.active_jobs[job_id])
    
    # === Job Query Methods ===
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job information."""
        return self.active_jobs.get(job_id)
    
    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        job = self.get_job(job_id)
        return job["status"] if job else "unknown"
    
    def get_output_files(self, job_id: str) -> List[str]:
        """Get list of output files for a job."""
        job = self.get_job(job_id)
        return job["output_files"] if job else []
    
    def get_job_progress(self, job_id: str) -> Dict:
        """Get job progress information."""
        job = self.get_job(job_id)
        if not job:
            return {"progress": 0, "message": "Job not found"}
        
        chapters = job.get("chapters", [])
        completed_chapters = [c for c in chapters if c.get("status") == "completed"]
        
        progress = 0
        if chapters:
            progress = len(completed_chapters) / len(chapters)
        
        return {
            "progress": progress,
            "completed": len(completed_chapters),
            "total": len(chapters),
            "status": job["status"],
            "message": f"Processing chapter {len(completed_chapters) + 1}/{len(chapters)}"
        }
    
    # === Job Persistence Methods ===
    def _save_job(self, job_id: str, job_info: Dict):
        """Save job information to file."""
        job_file = self.jobs_dir / job_id / "job_info.json"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(job_file, 'w', encoding='utf-8') as f:
            json.dump(job_info, f, indent=2, ensure_ascii=False)
    
    def load_job(self, job_id: str) -> Optional[Dict]:
        """Load job information from file."""
        job_file = self.jobs_dir / job_id / "job_info.json"
        
        if not job_file.exists():
            return None
        
        try:
            with open(job_file, 'r', encoding='utf-8') as f:
                job_info = json.load(f)
                self.active_jobs[job_id] = job_info
                return job_info
        except Exception as e:
            logger.error(f"Error loading job {job_id}: {e}")
            return None
    
    # === Utility Methods ===
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old job files."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                try:
                    mtime = job_dir.stat().st_mtime
                    if mtime < cutoff_time:
                        import shutil
                        shutil.rmtree(job_dir)
                        logger.info(f"Cleaned up old job: {job_dir.name}")
                except Exception as e:
                    logger.error(f"Error cleaning up job {job_dir.name}: {e}")
    
    def get_job_directory(self, job_id: str) -> Path:
        """Get the directory path for a job."""
        return self.jobs_dir / job_id

    def get_checkpoint_info(self, job_id: str) -> Dict:
        """Get checkpoint information for resumable jobs."""
        job = self.get_job(job_id)
        if not job:
            return {}

        return {
            "processed_fragments": job.get("processed_fragments", {}),
            "failed_fragments": job.get("failed_fragments", {}),
            "max_retries": job.get("max_retries", 3)
        }
    
    def update_fragment_status(self, job_id: str, chapter_idx: int, fragment_idx: int, status: str):
        """Update the status of a specific fragment."""
        if job_id not in self.active_jobs:
            return
        
        job = self.active_jobs[job_id]

        if "processed_fragments" not in job:
            job["processed_fragments"] = {}

        if "failed_fragments" not in job:
            job["failed_fragments"] = {}
        
        if status == "completed":
            if str(chapter_idx) not in job["processed_fragments"]:
                job["processed_fragments"][str(chapter_idx)] = []
            if fragment_idx not in job["processed_fragments"][str(chapter_idx)]:
                job["processed_fragments"][str(chapter_idx)].append(fragment_idx)
                
            if (str(chapter_idx) in job["failed_fragments"] and 
                str(fragment_idx) in job["failed_fragments"][str(chapter_idx)]):
                del job["failed_fragments"][str(chapter_idx)][str(fragment_idx)]
                if not job["failed_fragments"][str(chapter_idx)]:
                    del job["failed_fragments"][str(chapter_idx)]
                    
        elif status == "failed":
            if str(chapter_idx) not in job["failed_fragments"]:
                job["failed_fragments"][str(chapter_idx)] = {}
            if str(fragment_idx) not in job["failed_fragments"][str(chapter_idx)]:
                job["failed_fragments"][str(chapter_idx)][str(fragment_idx)] = 0
            job["failed_fragments"][str(chapter_idx)][str(fragment_idx)] += 1

            if (str(chapter_idx) in job["processed_fragments"] and 
                fragment_idx in job["processed_fragments"][str(chapter_idx)]):
                job["processed_fragments"][str(chapter_idx)].remove(fragment_idx)
                if not job["processed_fragments"][str(chapter_idx)]:
                    del job["processed_fragments"][str(chapter_idx)]
        
        self._save_job(job_id, job)
    
    def get_next_unprocessed_fragment(self, job_id: str) -> Optional[Tuple[int, int]]:
        """Get the next unprocessed fragment for a job."""
        job = self.get_job(job_id)
        if not job:
            return None
        
        chapters = job.get("chapters", [])
        processed_fragments = job.get("processed_fragments", {})
        failed_fragments = job.get("failed_fragments", {})
        max_retries = job.get("max_retries", 3)
 
        for chapter_idx, chapter in enumerate(chapters):
            fragments = chapter.get("fragments", [])
            for fragment_idx in range(len(fragments)):
                if (str(chapter_idx) in processed_fragments and 
                    fragment_idx in processed_fragments[str(chapter_idx)]):
                    continue

                if (str(chapter_idx) in failed_fragments and 
                    str(fragment_idx) in failed_fragments[str(chapter_idx)] and
                    failed_fragments[str(chapter_idx)][str(fragment_idx)] >= max_retries):
                    continue

                return (chapter_idx, fragment_idx)

        return None
    
    def is_fragment_processed(self, job_id: str, chapter_idx: int, fragment_idx: int) -> bool:
        """Check if a specific fragment has been processed."""
        job = self.get_job(job_id)
        if not job:
            return False
        
        processed_fragments = job.get("processed_fragments", {})
        return (str(chapter_idx) in processed_fragments and 
                fragment_idx in processed_fragments[str(chapter_idx)])
    
    def get_fragment_retry_count(self, job_id: str, chapter_idx: int, fragment_idx: int) -> int:
        """Get the number of times a fragment has been retried."""
        job = self.get_job(job_id)
        if not job:
            return 0
        
        failed_fragments = job.get("failed_fragments", {})
        if (str(chapter_idx) in failed_fragments and 
            str(fragment_idx) in failed_fragments[str(chapter_idx)]):
            return failed_fragments[str(chapter_idx)][str(fragment_idx)]
        return 0
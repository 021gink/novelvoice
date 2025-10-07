"""
Handles audio processing: merging fragments, normalization, and format conversion.
"""
import logging
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import wave
import audioop

from pydub import AudioSegment
from pydub.effects import normalize
import mutagen
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TIT2, TPE1, TALB
from scipy.signal import butter, filtfilt

from config import AUDIO_SETTINGS
from utils.decorators import performance_monitor, memory_monitor 

logger = logging.getLogger(__name__)


class AudioProcessor:

    def __init__(self):
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        self.target_sample_rate = 44100 
        self.target_bit_depth = 16      
        self.target_lufs = -23.0       

    @performance_monitor(threshold_ms=2000)
    @memory_monitor
    async def merge_chapter_audio(
        self,
        audio_files: List[str],
        chapter_title: str,
        job_id: str,
        output_format: str = None
    ) -> str:
        if not audio_files:
            raise ValueError("No audio files provided")

        job_dir = self.output_dir / "jobs" / job_id

        safe_chapter_title = "".join(c for c in chapter_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_chapter_title = safe_chapter_title.replace(' ', '_').lower()

        if not safe_chapter_title:
            safe_chapter_title = "untitled_chapter"
        chapter_dir = job_dir / f"chapter-{safe_chapter_title}"
        chapter_dir.mkdir(parents=True, exist_ok=True)

        processed_segments = []
        for i, audio_file in enumerate(audio_files):
            try:
                segment = AudioSegment.from_file(audio_file)
                segment = self._standardize_format(segment)
                segment = self._apply_noise_reduction(segment)
                segment = self._apply_volume_normalization(segment)
                processed_segments.append(segment)
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.error(f"Error processing audio file {audio_file}: {e}")
                continue

        if not processed_segments:
            raise ValueError("No audio files could be processed")

        combined = self._optimized_concatenate(processed_segments)
        combined = self._finalize_audio_quality(combined)

        if output_format is None:
            output_format = AUDIO_SETTINGS.get("default_output_format", "mp3")

        format_handlers = {
            "wav": {
                "extension": "wav",
                "export_params": lambda: [
                    "-ar", str(self.target_sample_rate), "-ac", "2", "-sample_fmt", "s16"
                ]
            },
            "mp3": {
                "extension": "mp3",
                "export_params": lambda: [
                    "-metadata", f"title={chapter_title}",
                    "-ar", str(self.target_sample_rate), "-ac", "2",
                    "-q:a", "0"
                ],
                "bitrate": AUDIO_SETTINGS["mp3_bitrate"]
            }
        }

        handler = format_handlers.get(output_format, format_handlers["mp3"])
        extension = handler["extension"]
        output_path = chapter_dir / f"{chapter_title}.{extension}"

        if "bitrate" in handler:
            combined.export(str(output_path), format=output_format,
                            bitrate=handler["bitrate"],
                            parameters=handler["export_params"]())
        else:
            combined.export(str(output_path), format=output_format,
                            parameters=handler["export_params"]())

        if output_format == "mp3":
            self._add_metadata(str(output_path), chapter_title)

        logger.info(f"Chapter audio optimized and saved: {output_path} (format: {output_format})")
        return str(output_path)

    def _add_metadata(self, file_path: str, title: str):
        try:
            audio = MP3(file_path, ID3=ID3)
            audio["TIT2"] = TIT2(encoding=3, text=title)
            audio["TPE1"] = TPE1(encoding=3, text="TTS Generator")
            audio["TALB"] = TALB(encoding=3, text="Generated Audiobook")
            audio.save()
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(f"Could not add metadata to {file_path}: {e}")

    @performance_monitor(threshold_ms=5000)
    @memory_monitor
    async def create_final_audiobook(
        self,
        chapter_files: List[str],
        book_title: str,
        job_id: str,
        output_format: str = None
    ) -> str:
        if not chapter_files:
            raise ValueError("No chapter files provided")

        if output_format is None:
            output_format = AUDIO_SETTINGS.get("default_output_format", "mp3")

        job_dir = self.output_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        format_handlers = {
            "wav": {
                "extension": "wav",
                "export_params": lambda: [
                    "-ar", str(self.target_sample_rate), "-ac", "2", "-sample_fmt", "s16"
                ]
            },
            "mp3": {
                "extension": "mp3",
                "export_params": lambda: [
                    "-metadata", f"title={book_title}",
                    "-ar", str(self.target_sample_rate), "-ac", "2",
                    "-q:a", "0"
                ],
                "bitrate": AUDIO_SETTINGS["mp3_bitrate"]
            }
        }

        handler = format_handlers.get(output_format, format_handlers["mp3"])
        extension = handler["extension"]
        output_path = job_dir / f"{book_title}.{extension}"

        combined = AudioSegment.empty()
        for chapter_file in chapter_files:
            segment = AudioSegment.from_file(chapter_file)
            combined += segment

        if "bitrate" in handler:
            combined.export(str(output_path), format=output_format,
                            bitrate=handler["bitrate"],
                            parameters=handler["export_params"]())
        else:
            combined.export(str(output_path), format=output_format,
                            parameters=handler["export_params"]())

        logger.info(f"Final audiobook created: {output_path} (format: {output_format})")
        return str(output_path)
   
    
    def get_audio_duration(self, file_path: str) -> float:
        try:
            audio = AudioSegment.from_file(file_path)
            return len(audio) / 1000.0  
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.error(f"Error getting duration for {file_path}: {e}")
            return 0.0

    def _standardize_format(self, audio: AudioSegment) -> AudioSegment:
        if audio.frame_rate != self.target_sample_rate:
            audio = audio.set_frame_rate(self.target_sample_rate)

        if audio.sample_width != 2:
            audio = audio.set_sample_width(2)
        
        return audio

    def _apply_noise_reduction(self, audio: AudioSegment) -> AudioSegment:
        try:
            samples = np.array(audio.get_array_of_samples())
            
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                left = samples[:, 0].astype(np.float32)
                right = samples[:, 1].astype(np.float32)
                
                left_clean = self._noise_gate(left)
                right_clean = self._noise_gate(right)
                
                clean_samples = np.column_stack((left_clean, right_clean))
                clean_samples = clean_samples.astype(np.int16)  
            else:
                mono = samples.astype(np.float32)
                clean_mono = self._noise_gate(mono)
                clean_samples = clean_mono.astype(np.int16)  
            
            if audio.channels == 2:
                interleaved = clean_samples.flatten()
                clean_audio = audio._spawn(interleaved.tobytes())
            else:
                clean_audio = audio._spawn(clean_samples.tobytes())
                
            return clean_audio
            
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(f"Noise reduction failed: {e}, using original audio")
            return audio

    def _noise_gate(self, samples: np.ndarray) -> np.ndarray:
        if len(samples) == 0:
            return samples

        frame_size = 256
        hop_size = frame_size // 2

        rms = np.sqrt(np.mean(samples**2))

        if rms < 0.001:  
            threshold = rms * 0.9  
        elif rms < 0.005: 
            threshold = rms * 0.7  
        elif rms < 0.02:  
            threshold = rms * 0.5  
        else:              
            threshold = rms * 0.3  

        frames = []
        for i in range(0, len(samples) - frame_size + 1, hop_size):
            frame = samples[i:i+frame_size]
            
            frame_energy = np.sqrt(np.mean(frame**2))

            if frame_energy < threshold:
                attenuation = frame_energy / threshold
                frame = frame * attenuation * 0.8  
                frame = frame * 1.1 
            
            frames.append(frame)

        clean_signal = np.zeros_like(samples)
        weights = np.zeros_like(samples)
        
        for i, frame in enumerate(frames):
            start_idx = i * hop_size
            end_idx = start_idx + frame_size
            
            if end_idx > len(clean_signal):
                break

            window = np.hanning(frame_size)
            clean_signal[start_idx:end_idx] += frame * window
            weights[start_idx:end_idx] += window

        weights[weights == 0] = 1
        clean_signal = clean_signal / weights

        original_energy = np.sum(samples**2)
        processed_energy = np.sum(clean_signal**2)
        
        if processed_energy > 0 and original_energy > 0:
            energy_ratio = np.sqrt(original_energy / processed_energy)
            energy_ratio = min(max(energy_ratio, 0.8), 1.2)  
            clean_signal = clean_signal * energy_ratio
        
        return clean_signal.astype(np.float32)

    def _apply_volume_normalization(self, audio: AudioSegment) -> AudioSegment:
        try:
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

            samples_normalized = samples / 32768.0
            
            if audio.channels == 2:
                samples_2d = samples_normalized.reshape((-1, 2))
                left = samples_2d[:, 0]
                right = samples_2d[:, 1]

                rms_left = np.sqrt(np.mean(left**2))
                rms_right = np.sqrt(np.mean(right**2))
                rms = (rms_left + rms_right) / 2
            else:
                rms = np.sqrt(np.mean(samples_normalized**2))

            if rms > 1e-12:  
                current_lufs = 20 * np.log10(rms)
                
                gain_db = self.target_lufs - current_lufs

                gain_db = max(-12, min(gain_db, 12))
 
                return audio.apply_gain(gain_db)
            else:
                return audio
            
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(f"Volume normalization failed: {e}, using basic normalization")
            return audio.apply_gain(-6)  

    def _optimized_concatenate(self, segments: List[AudioSegment]) -> AudioSegment:
        if not segments:
            return AudioSegment.empty()
        
        if len(segments) == 1:
            return segments[0]
        
        crossfade_duration = 50
        
        combined = segments[0]
        
        for segment in segments[1:]:
            combined = combined.append(segment, crossfade=crossfade_duration)
        
        return combined

    def _finalize_audio_quality(self, audio: AudioSegment) -> AudioSegment:
        if audio.max_possible_amplitude > 0:
            max_amplitude = audio.max
            if max_amplitude > 0.95 * audio.max_possible_amplitude:
                reduction_db = 20 * np.log10(0.95 * audio.max_possible_amplitude / max_amplitude)
                audio = audio.apply_gain(reduction_db)  
        
        return audio

    def _remove_clicks(self, audio: AudioSegment) -> AudioSegment:
        try:
            samples = np.array(audio.get_array_of_samples())
            channels = audio.channels
            
            if channels == 2:
                samples = samples.reshape((-1, 2))
                left = samples[:, 0].astype(np.float32)
                right = samples[:, 1].astype(np.float32)

                left_clean = self._simple_click_removal(left)
                right_clean = self._simple_click_removal(right)

                clean_samples = np.column_stack((left_clean, right_clean)).astype(samples.dtype)
            else:
                mono = samples.astype(np.float32)
                mono_clean = self._simple_click_removal(mono)
                clean_samples = mono_clean.astype(samples.dtype)
            
            return audio._spawn(clean_samples.flatten().tobytes())
            
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.warning(f"Click removal failed: {e}, returning original audio")
            return audio
    
    def _simple_click_removal(self, samples: np.ndarray) -> np.ndarray:
        if len(samples) < 10: 
            return samples

        diff = np.abs(np.diff(samples))

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
 
        threshold = mean_diff + 2.5 * std_diff 

        min_threshold = np.percentile(diff, 85) * 1.5 
        threshold = max(threshold, min_threshold)

        click_positions = []
        for i in range(len(diff)):
            if diff[i] > threshold:
                click_pos = i + 1  
                if 0 < click_pos < len(samples) - 1:
                    click_positions.append(click_pos)
 
        if len(click_positions) > len(samples) * 0.01:  
            return samples

        output = samples.copy()
        repaired_count = 0
        
        for pos in click_positions:
            window_size = 3
            start = max(0, pos - window_size)
            end = min(len(samples), pos + window_size + 1)

            nearby_clicks = sum(1 for p in click_positions 
                               if start <= p <= end and p != pos)

            if nearby_clicks > 2:
                continue

            window_samples = []
            for i in range(start, end):
                if i != pos and i not in click_positions:
                    window_samples.append(output[i])
            
            if len(window_samples) >= 2:  
                repair_val = np.mean(window_samples)
                output[pos] = repair_val
                repaired_count += 1
        
        return output
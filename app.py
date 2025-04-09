import gradio as gr
import os
import tempfile
import subprocess
import assemblyai as aai
from deep_translator import GoogleTranslator
import pysrt
import logging
import sys
import shutil
from pathlib import Path
import time
from tqdm import tqdm
from gtts import gTTS

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configuration
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
LANGUAGES = {
    "English": "en",
    "Spanish": "es", 
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Hindi": "hi"
}

# TTS voice mapping for different languages
TTS_VOICES = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "ja": "ja-JP",
    "hi": "hi-IN"
}

# Create a permanent output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_audio(video_path):
    """Extract audio from video file using ffmpeg"""
    try:
        logger.info(f"Extracting audio from video: {video_path}")
        audio_path = os.path.join(OUTPUT_DIR, "audio.wav")
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM format
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite output file
            audio_path
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode != 0:
            logger.error(f"Audio extraction failed: {process.stderr}")
            raise Exception(f"Audio extraction failed: {process.stderr}")
        
        return audio_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Audio extraction failed: {str(e)}")

def generate_subtitles(audio_path):
    """Generate subtitles using AssemblyAI"""
    try:
        logger.info(f"Transcribing audio with AssemblyAI: {audio_path}")
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)
        
        srt_path = os.path.join(OUTPUT_DIR, "subtitles.srt")
        logger.info(f"Saving subtitles to: {srt_path}")
        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(transcript.export_subtitles_srt())
            
        return srt_path
    except Exception as e:
        logger.error(f"Subtitle generation failed: {str(e)}", exc_info=True)
        raise Exception(f"Subtitle generation failed: {str(e)}")

def translate_subtitles(srt_path, target_langs):
    """Translate subtitles to target languages"""
    try:
        logger.info(f"Loading subtitles from: {srt_path}")
        subs = pysrt.open(srt_path, encoding="utf-8")
        results = {}
        
        for lang_code in target_langs:
            logger.info(f"Translating to language code: {lang_code}")
            translated_subs = subs[:]
            translator = GoogleTranslator(source="auto", target=lang_code)
            
            for i, sub in enumerate(translated_subs):
                try:
                    sub.text = translator.translate(sub.text)
                    if i % 10 == 0:  # Log progress every 10 subtitles
                        logger.info(f"Translated {i+1}/{len(translated_subs)} subtitles to {lang_code}")
                except Exception as e:
                    logger.warning(f"Failed to translate subtitle: {sub.text}. Error: {str(e)}")
                    # Keep original text if translation fails
            
            output_path = os.path.join(OUTPUT_DIR, f"subtitles_{lang_code}.srt")
            logger.info(f"Saving translated subtitles to: {output_path}")
            translated_subs.save(output_path, encoding='utf-8')
            results[lang_code] = output_path
            
        return results
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}", exc_info=True)
        raise Exception(f"Translation failed: {str(e)}")

def generate_translated_audio(srt_path, target_lang):
    """Generate translated audio using text-to-speech"""
    try:
        logger.info(f"Generating translated audio for {target_lang}")
        subs = pysrt.open(srt_path, encoding="utf-8")
        translated_text = [sub.text for sub in subs]
        
        # Create temporary directory for audio chunks
        temp_dir = os.path.join(OUTPUT_DIR, f"temp_audio_{target_lang}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate TTS for each subtitle
        audio_files = []
        timings = []
        
        for i, sub in enumerate(tqdm(subs, desc=f"Generating {target_lang} speech")):
            text = sub.text.strip()
            if not text:
                continue
                
            # Get timing information
            start_time = (sub.start.hours * 3600 + 
                         sub.start.minutes * 60 + 
                         sub.start.seconds + 
                         sub.start.milliseconds / 1000)
            
            end_time = (sub.end.hours * 3600 + 
                       sub.end.minutes * 60 + 
                       sub.end.seconds + 
                       sub.end.milliseconds / 1000)
            
            duration = end_time - start_time
            
            # Generate TTS audio
            tts_lang = TTS_VOICES.get(target_lang, target_lang)
            audio_file = os.path.join(temp_dir, f"chunk_{i:04d}.mp3")
            
            try:
                # Add a retry mechanism for Hindi and other potentially problematic languages
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        # For Hindi, use slower speed which might improve reliability
                        slow_option = target_lang == "hi" 
                        tts = gTTS(text=text, lang=target_lang, slow=slow_option)
                        tts.save(audio_file)
                        break
                    except Exception as e:
                        retry_count += 1
                        logger.warning(f"TTS attempt {retry_count} failed for {target_lang}: {str(e)}")
                        time.sleep(1)  # Wait before retrying
                        
                        # If still failing after retries, try with shorter text
                        if retry_count == max_retries and len(text) > 100:
                            logger.warning(f"Trying with shortened text for {target_lang}")
                            shortened_text = text[:100] + "..."
                            tts = gTTS(text=shortened_text, lang=target_lang, slow=True)
                            tts.save(audio_file)
                
                if os.path.exists(audio_file) and os.path.getsize(audio_file) > 0:
                    audio_files.append(audio_file)
                    timings.append((start_time, end_time, duration, audio_file))
                else:
                    logger.warning(f"Generated audio file is empty or does not exist: {audio_file}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate TTS for: {text}. Error: {str(e)}")
        
        # Check if we actually generated any audio files
        if not audio_files:
            logger.warning(f"No audio files were generated for {target_lang}")
            # Create a silent audio file as fallback
            silent_audio = os.path.join(OUTPUT_DIR, f"translated_audio_{target_lang}.wav")
            silent_cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', '180',  # 3 minutes default
                '-q:a', '0',
                '-y',
                silent_audio
            ]
            subprocess.run(silent_cmd, capture_output=True)
            return silent_audio
        
        # Create a silent audio track the same length as the original video
        silence_file = os.path.join(temp_dir, "silence.wav")
        try:
            video_duration_cmd = [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'default=noprint_wrappers=1:nokey=1', 
                os.path.join(OUTPUT_DIR, "base_video.mp4")
            ]
            
            duration_result = subprocess.run(video_duration_cmd, capture_output=True, text=True)
            video_duration = float(duration_result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not determine video duration: {str(e)}. Using default of 180 seconds.")
            video_duration = 180.0
        
        # Create silent audio track
        silent_cmd = [
            'ffmpeg',
            '-f', 'lavfi',
            '-i', f'anullsrc=r=44100:cl=stereo',
            '-t', str(video_duration),
            '-q:a', '0',
            '-y',
            silence_file
        ]
        subprocess.run(silent_cmd, capture_output=True)
        
        # Create a file with the audio mixing commands
        filter_complex = []
        input_count = 1  # Starting with 1 because 0 is the silence track
        
        # Start with silent track
        filter_parts = ["[0:a]"]
        
        # Add each audio segment
        for start_time, end_time, duration, audio_file in timings:
            filter_parts.append(f"[{input_count}:a]adelay={int(start_time*1000)}|{int(start_time*1000)}")
            input_count += 1
        
        # Mix all audio tracks
        filter_parts.append(f"amix=inputs={input_count}:dropout_transition=0:normalize=0[aout]")
        filter_complex = ";".join(filter_parts)
        
        # Build the ffmpeg command with all audio chunks
        cmd = ['ffmpeg', '-y']
        
        # Add silent base track
        cmd.extend(['-i', silence_file])
        
        # Add all audio chunks
        for audio_file in audio_files:
            cmd.extend(['-i', audio_file])
        
        # Add filter complex and output
        output_audio = os.path.join(OUTPUT_DIR, f"translated_audio_{target_lang}.wav")
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[aout]',
            output_audio
        ])
        
        # Run the command
        logger.info(f"Combining audio segments: {' '.join(cmd)}")
        process = subprocess.run(cmd, capture_output=True)
        
        if process.returncode != 0:
            logger.error(f"Audio combination failed: {process.stderr}")
            # Create a fallback silent audio as last resort
            silent_audio = os.path.join(OUTPUT_DIR, f"translated_audio_{target_lang}.wav")
            silent_cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', str(video_duration),
                '-q:a', '0',
                '-y',
                silent_audio
            ]
            subprocess.run(silent_cmd, capture_output=True)
            output_audio = silent_audio
        
        # Verify the output file exists
        if not os.path.exists(output_audio):
            logger.error(f"Output audio file does not exist: {output_audio}")
            # Create emergency fallback
            silent_audio = os.path.join(OUTPUT_DIR, f"translated_audio_{target_lang}.wav")
            silent_cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', '180',
                '-q:a', '0',
                '-y',
                silent_audio
            ]
            subprocess.run(silent_cmd, capture_output=True)
            output_audio = silent_audio
        
        # Clean up temporary files
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {str(e)}")
        
        return output_audio
    except Exception as e:
        logger.error(f"Audio translation failed: {str(e)}", exc_info=True)
        # Create an emergency fallback silent audio
        try:
            silent_audio = os.path.join(OUTPUT_DIR, f"translated_audio_{target_lang}.wav")
            silent_cmd = [
                'ffmpeg',
                '-f', 'lavfi',
                '-i', f'anullsrc=r=44100:cl=stereo',
                '-t', '180',
                '-q:a', '0',
                '-y',
                silent_audio
            ]
            subprocess.run(silent_cmd, capture_output=True)
            return silent_audio
        except:
            raise Exception(f"Audio translation failed: {str(e)}")

def combine_video_audio_subtitles(video_path, audio_path, srt_path, output_path):
    """Combine video with translated audio and subtitles"""
    try:
        logger.info(f"Combining video, audio, and subtitles")
        
        # Verify that all input files exist
        if not os.path.exists(video_path):
            raise Exception(f"Video file does not exist: {video_path}")
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file does not exist: {audio_path}")
        if not os.path.exists(srt_path):
            raise Exception(f"Subtitle file does not exist: {srt_path}")
            
        logger.info(f"Input files verified: Video: {os.path.getsize(video_path)} bytes, Audio: {os.path.getsize(audio_path)} bytes, Subtitles: {os.path.getsize(srt_path)} bytes")
        
        # Create a safe version of the subtitle path
        safe_srt_path = srt_path.replace(" ", "\\ ").replace(":", "\\:")
        
        # Command to combine video with translated audio and subtitles
        try:
            # Attempt method 1: Using subtitles filter
            cmd = [
                'ffmpeg',
                '-i', video_path,           # Input video
                '-i', audio_path,           # Input translated audio
                '-map', '0:v',              # Use video from first input
                '-map', '1:a',              # Use audio from second input
                '-vf', f"subtitles={safe_srt_path}:force_style='FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3'",  # Burn subtitles
                '-c:v', 'libx264',          # Video codec
                '-c:a', 'aac',              # Audio codec
                '-shortest',                # End when shortest input ends
                '-y',                       # Overwrite output file
                output_path
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                logger.warning(f"First method failed: {process.stderr}")
                raise Exception("First method failed")
                
        except Exception as e:
            logger.warning(f"First method failed: {str(e)}")
            
            try:
                # Attempt method 2: Using hardcoded subtitles approach
                temp_srt_dir = os.path.join(OUTPUT_DIR, "temp_srt")
                os.makedirs(temp_srt_dir, exist_ok=True)
                
                # Copy the SRT file to the temp directory
                temp_srt_path = os.path.join(temp_srt_dir, "temp.srt")
                shutil.copy(srt_path, temp_srt_path)
                
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', audio_path,
                    '-map', '0:v',
                    '-map', '1:a',
                    '-vf', f"subtitles={temp_srt_path}",
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-shortest',
                    '-y',
                    output_path
                ]
                
                logger.info(f"Running second method: {' '.join(cmd)}")
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    logger.warning(f"Second method failed: {process.stderr}")
                    raise Exception("Second method failed")
                    
                # Clean up temp directory
                shutil.rmtree(temp_srt_dir)
                
            except Exception as e:
                logger.warning(f"Second method failed: {str(e)}")
                
                # Attempt method 3: No subtitles as last resort
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-i', audio_path,
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-shortest',
                    '-y',
                    output_path
                ]
                
                logger.info(f"Running fallback method (no subtitles): {' '.join(cmd)}")
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    logger.error(f"All methods failed: {process.stderr}")
                    raise Exception(f"Failed to combine video and audio: {process.stderr}")
                else:
                    logger.warning("Created video without subtitles as fallback")
        
        # Verify the output file exists and has a reasonable size
        if not os.path.exists(output_path):
            raise Exception(f"Output file does not exist: {output_path}")
            
        if os.path.getsize(output_path) < 1000:
            raise Exception(f"Output file is too small: {os.path.getsize(output_path)} bytes")
            
        logger.info(f"Successfully created output file: {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path
    except Exception as e:
        logger.error(f"Combining failed: {str(e)}", exc_info=True)
        raise Exception(f"Combining failed: {str(e)}")

def process_video(video_file, source_lang, target_langs, progress=gr.Progress()):
    """Process video with translation of both subtitles and audio"""
    try:
        progress(0.05, "Starting processing...")
        logger.info(f"Processing video: {video_file}")
        
        # Make sure we have ffmpeg installed
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("ffmpeg is installed and working")
        except (subprocess.SubprocessError, FileNotFoundError):
            error_msg = "ffmpeg is not installed or not in PATH. Please install ffmpeg."
            logger.error(error_msg)
            return None, error_msg
        
        # Extract audio
        progress(0.1, "Extracting audio...")
        audio_path = extract_audio(video_file)
        
        # Generate subtitles
        progress(0.25, "Generating subtitles...")
        srt_path = generate_subtitles(audio_path)
        
        # Translate subtitles
        progress(0.4, "Translating subtitles...")
        target_lang_codes = [LANGUAGES[lang] for lang in target_langs]
        translated_subs = translate_subtitles(srt_path, target_lang_codes)
        
        # Create a copy of the video file in our output directory
        base_video = os.path.join(OUTPUT_DIR, "base_video.mp4")
        shutil.copy(video_file, base_video)
        
        # Process each target language
        output_videos = []
        
        for i, (lang_code, sub_path) in enumerate(translated_subs.items()):
            lang_name = next(name for name, code in LANGUAGES.items() if code == lang_code)
            progress(0.5 + (i * 0.5 / len(translated_subs)), f"Processing {lang_name}...")
            
            try:
                # Generate translated audio
                logger.info(f"Generating translated audio for {lang_code}")
                translated_audio = generate_translated_audio(sub_path, lang_code)
                
                # Verify audio file exists
                if not os.path.exists(translated_audio):
                    logger.error(f"Translated audio file does not exist: {translated_audio}")
                    continue
                    
                # Combine video, translated audio, and subtitles
                output_path = os.path.join(OUTPUT_DIR, f"output_{lang_code}.mp4")
                logger.info(f"Creating final video with {lang_code} audio and subtitles")
                
                output_video = combine_video_audio_subtitles(
                    base_video, 
                    translated_audio, 
                    sub_path, 
                    output_path
                )
                
                # Verify the output file exists and has content
                if os.path.exists(output_video) and os.path.getsize(output_video) > 1000:
                    logger.info(f"Successfully created output file: {output_video}")
                    output_videos.append(output_video)
                else:
                    logger.warning(f"Output file is missing or too small: {output_video}")
            except Exception as e:
                logger.error(f"Failed to process {lang_code}: {str(e)}")
        
        # If all output videos failed, return the original
        if not output_videos:
            logger.warning("All translations failed, returning original video")
            return base_video, "Failed to translate video, returning original"
            
        progress(1.0, "Done!")
        message = f"Processing complete. Created {len(output_videos)} translated videos."
        logger.info(message)
        return output_videos[0], message
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return None, f"Processing failed: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Complete Video Translation System")
    gr.Markdown("Translates both subtitles and audio to target languages")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            source_lang = gr.Dropdown(
                label="Source Language",
                choices=list(LANGUAGES.keys()),
                value="English"
            )
            target_langs = gr.CheckboxGroup(
                label="Target Languages (Both Audio & Subtitles)",
                choices=list(LANGUAGES.keys()),
                value=["Spanish"]
            )
            submit_btn = gr.Button("Translate", variant="primary")
        
        with gr.Column(scale=2):
            output_video = gr.Video(label="Translated Video")
            status_text = gr.Textbox(label="Status", interactive=False)
            output_info = gr.Markdown("Output videos will be saved in the 'outputs' directory")

    submit_btn.click(
        process_video,
        inputs=[video_input, source_lang, target_langs],
        outputs=[output_video, status_text]
    )

if __name__ == "__main__":
    # Check dependencies at startup
    missing_deps = []
    
    # Check ffmpeg
    try:
        version_info = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        ffmpeg_version = version_info.stdout.split('\n')[0]
        logger.info(f"ffmpeg version: {ffmpeg_version}")
    except:
        logger.warning("ffmpeg not found - required for video processing")
        missing_deps.append("ffmpeg")
    
    # Check Python dependencies
    try:
        import assemblyai
        logger.info("AssemblyAI package found")
    except ImportError:
        logger.warning("AssemblyAI package not found - required for transcription")
        missing_deps.append("assemblyai")
        
    try:
        import gtts
        logger.info("gTTS package found")
    except ImportError:
        logger.warning("gTTS package not found - required for text-to-speech")
        missing_deps.append("gtts")
    
    try:
        import deep_translator
        logger.info("deep_translator package found")
    except ImportError:
        logger.warning("deep_translator package not found - required for translation")
        missing_deps.append("deep_translator")
    
    # Print installation instructions if dependencies are missing
    if missing_deps:
        logger.warning("Missing dependencies detected. Please install:")
        if "ffmpeg" in missing_deps:
            logger.warning("- ffmpeg: https://ffmpeg.org/download.html")
        
        python_deps = [dep for dep in missing_deps if dep != "ffmpeg"]
        if python_deps:
            deps_str = " ".join(python_deps)
            logger.warning(f"- Python packages: pip install {deps_str}")
    
    # Start the app
    demo.launch()
# 🌐 LinguaStream – Multilingual Video and Audio Translator

**LinguaStream** is an advanced end-to-end application that takes any input video, transcribes its audio using AI, translates both subtitles and speech into multiple target languages, and outputs a fully localized video — all within an intuitive Gradio interface.

---

## 🚀 Features

- 🎙️ **Speech-to-Text with Subtitles**  
  Transcribes original audio into accurate `.srt` subtitle files using [AssemblyAI](https://www.assemblyai.com/).

- 🌍 **Multilingual Translation**  
  Translates subtitles into selected languages using [deep-translator](https://github.com/nidhaloff/deep-translator) (Google Translate backend).

- 🔊 **Text-to-Speech (TTS) Dubbing**  
  Converts translated text into synchronized audio using [gTTS](https://pypi.org/project/gTTS/) with support for languages like Spanish, French, Hindi, and more.

- 🎞️ **Video & Audio Merging**  
  Combines translated audio, burned subtitles, and original video using `ffmpeg` for smooth output generation.

- 🧠 **Fault-Tolerant Pipeline**  
  Robust retry mechanisms, silent audio fallback, and multi-step error handling ensure reliability at scale.

- 🌐 **User Interface**  
  Powered by [Gradio](https://gradio.app/) for seamless interaction — simply upload a video, select source and target languages, and download the final output.

---

## 📦 Demo (Hugging Face Spaces)

👉 [Try it Live on Hugging Face](https://huggingface.co/spaces/YourUsername/LinguaStream-multilingual-video-translator)

---

## 🛠️ Tech Stack

| Category            | Tools/Libraries                             |
|---------------------|---------------------------------------------|
| UI                  | Gradio                                      |
| Transcription       | AssemblyAI                                  |
| Translation         | deep-translator (Google Translate API)      |
| TTS Audio           | gTTS (Google Text-to-Speech)                |
| Audio/Video Editing | ffmpeg, ffmpeg-python, moviepy              |
| Subtitles           | pysrt                                       |
| Automation          | CI/CD (GitHub Actions), Hugging Face Spaces |

---

## ⚙️ How It Works

1. **Upload** a video (MP4 or similar).
2. The app:
   - Extracts the audio using `ffmpeg`.
   - Generates English subtitles using AssemblyAI.
   - Translates those subtitles into selected languages.
   - Uses gTTS to generate translated speech synced with timing.
   - Recombines everything into a new video with embedded translated audio and subtitles.
3. **Download** the final multilingual video.

---

## 📁 Directory Structure


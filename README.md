# AI-VOICELINE-PIPELINE-DESIGN

## Overview

This repository contains a Python implementation of a voice query pipeline that captures voice commands, processes them to extract text, generates formal definitions using a language model, and converts the responses to speech. The solution is designed for low latency and includes voice activity detection (VAD) and customizable speech characteristics.

## Features

- **Voice Activity Detection (VAD):** Filters out non-speech segments from the recorded audio.
- **Speech-to-Text:** Converts spoken language into text using the Whisper model.
- **Response Generation:** Produces formal definitions using the GPT-Neo language model.
- **Text-to-Speech:** Converts generated text responses into speech using the `edge-tts` library.
- **Playback:** Plays the generated speech audio file.

## Requirements

- Python 3.7+
- `numpy`
- `sounddevice`
- `webrtcvad`
- `whisper`
- `torch`
- `transformers`
- `edge-tts`
- `soundfile`
- `asyncio`

You can install the required Python libraries using the following command:

```bash
pip install numpy sounddevice webrtcvad whisper torch transformers edge-tts soundfile

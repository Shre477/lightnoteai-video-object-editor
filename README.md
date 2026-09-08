# LightNoteAI Video Object Editor

An AI-assisted video object editing system that allows users to edit objects in a video using natural-language instructions.

## ✨ Features

- Natural-language video editing instructions
- AI-based instruction parsing
- Object detection using OWL-ViT
- Object segmentation using SAM / GrabCut
- Object tracking using CSRT
- Object removal using OpenCV inpainting
- Object replacement and compositing
- FastAPI backend
- HTML, CSS and JavaScript frontend
- Demo mode for local testing

## 🏗️ Architecture

```text
User
  ↓
Frontend
  ↓
FastAPI Backend
  ↓
Instruction Parser
  ↓
Object Detector
  ↓
Object Segmenter
  ↓
Object Tracker
  ↓
Video Editing Pipeline
  ↓
Inpainting / Compositing
  ↓
Processed Video
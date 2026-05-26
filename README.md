# 🚗 Autonomous Vehicle Perception System on Quanser QCar

> **MSc Mechatronics Dissertation** - End-to-end deep learning perception 
> pipeline with multi-sensor fusion, deployed on Quanser QCar hardware.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

End-to-end autonomous vehicle perception system developed for MSc Mechatronics 
dissertation at Kingston University, London. The system progresses from 
**simulation (QLabs)** to **real hardware deployment (Quanser QCar)**, 
demonstrating sim-to-real transfer of deep learning models.

## 🏆 Key Achievements

| Metric | Simulation | Real Hardware |
|--------|------------|---------------|
| **YOLOv8 Traffic Detection (mAP@0.5)** | 99.50% | 92.87% |
| **ENet Lane Segmentation (mIoU)** | - | 93.73% |
| **End-to-End Performance** | 25 FPS | 18-22 FPS |
| **System Latency** | <30ms | <50ms |
| **Sensor Fusion False Positives** | 0 | 0 |

**Sim-to-Real Gap:** Quantified at 6.57% performance degradation.

## 🏗️ System Architecture

The system uses a **distributed architecture** between the Quanser QCar 
(running on Jetson TX2) and a laptop with NVIDIA RTX 3060 GPU.

**Sensors on QCar:**
- CSI Camera (RGB) - Lane detection input
- Intel RealSense D435 - Depth perception
- RPLidar A2 - 360° obstacle detection

**Communication:** TCP/IP socket connection with <50ms latency

## ✨ Features

### 🔍 Object Detection (YOLOv8)
- 5-class custom-trained model
- Classes: Red light, Green light, Yellow light, Stop sign, No-right-turn
- 3,802 annotated training instances
- Real-time inference at 18-25 FPS

### 🛣️ Lane Segmentation (ENet vs U-Net)
- **ENet**: 93.73% mIoU, 0.37M params (chosen for deployment)
- **U-Net**: 94.12% mIoU, 32M params (baseline)
- 87x parameter reduction with only 0.39% accuracy loss

### 🎯 Multi-Sensor Fusion
- **Intel RealSense D435** depth camera
- **RPLidar A2** 360° LiDAR
- AND-logic fusion for zero false positives
- Kalman filtering for state estimation

### 🏎️ Autonomous Control
- Pure Pursuit trajectory tracking
- Adaptive speed control (slow zones, post-curve)
- Automatic braking on obstacle detection
- Waypoint-based navigation for complex maneuvers

### 🌐 Distributed Architecture
- TCP/IP communication between QCar (Jetson) and laptop GPU
- <50ms end-to-end latency
- Production-ready scalable design

## 📁 Repository Structure

```
├── docs/                          # Documentation
│   └── dissertation.pdf
│
├── simulation/                    # QLabs Simulation Code
│   ├── main/
│   │   ├── final_method.py        ⭐ Main file
│   │   ├── environment.py
│   │   └── qlabs_setup.py
│   ├── yolo_training/
│   ├── dataset_tools/
│   ├── utilities/
│   ├── waypoints/
│   └── models/
│
├── real_qcar/                     # Real Hardware Code
│   ├── stream/
│   │   └── stream_server_enet_lidar+realsense.py ⭐ Production
│   ├── perception/
│   └── utilities/
│
├── demo/                          # Videos and images
└── results/                       # Performance metrics

```                

## 🚀 Quick Start

### Simulation (QLabs)

```bash
cd simulation/main
python final_method.py
```

### Real Hardware (QCar)

**On QCar (Jetson TX2):**
```bash
cd real_qcar/stream
python stream_client_lidar.py
```

**On Laptop (GPU inference):**
```bash
cd real_qcar/stream
python stream_server_enet_lidar+realsense.py
```

## 🛠️ Tech Stack

**Languages:** Python 3.10  
**Deep Learning:** TensorFlow, Keras, PyTorch (YOLOv8), Ultralytics  
**Computer Vision:** OpenCV  
**Robotics:** Quanser QCar API, QLabs Interactive Labs  
**Hardware:** NVIDIA Jetson TX2, Quanser QCar 1:10 scale  
**Sensors:** Intel RealSense D435, RPLidar A2, CSI Camera  
**Networking:** TCP/IP socket communication  
**Future:** Migrating to ROS2 Humble (in progress)

## 📊 Development Journey

1. **Simulation First (QLabs)** - Validated algorithms in safe environment
2. **U-Net Lane Detection** - Baseline implementation (32M parameters)
3. **ENet Optimization** - 87x parameter reduction for embedded deployment
4. **LiDAR Integration** - 360° obstacle detection
5. **RealSense Integration** - Depth-based perception
6. **Multi-Sensor Fusion** - AND-logic for zero false positives
7. **Real Hardware Deployment** - QCar with distributed architecture

## 📖 Dissertation

📄 **[Full Dissertation (120 pages)](docs/dissertation.pdf)**

## 🎓 About

**Author:** Sarath Kumar Komathukattil  
**Degree:** MSc Mechatronic Systems  
**University:** Kingston University, London (2025)

## 🔮 Future Work

- [x] Simulation in QLabs
- [x] Real hardware deployment (QCar)
- [x] Multi-sensor fusion
- [x] ROS2 Basics certification
- [ ] Migration to ROS2 (in progress)
- [ ] ROS2 Perception integration
- [ ] Nav2 navigation stack
- [ ] TensorRT optimization

## 📫 Contact

- 🇨🇦 Location: Canada 
- 🎯 **Seeking:** Perception Engineer, Computer Vision Engineer, Robotics Software Engineer roles

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

⭐ **If you find this project interesting, please consider starring the repository!**
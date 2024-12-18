# CV demos for for Raspberry Pi
Here, we will work through a series of demos demonstrating computer vision application on the Raspberry Pi

## Dependencies and installation
The pis you will be working on during the workshop will have libraries and dependencies installed ahead of time. If you are working from a new Pi setup, you will need:

These exercises assume you have:
- A fully updated (Dec 2025) Raspberry Pi 5 with Python installed.
- A generic USB webcam connected.
- openCV for python installed and active in your environment. For the exercises here, we created a virtual enviornment and installed using the following commands:

```bash
python3 -m venv sicb_2025
source sicb_2025/bin/activate
pip3 install opencv-contrib-python
```
You can then clone the repository with these exercises to your pi with this command:

```bash
git clone https://github.com/Crall-Lab/SICB2025_CV.git
```

## Demos
### Webcam preview
### Motion tracking
### Color-based tracking
### ArUco tracking

## Bonus exercise
### tracking a red, moving object
### Use motion detection to trigger video capture
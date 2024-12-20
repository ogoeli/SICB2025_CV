# CV demos for for Raspberry Pi
Here, we will work through a series of demos demonstrating computer vision application on the Raspberry Pi

## Dependencies and installation
The pis you will be working on during the workshop will have libraries and dependencies installed ahead of time. If you are working from a new Pi setup, you will need:

These exercises assume you have:
- A fully updated (Dec 2025) Raspberry Pi 5 with Python installed.
- A generic USB webcam connected.
- openCV for python installed and active in your environment. For the exercises here, we created a virtual enviornment and installed using the following commands:

For the workshop, we will connect to your Raspberry Pi devices using [Raspberry Pi connect](https://connect.raspberrypi.com/sign-in), a simple remote desktop application for Raspberry Pis. Login info will be provided the day of the workshop.

```bash
python3 -m venv sicb_2025
source sicb_2025/bin/activate
pip3 install opencv-contrib-python
```

You can then clone the repository with these exercises to your pi with this command:
```bash
cd ~/Documents
git clone https://github.com/Crall-Lab/SICB2025_CV.git
```

Then navigate into the repository you just cloned, and confirm the 
```bash
cd SICB2025_CV
ls
```
Now navigate into the pi_exercises subdirectory
```bash
cd pi_exercises
```

## Demos


First, we will connect to the webcam and make sure it's working properly!
### Webcam preview
```bash
python3 webcam_preview.py
```


### Motion tracking
Next, we will test out some motion-based tracking, using pixel-level difference thresholding
```bash
python3 motion_tracking_demo.py
```

### Color-based tracking
Next, we will implement the same kind of color thresholding we performed 
```bash
python3 color_tracking_demo.py
```

### ArUco tracking
Finally, here's an example of built-in functionality in openCV to detect unique tags (ArUco markers). For an example of how this type of tracking can be implemented for pollinator experiments, check out the [BUmbleBox repository](https://github.com/Crall-Lab/BumbleBox)
```bash
python3 aruco_live_demo.py
```

## Bonus exercise
If you want to test out what you've learned, try writing either or both of the following scripts (either during the workshop, or on your own time):


### tracking a red, moving object
First, try tracking a moving, red object by making a new script called 'track_moving_red_obj.py':
```bash
python3 track_moving_red_obj.py
```
*Hint: this will require integrating parts of the color and motion tracking scripts*



### Use motion detection to trigger video capture
Second, make a script that records a brief (10 second) video to the Documents folder, called 'motion_capture.py':
```bash
python3 track_moving_red_obj.py
```
*Hint: you will need to explore options for writing video files to disk in Python*
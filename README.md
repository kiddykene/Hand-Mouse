# Hand-Mouse
This is a python project that is aimed at using a webcam to enable your hands to have full reckoning over your computer, this is extremely stupid.

DO NOTE IF YOU WANT TO RUN THE RAW CODE, YOU MUST USE MEDIAPIPE VERSION 0.10.9

run main.py. Upon launching make sure you can see your hand is being detected, while it is, follow the instructions to callibrate at the top (aka point towards each corner of your monitor and for each position click the calibration key, by default it's f8 but you can change it to what ever. Then make a fist and once again point towards each corner of the monitor, then while retaining the fist, point towards the center of the monitor, this center pos will be your resting point in raw mode).  Once you're done calibrating, you will be able to control your computer with your hand, just point where you want the cursor to go and it should feel quite clean, if not, tweak the settings to fit your style idfk. To swap to gaming mode, hold a fist for the length of the raw_mode_threshold variable, you should see it swap based on the visual. In this raw mode, you can use your pinky and thumb to click by default, and your left hand could also then be used for wasd. To leave this mode, stop making a fist and flatten out your hand. If you want to change the keybinds, open the keybinds menu, go to the mode you want, and change the corresponding finger's key. If you believe the thresholds feel weird, change them to better fit what's needed. If you want to remove a bind from a finger, right click the finger. In the main gui for style points you can also change the color of the whole gui (so fancy).

Just about the worse part of all of this is the mistake I made at the start of the project, which was to try using an llm to get the main framework layed out, misserable idea. Aaand because of that i couldn't be bothered to fix the layout much so enjoy the messed up speggeti code (WTFFFFFF).

Most of the work done for this project is under under the process_frame function in Logic

This project features homography for mapping the position of my finger tip to screen coordinates, and thin plate spline for appying better warping in raw mode (aka gaming mode).

written in python 3.10

I am so tired

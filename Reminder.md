# The two pip OpenCV packages: opencv-python and opencv-contrib-python
Before we get started I want to remind you that the methods I’m coming here today are unofficial pre-built OpenCV 
packages that can be installed via pip — they are not official OpenCV packages released by OpenCV.org.
Just because they are not official packages doesn’t mean you should feel uncomfortable using them, 
but it’s important for you to understand that they are not endorsed and supported directly by the official OpenCV.org team.
All that said — there are four OpenCV packages that are pip-installable on the PyPI repository:
- opencv-python: This repository contains just the main modules of the OpenCV library. If you’re a PyImageSearch reader you do not want to install this package.
- opencv-contrib-python: The opencv-contrib-python repository contains both the main modules along with the contrib modules — this is the library I recommend you install as it includes all OpenCV functionality.
- opencv-python-headless: Same as opencv-python but no GUI functionality. Useful for headless systems.
- opencv-contrib-python-headless: Same as opencv-contrib-python but no GUI functionality. Useful for headless systems.
Again, in the vast majority of situations you will want to install opencv-contrib-python on your system.
You DO NOT want to install both opencv-python and opencv-contrib-python — pick ONE of them.

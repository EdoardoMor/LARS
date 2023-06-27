# DrumsDemix
MAE Capston project - a Drums Demixing plugin

<div style="background-color: rgb(167, 202, 212); border-radius: 15px; padding: 5px">
<image src="drums_demix/DrumsDemixUtils/DrumsDemixImages/DD_GUI.png" style="margin-left: -2px; border-radius: 15px"></image>
</div>



## Description
We present Drums-Demix, the first to-our-knowledge neural network-based commercial application for drum stems separation.

## Requirements
* CMake
* Libtorch
* Juce

## How to run it
* Download/clone the project repo
* In the CMakeLists.txt file, modify these lines typing in your paths to the libtorch and Juce folders:
  * set(CMAKE_PREFIX_PATH "[your libtorch folder]")
  * add_subdirectory([your JUCE folder] ./JUCE)
* Copy and paste the DrumsDemixUtils folder on you Desktop
* In the project folder:
```console
cmake -B build .
```
* Compile with the IDE of your preference and run the plug-in

## Functionalities
* Load your drums stem file by:
  * Drag and dropping
  * Clicking the load button
* Click the separate button and wait for the resulting stems of 5 different drum kit parts, which will be:
  * Kick
  * Snare
  * Toms
  * Hihat
  * Cymbals
* Use the Play/Pause buttons and click on the waveforms to listen to the segment of the drum part you are more interested in
* You can download the drum part stem in a .wav format in two different ways, by:
  * Clicking the download button
  * Drag and dropping the file directly from the plug-in to anywhere you want!   

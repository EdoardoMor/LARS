# Drums-Demix
MAE Capston project - a Drums Demixing plugin

<div style="background-color: rgb(167, 202, 212); border-radius: 15px; padding: 5px">
<image src="drums_demix/DrumsDemixUtils/DrumsDemixImages/GUI.PNG" style="margin-left: -2px; border-radius: 15px"></image>
</div>



## Description
We present Drums-Demix, the first to-our-knowledge neural network-based commercial application for drum stems separation. 
Trained in Python and running in C++, our plug-in offers surprising versatility and ease of use to allow everybody to enjoy it either in a professional or amateur environment. 
An intuitive GUI gives all the possibilities to interact with the DAW and to extract the separated tracks, besides being able to play independently the single audio files.

## Requirements
* [CMake](https://cmake.org) 
* [Libtorch](https://pytorch.org/get-started/locally/)
* [Juce](https://juce.com)

## How to run it
* Download/clone the project repo
* In the CMakeLists.txt file, modify these lines typing in your paths to the libtorch and Juce folders:
  * set(CMAKE_PREFIX_PATH "[your libtorch folder]")  [Line 25]
  * add_subdirectory([your JUCE folder] ./JUCE)   [Line 31]
* Copy and paste the DrumsDemixUtils folder on your Desktop
* From terminal, go to the project folder and run:
```console
cmake -B build .
```
* This will generate a project file in the build folder, open it
* Compile with the IDE of your preference and run the plug-in

## Functionalities
* Load your drums track file by:
  * Drag and dropping from external files
  * Clicking the load button
* Click the separate button and wait for the resulting stems of 5 different drum kit parts, which will be:
  * Kick
  * Snare
  * Toms
  * Hihat
  * Cymbals
* Use the Play/Pause buttons or double click on the waveforms to listen to the segment of the drum part you are more interested in
* You can download the separated stems in a .wav format in two different ways, by:
  * Clicking the download button
  * Drag and dropping the file directly from the plug-in to anywhere you want!

## Link

* [Presentation](https://drive.google.com/file/d/19SA2RIHljjGD7Um65_ZcaB2VlucqQiXA/view?usp=drivesdk)
* [GitHub](https://github.com/EdoardoMor/DrumsDemix)
 

 
 

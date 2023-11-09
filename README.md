# LARS
A neural drums demixing plug-in

<div style="background-color: rgb(167, 202, 212); border-radius: 15px; padding: 5px">
<image src="drums_demix/DrumsDemixUtils/DrumsDemixImages/GUI.PNG" style="margin-left: -2px; border-radius: 15px"></image>
</div>



## Description
We present LARS, the first open-source AI-powered plug-in for drum source separation.

Developed in Python and running in C++, LARS is a versatile tool for music producers and sound engineers alike. Thanks to its user-friendly interface, LARS makes it easy to accomplish complex tasks such as drum replacement, drum loop decomposition, audio restoration, remixing, and remastering—all within your DAW of choice.

LARS can separate a stereo drum track into five audio stems: **kick drum**, **snare**, **toms**, **hi-hat**, and **cymbals**. 

LARS is available both as a VST3 and Audio Unit (AU) plug-in.

## Requirements
* [CMake](https://cmake.org) 
* [Libtorch](https://pytorch.org/get-started/locally/)
* [Juce](https://juce.com)

## How to run LARS
* Download/clone the project repo
* In the `CMakeLists.txt` file, modify the following lines by typing in the path to your LibTorch and JUCE folders:
  * `set(CMAKE_PREFIX_PATH "{your-libtorch-folder}")`  [Line 25]
  * `add_subdirectory({your-JUCE-folder} ./JUCE)`  [Line 31]
* On your terminal, go to the project folder and run:
```console
cmake -B build .
```
* This will generate a project file in the build folder, open it.
* Compile the plug-in with your favorite IDE (we tested it only on VSCommunity and XCode) and run it!

## Functionalities
* Load your drum tracks:
  * Drag and drop your external files;
  * Click the load button.
* Click the `separate` button and extract five drum kit stems:
  * Kick 
  * Snare
  * Toms
  * Hi-hat
  * Cymbals
* Use the `play`/`pause` buttons or double click on the waveforms to listen to the portion of the audio files you are more interested in.
* You can download the separated stems in a `wav` format in two different ways:
  * Click the download button;
  * Drag and drop the audio clips directly from the plug-in to anywhere you want!

## Links

* [Presentation](https://drive.google.com/file/d/19SA2RIHljjGD7Um65_ZcaB2VlucqQiXA/view?usp=drivesdk)
* [GitHub](https://github.com/EdoardoMor/DrumsDemix)

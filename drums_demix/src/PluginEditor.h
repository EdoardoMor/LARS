/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

// not this:
//#include <JuceHeader.h>
// but this:
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cmath>


#include "PluginProcessor.h"
#include "NeuralNetwork.h"

//==============================================================================
/**
*/
class DrumsDemixEditor  :   public juce::AudioProcessorEditor,
                          // listen to buttons
                          public juce::Button::Listener, 
                          // listen to sliders
                          public juce::Slider::Listener, 
                          // listen to piano keyboard widget
                          private juce::MidiKeyboardState::Listener,
                          // listen to AudioThumbnail
                          public juce::ChangeListener,
                          public juce::FileDragAndDropTarget
                          

{
public:
    DrumsDemixEditor (DrumsDemixProcessor&);
    ~DrumsDemixEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

    void sliderValueChanged (juce::Slider *slider) override;
    void buttonClicked(juce::Button* btn) override;
    // from MidiKeyboardState
    void handleNoteOn(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float
 velocity) override; 
     // from MidiKeyboardState
    void handleNoteOff(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity) override; 

    juce::AudioBuffer<float> getAudioBufferFromFile(juce::File file);

    //VISUALIZER
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void thumbnailChanged();
    void paintIfNoFileLoaded(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds);
    void paintIfFileLoaded(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav);

    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

    void loadFile(const juce::String& path);
    


private:


    enum TransportState
    {
      Stopped,
      Starting,
      Stopping,
      Playing
    };

    TransportState state;


    juce::ToggleButton envToggle; 

    // torch::nn::Linear linear{1, 2}; 
    //NeuralNetwork nn{1, 2};

    // needed for the mini piano keyboard
    juce::MidiKeyboardState kbdState;
    juce::MidiKeyboardComponent miniPianoKbd; 

    //the test button
    juce::TextButton testButton;
    juce::TextButton openButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;
    
    //VISUALIZER
    juce::AudioThumbnail thumbnail;
    juce::AudioThumbnailCache thumbnailCache;

    juce::AudioThumbnail thumbnailOut;
    juce::AudioThumbnailCache thumbnailCacheOut;
    
    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::AudioFormatReaderSource> playSource;
    juce::File myFile;
    juce::File myFileOut;
    void transportStateChanged(TransportState newState);
    void displayOut(juce::File file);

    juce::AudioBuffer<float> bufferY;
    juce::AudioBuffer<float> bufferOut;

    std::vector<float> audioPoints;
    std::vector<float> vectoryOut;
    std::vector<float> vectoryOut2;
    

    //audioPoints.call_back(new float (args));
    bool paintOut{ false };
    
    
    //load a TorchScript module:
    torch::jit::script::Module mymodule;
    

    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    DrumsDemixProcessor& audioProcessor;
    

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumsDemixEditor)
};

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


#include "PluginProcessor.h"
#include "NeuralNetwork.h"

//==============================================================================
/**
*/
class FMPluginEditor  :   public juce::AudioProcessorEditor,
                          // listen to buttons
                          public juce::Button::Listener, 
                          // listen to sliders
                          public juce::Slider::Listener, 
                          // listen to piano keyboard widget
                          private juce::MidiKeyboardState::Listener

{
public:
    FMPluginEditor (FMPluginProcessor&);
    ~FMPluginEditor() override;

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



private:
    juce::ToggleButton envToggle; 

    // torch::nn::Linear linear{1, 2}; 
    //NeuralNetwork nn{1, 2};

    // needed for the mini piano keyboard
    juce::MidiKeyboardState kbdState;
    juce::MidiKeyboardComponent miniPianoKbd; 

    //the test button
    juce::TextButton testButton;
    

    //load a TorchScript module:
    torch::jit::script::Module mymodule;

    juce::AudioFormatManager formatManager;

    /*
    try{
      mymodule = torch::jit::load('my_scripted_module.pt');
    }
    catch(const c10::Error& e) {
      return -1; //indicate error to calling the code
    }*/

    

    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    FMPluginProcessor& audioProcessor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (FMPluginEditor)
};

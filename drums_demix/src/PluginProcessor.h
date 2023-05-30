/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

// not this:
//#include <JuceHeader.h>
// but this:


#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_analytics/juce_analytics.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_events/juce_events.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_audio_plugin_client/juce_audio_plugin_client.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_box2d/juce_box2d.h>
#include <juce_core/juce_core.h>
#include <juce_cryptography/juce_cryptography.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_opengl/juce_opengl.h>
#include <juce_osc/juce_osc.h>
#include <juce_product_unlocking/juce_product_unlocking.h>
#include <juce_video/juce_video.h>


//==============================================================================
/**
*/
class DrumsDemixProcessor  : public juce::AudioProcessor
                            #if JucePlugin_Enable_ARA
                             , public juce::AudioProcessorARAExtension
                            #endif
{
public:
    //==============================================================================
    DrumsDemixProcessor();
    ~DrumsDemixProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    /** add some midi to be played at the sent sample offset*/
    void addMidi(juce::MidiMessage msg, int sampleOffset);
    /** set env length in seconds*/
    void setEnvLength(double envLengthSecs);
    
    juce::AudioTransportSource transportProcessor;
    juce::AudioTransportSource transportProcessorKick;
    juce::AudioTransportSource transportProcessorSnare;
    juce::AudioTransportSource transportProcessorToms;
    juce::AudioTransportSource transportProcessorHihat;
    juce::AudioTransportSource transportProcessorCymbals;

    bool playInput{ false };
    bool playKick {false};
    bool playSnare{ false };
    bool playToms{ false };
    bool playHihat{ false };
    bool playCymbals{ false };
private:



      //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumsDemixProcessor)
};

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
#include "ClickableArea.h"



//==============================================================================
/**
*/



class DrumsDemixEditor  : public juce::AudioProcessorEditor,
                          // listen to buttons
                          public juce::Button::Listener,
                          // listen to AudioThumbnail
                          public juce::ChangeListener,
                          public juce::FileDragAndDropTarget,
                          private juce::Timer
                          

{
public:
    DrumsDemixEditor (DrumsDemixProcessor&);
    ~DrumsDemixEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;

    void buttonClicked(juce::Button* btn) override;

    juce::AudioBuffer<float> getAudioBufferFromFile(juce::File file);
    
    //juce::File Absolute = juce::File("/Users/alessandroorsatti/Documents/GitHub/DrumsDemix/drums_demix");
    juce::File absolutePath = juce::File::getCurrentWorkingDirectory().getParentDirectory();
    //juce::String Path = Absolute.getFullPathName();



    
    //VISUALIZER
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void thumbnailChanged();
    
    void displayOut(juce::AudioBuffer<float>& buffer, juce::AudioThumbnail& thumbnailOut);

    void paintIfNoFileLoaded(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, at::string Phrase);

    void paintIfFileLoaded(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color);

    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

    void loadFile(const juce::String& path);

    //MODEL INFERENCE
    void InferModels(std::vector<torch::jit::IValue> my_input, torch::Tensor phase, int size);

    //CREATE WAV
    void CreateWavQuick(torch::Tensor yKickTensor, juce::String path, juce::String name); 
    void CreateWav(std::vector<at::Tensor> tList, juce::String name);



private:

    juce::String inputFileName;

    juce::File docsDir;
    juce::File filesDir;


    enum TransportState
    {
      Stopped,
      Starting,
      Stopping,
      Playing
    };

    TransportState state;
    
    juce::Image background;
    
    juce::ImageComponent imageKit;
    juce::ImageComponent imageKick;
    juce::ImageComponent imageSnare;
    juce::ImageComponent imageToms;
    juce::ImageComponent imageHihat;
    juce::ImageComponent imageCymbals;

    //buttons
    juce::TextButton testButton;
    juce::TextButton openButton;
    juce::TextButton playButton;
    juce::TextButton stopButton;
    juce::TextButton downloadKickButton;
    
    juce::TextButton playKickButton;
    juce::TextButton stopKickButton;
    
    juce::TextButton playSnareButton;
    juce::TextButton stopSnareButton;
    
    juce::TextButton playTomsButton;
    juce::TextButton stopTomsButton;
    
    juce::TextButton playHihatButton;
    juce::TextButton stopHihatButton;
    
    juce::TextButton playCymbalsButton;
    juce::TextButton stopCymbalsButton;
    
    juce::ImageComponent kickImage;
    juce::ImageComponent snareImage;
    juce::ImageComponent tomsImage;
    juce::ImageComponent hihatImage;
    juce::ImageComponent cymbalsImage;

    ClickableArea areaKick;
    ClickableArea areaSnare;
    ClickableArea areaToms;
    ClickableArea areaHihat;
    ClickableArea areaCymbals;

    ClickableArea areaFull;
    
    //VISUALIZER
    
    juce::AudioThumbnail* thumbnail;
    juce::AudioThumbnailCache* thumbnailCache;

    juce::AudioThumbnail* thumbnailKickOut;
    juce::AudioThumbnailCache* thumbnailCacheKickOut;
    
    juce::AudioThumbnail* thumbnailSnareOut;
    juce::AudioThumbnailCache* thumbnailCacheSnareOut;
    
    juce::AudioThumbnail* thumbnailTomsOut;
    juce::AudioThumbnailCache* thumbnailCacheTomsOut;
    
    juce::AudioThumbnail* thumbnailHihatOut;
    juce::AudioThumbnailCache* thumbnailCacheHihatOut;
    
    juce::AudioThumbnail* thumbnailCymbalsOut;
    juce::AudioThumbnailCache* thumbnailCacheCymbalsOut;

    
    //------------------------------------------------------------------------------------
    
    juce::AudioFormatManager formatManager;
    std::unique_ptr<juce::AudioFormatReaderSource> playSource;
    std::unique_ptr<juce::MemoryAudioSource> playSourceKick;
    std::unique_ptr<juce::MemoryAudioSource> playSourceSnare;
    std::unique_ptr<juce::MemoryAudioSource> playSourceToms;
    std::unique_ptr<juce::MemoryAudioSource> playSourceHihat;
    std::unique_ptr<juce::MemoryAudioSource> playSourceCymbals;
    juce::File myFile;
    juce::File myFileOut;
    void transportStateChanged(TransportState newState, juce::String id);

    juce::AudioBuffer<float> bufferY;
    juce::AudioBuffer<float> bufferOut;

    std::vector<float> audioPoints;
    
    //audioPoints.call_back(new float (args));
    bool paintOut{ false };

    void timerCallback() override
    {
        repaint();
    }
    
    //load TorchScript modules:
    torch::jit::script::Module mymoduleKick;
    torch::jit::script::Module mymoduleSnare;
    torch::jit::script::Module mymoduleToms;
    torch::jit::script::Module mymoduleHihat;
    torch::jit::script::Module mymoduleCymbals;

    //output tensors
    at::Tensor yKick;
    at::Tensor ySnare;
    at::Tensor yToms;
    at::Tensor yHihat;
    at::Tensor yCymbals;


    juce::Label textLabel;
    bool separated {false};

    
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    DrumsDemixProcessor& audioProcessor;
    

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (DrumsDemixEditor)
};



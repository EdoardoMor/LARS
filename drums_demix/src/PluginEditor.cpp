/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include <iostream>
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "Utils.cpp"


#include <torch/torch.h>
#include <torch/script.h>
#include <cmath>

//==============================================================================
FMPluginEditor::FMPluginEditor (FMPluginProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p), 
    miniPianoKbd{kbdState, juce::MidiKeyboardComponent::horizontalKeyboard} 

{    
    // listen to the mini piano
    kbdState.addListener(this);

    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (600, 500);

    addAndMakeVisible(envToggle);
    envToggle.addListener(this);


    addAndMakeVisible(miniPianoKbd);

    addAndMakeVisible(testButton);
    testButton.setButtonText("TEST");
    testButton.addListener(this);

    formatManager.registerBasicFormats();

    try{
        mymodule=torch::jit::load("../src/scripted_modules/my_scripted_module.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }
    

}

FMPluginEditor::~FMPluginEditor()
{
}

//==============================================================================
void FMPluginEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

   // g.setColour (juce::Colours::white);
   // g.setFont (15.0f);
   // g.drawFittedText ("Hello World!", getLocalBounds(), juce::Justification::centred, 1);
}

void FMPluginEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    float rowHeight = getHeight()/5; 
    envToggle.setBounds(0, 0, getWidth()/2, rowHeight);
    miniPianoKbd.setBounds(0, rowHeight * 3, getWidth(), rowHeight);
    testButton.setBounds(getWidth()/2, rowHeight * 4, getWidth()/2, getHeight()/5);

    
}

 void FMPluginEditor::sliderValueChanged (juce::Slider *slider)
{

}

juce::AudioBuffer<float> FMPluginEditor::getAudioBufferFromFile(juce::File file)
{
    //juce::AudioFormatManager formatManager - declared in header...`;
    auto* reader = formatManager.createReaderFor(file);
    juce::AudioBuffer<float> audioBuffer;
    audioBuffer.setSize(reader->numChannels, reader->lengthInSamples);
    reader->read(&audioBuffer, 0, reader->lengthInSamples, 0, true, true);
    delete reader;
    return audioBuffer;

}

void FMPluginEditor::buttonClicked(juce::Button* btn)
{
    if (btn == &envToggle){
        double envLen = 0;
        if (envToggle.getToggleState()) { // one
            envLen = 1;
        }
        audioProcessor.setEnvLength(envLen);
    }

    if (btn == &testButton){

        Utils utils = Utils::Utils();

        //***TAKE THE INPUT FROM THE MIXED DRUMS FILE***


        //-From Wav to AufioBuffer
        juce::AudioBuffer<float> fileAudiobuffer = getAudioBufferFromFile(juce::File({"C:/POLIMI/MAE_Capstone/audio/1_funk-groove1_138_beat_4-4_socal.wav"}));

        DBG("number of samples, audiobuffer");
        DBG(fileAudiobuffer.getNumSamples());

        const float* readPointer1 = fileAudiobuffer.getReadPointer(0);
        const float* readPointer2 = fileAudiobuffer.getReadPointer(1);


        auto options = torch::TensorOptions().dtype(torch::kFloat32);


        //-From a stereo AudioBuffer to a 2D Tensor
        torch::Tensor fileTensor1 = torch::from_blob((float*)readPointer1, { 1, fileAudiobuffer.getNumSamples() }, options);
        torch::Tensor fileTensor2 = torch::from_blob((float*)readPointer2, { 1, fileAudiobuffer.getNumSamples() }, options);

        torch::Tensor fileTensor = torch::cat({ fileTensor1, fileTensor2 }, 0);
        DBG("audio tensor dim 0");
        DBG(fileTensor.sizes()[0]);

        DBG("audio tensor dim 1");
        DBG(fileTensor.sizes()[1]);

        //-Compute STFT

        torch::Tensor stftFilePhase;

        //need to pass the stftPhase tensor in order to have it back in return
        torch::Tensor stftFileMag = utils.batch_stft(fileTensor, stftFilePhase);

        stftFileMag = torch::unsqueeze(stftFileMag, 0);

        //stftFilePhase = torch::unsqueeze(stftFilePhase, 0);

        DBG("stftFileMag sizes: ");
        DBG(stftFileMag.sizes()[0]);
        DBG(stftFileMag.sizes()[1]);
        DBG(stftFileMag.sizes()[2]);
        DBG(stftFileMag.sizes()[3]);

        DBG("stftFilePhase sizes: ");
        DBG(stftFilePhase.sizes()[0]);
        DBG(stftFilePhase.sizes()[1]);
        DBG(stftFilePhase.sizes()[2]);
        //DBG(stftFilePhase.sizes()[3]);


        //-From stft Tensor to IValue
        std::vector<torch::jit::IValue> my_input;
        my_input.push_back(stftFileMag);


        //***INFER THE MODEL***


        //-Forward
        at::Tensor outputs = mymodule.forward(my_input).toTensor();

        //-Need another dimension to do batch_istft
        outputs = torch::squeeze(outputs, 0);
        DBG("outputs sizes: ");
        DBG(outputs.sizes()[0]);
        DBG(outputs.sizes()[1]);
        DBG(outputs.sizes()[2]);
        //DBG(outputs.sizes()[3]);
        

        //-Compute ISTFT

        at::Tensor y = utils.batch_istft(outputs, stftFilePhase, fileTensor.sizes()[1]);

        DBG("outputTensor sizes: ");
        DBG(y.sizes()[0]);
        DBG(y.sizes()[1]);

        //***CREATE A STEREO, AUDIBLE OUTPUT***

        //-Split output tensor in Left & Right
        torch::autograd::variable_list ySplit = torch::split(y, 1);
        at::Tensor yL = ySplit[0];
        at::Tensor yR = ySplit[1];

        

        DBG("yL sizes: ");
        DBG(yL.sizes()[0]);
        DBG(yL.sizes()[1]);

        DBG("yR sizes: ");
        DBG(yR.sizes()[0]);
        DBG(yR.sizes()[1]);


        //-Make a std vector for every channel (L & R)
        yL = yL.contiguous();
        std::vector<float> vectoryL(yL.data_ptr<float>(), yL.data_ptr<float>() + yL.numel());

        yR = yR.contiguous();
        std::vector<float> vectoryR(yR.data_ptr<float>(), yR.data_ptr<float>() + yR.numel());

        //-Create an array of 2 float pointers from the 2 std vectors 
        float* dataPtrs[2];
        dataPtrs[0] = { vectoryL.data() };
        dataPtrs[1] = { vectoryR.data() };


        //-Create the stereo AudioBuffer
        juce::AudioBuffer<float> bufferY = juce::AudioBuffer<float>(dataPtrs, 2, 1377648); //need to change last argument to let it be dynamic!

        //-Print Wav
        juce::WavAudioFormat formatWav;
        std::unique_ptr<juce::AudioFormatWriter> writerY;
        writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File({"C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/testWavJuce4.wav"})),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));
        if (writerY != nullptr)
            writerY->writeFromAudioSampleBuffer (bufferY, 0, bufferY.getNumSamples());

       

        DBG("wav scritto!");

        


    }

}

void FMPluginEditor::handleNoteOn(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity)
{
    juce::MidiMessage msg1 = juce::MidiMessage::noteOn(midiChannel, midiNoteNumber, velocity);
    audioProcessor.addMidi(msg1, 0);
    
}

void FMPluginEditor::handleNoteOff(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity)
{
    juce::MidiMessage msg2 = juce::MidiMessage::noteOff(midiChannel, midiNoteNumber, velocity);
    audioProcessor.addMidi(msg2, 0); 
}

/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include <iostream>
#include "PluginProcessor.h"
#include "PluginEditor.h"


#include <torch/torch.h>
#include <torch/script.h>

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

    /*test
    torch::Tensor tensor = torch::rand({1, 1});
    std::cout << "prova" << std::endl;
    std::cout << "JUCE and torch " << tensor << std::endl;
    DBG(tensor[0].item<float>());
    */

    try{
        mymodule=torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/testplugin/python_models/my_scripted_module.pt");
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
        /*float in = (float) trainKnob.getValue();
        float out1 = (float)modIndexSlider.getValue() / modIndexSlider.getMaximum();
        float out2 = (float)modDepthSlider.getValue() / modDepthSlider.getMaximum();
        nn.addTrainingData({in},{out1, out2});
        */
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({28*28}));

       at::Tensor outputs = mymodule.forward(inputs).toTensor();
       std::vector<float> v(outputs.data_ptr<float>(), outputs.data_ptr<float>() + outputs.numel()); //conversione da tensor a std vector
       DBG(v[0]);

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



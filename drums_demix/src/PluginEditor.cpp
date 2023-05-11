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
DrumsDemixEditor::DrumsDemixEditor (DrumsDemixProcessor& p)
    : AudioProcessorEditor (&p), miniPianoKbd{kbdState, juce::MidiKeyboardComponent::horizontalKeyboard}, formatManager(), thumbnailCache {5},thumbnail {512, formatManager, thumbnailCache},  audioProcessor (p)

{    
    // listen to the mini piano
    kbdState.addListener(this);

    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (600, 500);

    //addAndMakeVisible(envToggle);
    //envToggle.addListener(this);


    //addAndMakeVisible(miniPianoKbd);

    addAndMakeVisible(testButton);
    testButton.setButtonText("TEST");
    testButton.setEnabled(false);
    testButton.addListener(this);

    addAndMakeVisible(playButton);
    playButton.setButtonText("PLAY");
    playButton.setEnabled(false);
    playButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    playButton.addListener(this);

    addAndMakeVisible(stopButton);
    stopButton.setButtonText("STOP");
    stopButton.setEnabled(false);
    stopButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopButton.addListener(this);

    addAndMakeVisible(openButton);
    openButton.setButtonText("OPEN");
    openButton.addListener(this);

    formatManager.registerBasicFormats();
    
    //VISUALIZER
    thumbnail.addChangeListener(this);
        

    try{
        mymodule=torch::jit::load("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }
    

}

DrumsDemixEditor::~DrumsDemixEditor()
{
}

//==============================================================================
void DrumsDemixEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));

   // g.setColour (juce::Colours::white);
   // g.setFont (15.0f);
   // g.drawFittedText ("Hello World!", getLocalBounds(), juce::Justification::centred, 1);
    
    //VISUALIZER
    juce::Rectangle<int> thumbnailBounds (10, 10, getWidth() - 20, getHeight() - 150);
    
           if (thumbnail.getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBounds);
           else
               paintIfFileLoaded (g, thumbnailBounds);
    
}

void DrumsDemixEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    float rowHeight = getHeight()/5; 
    //envToggle.setBounds(0, 0, getWidth()/2, rowHeight);
    //miniPianoKbd.setBounds(0, rowHeight * 3, getWidth(), rowHeight);
    testButton.setBounds(getWidth()/2, rowHeight * 4, getWidth()/2, getHeight()/5);
    openButton.setBounds(0, rowHeight * 4, getWidth()/2, getHeight()/5);
    playButton.setBounds(0, rowHeight * 4-getHeight()/12, getWidth()/2, getHeight()/12);
    stopButton.setBounds(getWidth()/2, rowHeight * 4-getHeight()/12, getWidth()/2, getHeight()/12);

    
}

 void DrumsDemixEditor::sliderValueChanged (juce::Slider *slider)
{

}

juce::AudioBuffer<float> DrumsDemixEditor::getAudioBufferFromFile(juce::File file)
{
    //juce::AudioFormatManager formatManager - declared in header...`;
    auto* reader = formatManager.createReaderFor(file);
    juce::AudioBuffer<float> audioBuffer;
    audioBuffer.setSize(reader->numChannels, reader->lengthInSamples);
    reader->read(&audioBuffer, 0, reader->lengthInSamples, 0, true, true);
    delete reader;
    return audioBuffer;

}

void DrumsDemixEditor::buttonClicked(juce::Button* btn)
{
    if (btn == &envToggle){
        double envLen = 0;
        if (envToggle.getToggleState()) { // one
            envLen = 1;
        }
        audioProcessor.setEnvLength(envLen);
    }

    if (btn == &testButton){


        //***TAKE THE INPUT FROM THE MIXED DRUMS FILE***


        //-From Wav to AudiofileBuffer

        Utils utils = Utils::Utils();
        juce::AudioBuffer<float> fileAudiobuffer = getAudioBufferFromFile(myFile);

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
        juce::AudioBuffer<float> bufferY = juce::AudioBuffer<float>(dataPtrs, 2, y.sizes()[1]); //need to change last argument to let it be dynamic!

        //-Print Wav
        juce::WavAudioFormat formatWav;
        std::unique_ptr<juce::AudioFormatWriter> writerY;
        writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/testWavJuce5.wav")),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));
        if (writerY != nullptr)
            writerY->writeFromAudioSampleBuffer (bufferY, 0, bufferY.getNumSamples());

       

        DBG("wav scritto!");
        
        //VISUALIZER
        thumbnail.setSource (new juce::FileInputSource (myFile));
      
        


    }
    if (btn == &openButton) {

        juce::FileChooser chooser("Choose a Wav or Aiff File", juce::File::getSpecialLocation(juce::File::userDesktopDirectory), "*.wav;*.aiff;*.mp3");

        if (chooser.browseForFileToOpen())
        {
            //juce::File myFile;
            myFile = chooser.getResult();
            juce::AudioFormatReader* reader = formatManager.createReaderFor(myFile);

            if (reader != nullptr)
            {

                std::unique_ptr<juce::AudioFormatReaderSource> tempSource(new juce::AudioFormatReaderSource(reader, true));

                //audioProcessor.transportProcessor.setSource(tempSource.get());
                //transportStateChanged(Stopped);

                playSource.reset(tempSource.release());
                DBG("IFopenbuttonclicked");

            }
            DBG("openbuttonclicked");
            testButton.setEnabled(true);
            playButton.setEnabled(true);



        }
    }
    if (btn == &playButton){
        DBG("playbuttonclicked");
        playButton.setEnabled(false);
        stopButton.setEnabled(true);
        

    }
    if (btn == &stopButton){
        DBG("stopbuttonclicked");
        playButton.setEnabled(true);
        stopButton.setEnabled(false);

    }


}

void DrumsDemixEditor::handleNoteOn(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity)
{
    juce::MidiMessage msg1 = juce::MidiMessage::noteOn(midiChannel, midiNoteNumber, velocity);
    audioProcessor.addMidi(msg1, 0);
    
}

void DrumsDemixEditor::handleNoteOff(juce::MidiKeyboardState *source, int midiChannel, int midiNoteNumber, float velocity)
{
    juce::MidiMessage msg2 = juce::MidiMessage::noteOff(midiChannel, midiNoteNumber, velocity);
    audioProcessor.addMidi(msg2, 0); 
}




//VISUALIZER
void DrumsDemixEditor::changeListenerCallback (juce::ChangeBroadcaster* source)
  {
      if (source == &thumbnail)       repaint();
  }

void DrumsDemixEditor::paintIfNoFileLoaded (juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds)
  {
      g.setColour (juce::Colours::darkgrey);
      g.fillRect (thumbnailBounds);
      g.setColour (juce::Colours::white);
      g.drawFittedText ("No File Loaded", thumbnailBounds, juce::Justification::centred, 1);
  }

void DrumsDemixEditor::paintIfFileLoaded (juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds)
 {
     g.setColour (juce::Colours::white);
     g.fillRect (thumbnailBounds);

     g.setColour (juce::Colours::red);                               // [8]

     thumbnail.drawChannels (g,                                      // [9]
                             thumbnailBounds,
                             0.0,                                    // start time
                             thumbnail.getTotalLength(),             // end time
                             1.0f);                                  // vertical zoom
 }


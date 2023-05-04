/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include <iostream>
#include "PluginProcessor.h"
#include "PluginEditor.h"
#include "Utils.cpp"
#include "cpp-create-wav.cpp"


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

    /*test
    torch::Tensor tensor = torch::rand({1, 1});
    std::cout << "prova" << std::endl;
    std::cout << "JUCE and torch " << tensor << std::endl;
    DBG(tensor[0].item<float>());
    */

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

       /*
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({28*28}));

       at::Tensor outputs = mymodule.forward(inputs).toTensor();
       std::vector<float> v(outputs.data_ptr<float>(), outputs.data_ptr<float>() + outputs.numel()); //conversione da tensor a std vector
       DBG(v[0]);

       */

        Utils utils = Utils::Utils();
        juce::AudioBuffer<float> fileAudiobuffer = getAudioBufferFromFile(juce::File({"C:/POLIMI/MAE_Capstone/audio/1_funk-groove1_138_beat_4-4_socal.wav"}));

        DBG("number of samples, audiobuffer");
        DBG(fileAudiobuffer.getNumSamples());

        const float* readPointer1 = fileAudiobuffer.getReadPointer(0);
        const float* readPointer2 = fileAudiobuffer.getReadPointer(1);


        auto options = torch::TensorOptions().dtype(torch::kFloat32);

        torch::Tensor fileTensor1 = torch::from_blob((float*)readPointer1, { 1, fileAudiobuffer.getNumSamples() }, options);
        torch::Tensor fileTensor2 = torch::from_blob((float*)readPointer2, { 1, fileAudiobuffer.getNumSamples() }, options);

        torch::Tensor fileTensor = torch::cat({ fileTensor1, fileTensor2 }, 0);
        DBG("audio tensor dim 0");
        DBG(fileTensor.sizes()[0]);

        DBG("audio tensor dim 1");
        DBG(fileTensor.sizes()[1]);



        /*
        torch::Tensor stftFile = utils._stft(fileTensor);

        DBG("STFT rows");
        DBG(stftFile.sizes()[0]);

        DBG("STFT columns");
        DBG(stftFile.sizes()[1]);

        DBG("STFT dim 2");
        DBG(stftFile.sizes()[2]);

        DBG("STFT dim 3");
        DBG(stftFile.sizes()[3]);

        */



        //std::cout << fileTensor.sizes() << std::endl;

        //std::tuple<torch::Tensor, torch::Tensor> [stftFileMag, stftFilePhase]  = utils.batch_stft(fileTensor);

        torch::Tensor stftFilePhase;

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

        //std::vector<float> fileInput(stftFile.data_ptr<float>(), stftFile.data_ptr<float>() + stftFile.numel());

        //auto torch_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        //torch::Tensor my_tensor = torch::from_blob(
          //my_array.begin(),
          //{3, 20},
          //torch_options); 

        std::vector<torch::jit::IValue> my_input;
        //my_input.push_back(torch::ones({1, 2, 2049, 1347}));
        my_input.push_back(stftFileMag);

        //c10::IValue mymodForward = mymodule.forward(my_input);

        //DBG(mymodForward.toInt());

        at::Tensor outputs = mymodule.forward(my_input).toTensor();

        outputs = torch::squeeze(outputs, 0);
        DBG("outputs sizes: ");
        DBG(outputs.sizes()[0]);
        DBG(outputs.sizes()[1]);
        DBG(outputs.sizes()[2]);
        //DBG(outputs.sizes()[3]);
        

        /*

        at::Tensor masked = torch::mul(outputs, stftFileMag);

        DBG("masked sizes: ");
        DBG(masked.sizes()[0]);
        DBG(masked.sizes()[1]);
        DBG(masked.sizes()[2]);
        DBG(masked.sizes()[3]);

        */

        

        at::Tensor y = utils.batch_istft(outputs, stftFilePhase, fileTensor.sizes()[1]);

        DBG("outputTensor sizes: ");
        DBG(y.sizes()[0]);
        DBG(y.sizes()[1]);

        torch::save(y, "test2.pt");

        /*

        std::vector<double> yvec( 100000 );
        FillSin(yvec);
        
        for( int n=0; n < yvec.size(); ++n )
        {
            DBG(yvec[n]);
        }

        auto options2 = torch::TensorOptions().dtype(torch::kFloat32);

        torch::Tensor y = torch::from_blob(yvec.data(), {1, 100000}, options2);
        */


        

        //write_binary(y);
        //y = y.view(tensor.size(0), -1); //Here the shape of the data is batch, num_tensors

        //write_wav();

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

void FMPluginEditor::write_binary(at::Tensor t)
{
    t = t.contiguous();

    int sizesMul = t.sizes()[0] * t.sizes()[1];
    //std::vector<float> vectorT(t.data_ptr<float>(), t.data_ptr<float>()+t.numel());

    const float* vectorT = t.data_ptr<float>();

    /*
    int nx = 10, ny = 10;

    long double *buff1 = new long double[nx * ny];
    long double *buff2 = new long double[nx * ny];

    long double **data = new long double *[nx];
    long double **data_read = new long double *[nx];
    

    for (int i = 0; i < nx; i++)
    {
        data[i] = buff1 + (i*ny);
        data_read[i] = buff2 + (i*ny);
    }
    

    data[4][4] = 10.0;
    printf("%LF\n", data[4][4]);

    */

    FILE *file = fopen("test.bin", "wb");
    fwrite(vectorT, sizeof(*vectorT), sizesMul, file);
    fclose(file);

    DBG("bin scritto!");


    // delete pointer arrays
   // delete [] vectorT;
}

void FMPluginEditor::FillSin( std::vector<double>& v )
{
    static const double PI = 4*atan(1.0);
 
    for( int n=0; n < v.size(); ++n )
    {
        v[ n ] = sin( ( 2 * PI / n ) * 440);
        //v[n] = n;
        //DBG(v[n]);
    }
}
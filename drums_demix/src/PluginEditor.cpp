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
#include <chrono>
#include <thread>

//==============================================================================
DrumsDemixEditor::DrumsDemixEditor (DrumsDemixProcessor& p)
    : AudioProcessorEditor (&p), miniPianoKbd{kbdState, juce::MidiKeyboardComponent::horizontalKeyboard}, formatManager(), thumbnailCache {5},thumbnail {512, formatManager, thumbnailCache},thumbnailCacheOut {5},thumbnailOut {512, formatManager, thumbnailCacheOut},  audioProcessor (p), state(Stopped)

{    
    // listen to the mini piano
    kbdState.addListener(this);

    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (1000, 500);

    //addAndMakeVisible(envToggle);
    //envToggle.addListener(this);


    //addAndMakeVisible(miniPianoKbd);

    addAndMakeVisible(testButton);
    testButton.setButtonText("SEPARATE");
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
    openButton.setButtonText("LOAD A FILE");
    openButton.addListener(this);

    formatManager.registerBasicFormats();
    audioProcessor.transportProcessor.addChangeListener(this);
    
    //VISUALIZER
    thumbnail.addChangeListener(this);
    thumbnailOut.addChangeListener(this);    
        

    try{
        mymoduleKick=torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_kick.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }
    

    try{
        mymoduleSnare=torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_snare.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

    try{
        mymoduleToms=torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_toms.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

    try{
        mymoduleHihat=torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_hihat.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

    try{
        mymoduleCymbals=torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_Cymbals.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

}

DrumsDemixEditor::~DrumsDemixEditor()
{
}

//==============================================================================
void DrumsDemixEditor::paint(juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    g.setColour(juce::Colours::white);

    if (paintOut)
    {

        juce::Path p;
        auto ratio = bufferOut.getNumSamples() / getWidth();
        const float* buffer = bufferOut.getReadPointer(0);
        DBG("entrato");

        for (int sample = 0; sample < bufferOut.getNumSamples(); sample += ratio)
        {
            DBG(buffer[sample]);
            audioPoints.push_back(buffer[sample]);
        }
        DBG(audioPoints.size());

        DBG("uscito");

        p.startNewSubPath(10, (60 + ((getHeight() - 200) / 2) + ((getHeight() - 200) / 2) / 2));

        for (int sample = 0; sample < audioPoints.size(); ++sample)
        {
            auto point = juce::jmap<float>(audioPoints[sample], -1.0f, 1.0f, (60 + (getHeight() - 200) / 2) + ((getHeight() - 200) / 2) / 2 + 100, (60 + (getHeight() - 200) / 2) + ((getHeight() - 200) / 2) / 2 - 100);
            //auto point = juce::jmap<float>(audioPoints[sample], -1.0f, 1.0f, (20 + (getHeight() - 200) / 2) + ((getHeight() - 200) / 2) / 2 + (((getHeight() - 200) / 2)), (((getHeight() - 200) / 2) / 2) + 40);
            p.lineTo(sample, point);

        }

        g.strokePath(p, juce::PathStrokeType(2));
        paintOut = false;
    }

   // g.setColour (juce::Colours::white);
   // g.setFont (15.0f);
   // g.drawFittedText ("Hello World!", getLocalBounds(), juce::Justification::centred, 1);
    
    //VISUALIZER
    juce::Rectangle<int> thumbnailBounds (10, (getHeight() / 9)+10, getWidth() - 220, (getHeight() - 200)/4);
    
           if (thumbnail.getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBounds);
           else
               paintIfFileLoaded (g, thumbnailBounds, thumbnail);
        
    //juce::Rectangle<int> thumbnailBoundsOut (10, 20+(getHeight() - 200)/2, getWidth() - 20, (getHeight() - 200)/2);
    //
    //       if (thumbnailOut.getNumChannels() == 0)
    //           paintIfNoFileLoaded (g, thumbnailBoundsOut);
    //       else
    //           paintIfFileLoaded (g, thumbnailBoundsOut, thumbnailOut);
    
}

void DrumsDemixEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    float rowHeight = getHeight()/5; 
    //envToggle.setBounds(0, 0, getWidth()/2, rowHeight);
    //miniPianoKbd.setBounds(0, rowHeight * 3, getWidth(), rowHeight);
    testButton.setBounds(getWidth()/2,0, getWidth()/2, getHeight()/9);
    openButton.setBounds(0,0, getWidth()/2, getHeight()/9);
    playButton.setBounds(getWidth() - 220 +20, getHeight() / 9 +10, (getHeight() - 200) / 4, (getHeight() - 200) / 4);
    stopButton.setBounds(getWidth() - 220 + 30 + (getHeight() - 200) / 4, getHeight() / 9 +10, (getHeight() - 200) / 4, (getHeight() - 200) / 4);

    
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

    if (btn == &testButton) {


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


        InferModels(my_input, stftFilePhase, fileTensor.sizes()[1]);




        //***CREATE A STEREO, AUDIBLE OUTPUT***

        at::Tensor yOut = yKick.contiguous();
        std::vector<float> vectoryOutprova(yOut.data_ptr<float>(), yOut.data_ptr<float>() + yOut.numel());
        

        for (int sample = 0; sample < vectoryOutprova.size(); sample++)
        {
            ///DBG(buffer[sample]);
            vectoryOut.push_back(vectoryOutprova[sample]);
        }

        // DECOMMENT IF USING ONLY THE KICK FOR QUICK DEBUGGING
        CreateWavQuick(yKick);

        // DECOMMENT IF USING ALL THE DRUMS
        //std::vector<at::Tensor> tensorList = {yKick, ySnare, yToms, yHihat, yCymbals};
        //CreateWav(tensorList);

        
        
     
        


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

                audioProcessor.transportProcessor.setSource(tempSource.get());
                transportStateChanged(Stopped);

                playSource.reset(tempSource.release());
                DBG("IFopenbuttonclicked");

            }
            DBG("openbuttonclicked");
            testButton.setEnabled(true);
            playButton.setEnabled(true);



        }
        //VISUALIZER
        thumbnail.setSource(new juce::FileInputSource(myFile));
    }
    if (btn == &playButton){
        transportStateChanged(Starting);
        DBG("playbuttonclicked");
        //playButton.setEnabled(false);
        //stopButton.setEnabled(true);
        

    }
    if (btn == &stopButton){
        transportStateChanged(Stopping);
        DBG("stopbuttonclicked");
        //playButton.setEnabled(true);
        //stopButton.setEnabled(false);

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

void DrumsDemixEditor::transportStateChanged(TransportState newState)
{
  if(newState != state)
  {
    state = newState;

    switch (state)
    {
    case Stopped:
        audioProcessor.transportProcessor.setPosition(0.0);
      break;
    case Starting:
      stopButton.setEnabled(true);
      playButton.setEnabled(false);
      audioProcessor.transportProcessor.start();
      break;
    case Playing:
      stopButton.setEnabled(true);
      break;
    case Stopping:
      stopButton.setEnabled(false);
      playButton.setEnabled(true);
      audioProcessor.transportProcessor.stop();
      break;
    }
  }
}

void DrumsDemixEditor::displayOut(juce::File file)
{
    DBG(file.getFileName());
    std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    thumbnailOut.setSource(new juce::FileInputSource(file));
}




//VISUALIZER
void DrumsDemixEditor::changeListenerCallback (juce::ChangeBroadcaster* source)
  {
    if (source == &thumbnail) { repaint(); }
    if (source == &thumbnailOut) { repaint(); }
    if(source == &audioProcessor.transportProcessor)
    {
        if(audioProcessor.transportProcessor.isPlaying())
        {
        transportStateChanged(Playing);
        }
        else
        {
        transportStateChanged(Stopped);
        }
    }
  }

void DrumsDemixEditor::paintIfNoFileLoaded (juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds)
  {
      g.setColour (juce::Colours::darkgrey);
      g.fillRect (thumbnailBounds);
      g.setColour (juce::Colours::white);
      g.drawFittedText ("Drop a file or load it", thumbnailBounds, juce::Justification::centred, 1);
  }

void DrumsDemixEditor::paintIfFileLoaded (juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav)
 {
     g.setColour (juce::Colours::white);
     g.fillRect (thumbnailBounds);

     g.setColour (juce::Colours::red);                               // [8]

     thumbnailWav.drawChannels (g,                                      // [9]
                             thumbnailBounds,
                             0.0,                                    // start time
                             thumbnailWav.getTotalLength(),             // end time
                             1.0f);                                  // vertical zoom
 }

bool DrumsDemixEditor::isInterestedInFileDrag(const juce::StringArray& files)
{
    for (auto file : files)
    {
        if (file.contains(".wav") || file.contains(".mp3") || file.contains(".aiff"))
        {
            return true;
        }
    }

    return false;
}

void DrumsDemixEditor::filesDropped(const juce::StringArray& files, int x, int y)
{
    for (auto file : files)
    {
        if (isInterestedInFileDrag(files)) 
        {
            loadFile(file);

        }
    }
    repaint();

}

void DrumsDemixEditor::loadFile(const juce::String& path)
{


    auto file = juce::File(path);
    myFile = file;
    juce::AudioFormatReader* reader = formatManager.createReaderFor(file);
    if (reader != nullptr)
    {

        std::unique_ptr<juce::AudioFormatReaderSource> tempSource(new juce::AudioFormatReaderSource(reader, true));

        audioProcessor.transportProcessor.setSource(tempSource.get());
        transportStateChanged(Stopped);

        playSource.reset(tempSource.release());
        DBG("IFopenbuttonclicked");

    }
    DBG("openbuttonclicked");
    testButton.setEnabled(true);
    playButton.setEnabled(true);

    thumbnail.setSource(new juce::FileInputSource(file));

}

void DrumsDemixEditor::InferModels(std::vector<torch::jit::IValue> my_input, torch::Tensor phase, int size) 
{
    DBG("Infering the Models...");
    Utils utils = Utils::Utils();
    //***INFER THE MODEL***


        //-Forward
        at::Tensor outputsKick = mymoduleKick.forward(my_input).toTensor();

        // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
        //at::Tensor outputsSnare = mymoduleSnare.forward(my_input).toTensor();
        //at::Tensor outputsToms = mymoduleToms.forward(my_input).toTensor();
        //at::Tensor outputsHihat = mymoduleHihat.forward(my_input).toTensor();
        //at::Tensor outputsCymbals = mymoduleCymbals.forward(my_input).toTensor();

        //-Need another dimension to do batch_istft
        outputsKick = torch::squeeze(outputsKick, 0);

        DBG("outputs sizes: ");
        DBG(outputsKick.sizes()[0]);
        DBG(outputsKick.sizes()[1]);
        DBG(outputsKick.sizes()[2]);
        //DBG(outputs.sizes()[3]);


        // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
        //outputsSnare = torch::squeeze(outputsSnare, 0);
        //outputsToms = torch::squeeze(outputsToms, 0);
        //outputsHihat = torch::squeeze(outputsHihat, 0);
        //outputsCymbals = torch::squeeze(outputsCymbals, 0);
        

        //-Compute ISTFT

        yKick = utils.batch_istft(outputsKick, phase, size);

        DBG("y tensor sizes: ");
        DBG(yKick.sizes()[0]);
        DBG(yKick.sizes()[1]);


        // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
        
        //ySnare = utils.batch_istft(outputsSnare, phase, size);
        //yToms = utils.batch_istft(outputsToms, phase, size);
        //yHihat = utils.batch_istft(outputsHihat, phase, size);
        //yCymbals = utils.batch_istft(outputsCymbals, phase, size);
        


        /* RELOADARE I MODELLI E' UN MODO PER NON FAR CRASHARE AL SECONDO SEPARATE CONSECUTIVO, MA FORSE NON IL MIGLIOR MODO! (RALLENTA UN PO')

        try {
            mymoduleKick = torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_kick.pt");
        }
        catch (const c10::Error& e) {
            DBG("error"); //indicate error to calling code
        }


        try {
            mymoduleSnare = torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_snare.pt");
        }
        catch (const c10::Error& e) {
            DBG("error"); //indicate error to calling code
        }

        try {
            mymoduleToms = torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_toms.pt");
        }
        catch (const c10::Error& e) {
            DBG("error"); //indicate error to calling code
        }

        try {
            mymoduleHihat = torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_hihat.pt");
        }
        catch (const c10::Error& e) {
            DBG("error"); //indicate error to calling code
        }

        try {
            mymoduleCymbals = torch::jit::load("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/src/scripted_modules/my_scripted_module_Cymbals.pt");
        }
        catch (const c10::Error& e) {
            DBG("error"); //indicate error to calling code
        }

        */

}

void DrumsDemixEditor::CreateWav(std::vector<at::Tensor> tList)
{
    for(at::Tensor yInstr : tList) {

        DBG("y sizes: ");
        DBG(yInstr.sizes()[0]);
        DBG(yInstr.sizes()[1]);


         //-Split output tensor in Left & Right
        torch::autograd::variable_list ySplit = torch::split(yInstr, 1);
        at::Tensor yL = ySplit[0];
        at::Tensor yR = ySplit[1];



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
        juce::AudioBuffer<float> bufferY = juce::AudioBuffer<float>(dataPtrs, 2, yInstr.sizes()[1]); //need to change last argument to let it be dynamic!

        //-Print Wav
        juce::WavAudioFormat formatWav;
        std::unique_ptr<juce::AudioFormatWriter> writerY;

        if(torch::equal(yInstr, yKick)) {
            writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/wavs/testWavJuceKick.wav")),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));

        }

        else if(torch::equal(yInstr, ySnare)) {
            writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/wavs/testWavJuceSnare.wav")),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));

        }

        else if(torch::equal(yInstr, yToms)){
            writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/wavs/testWavJuceToms.wav")),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));

        }

        else if(torch::equal(yInstr, yHihat)){
            writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/wavs/testWavJuceHihat.wav")),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));

        }

        else if(torch::equal(yInstr, yCymbals)){
            writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/wavs/testWavJuceCymbals.wav")),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));

        }


        if (writerY != nullptr)
            writerY->writeFromAudioSampleBuffer (bufferY, 0, bufferY.getNumSamples());

       

        DBG("wav scritto!");
    }
       

}

void DrumsDemixEditor::CreateWavQuick(torch::Tensor yKickTensor)
{


        DBG("y sizes: ");
        DBG(yKickTensor.sizes()[0]);
        DBG(yKickTensor.sizes()[1]);


         //-Split output tensor in Left & Right
        torch::autograd::variable_list ySplit = torch::split(yKickTensor, 1);
        at::Tensor yL = ySplit[0];
        at::Tensor yR = ySplit[1];



        //-Make a std vector for every channel (L & R)
        yL = yL.contiguous();
        std::vector<float> vectoryL(yL.data_ptr<float>(), yL.data_ptr<float>() + yL.numel());

        yR = yR.contiguous();
        std::vector<float> vectoryR(yR.data_ptr<float>(), yR.data_ptr<float>() + yR.numel());

        //-Create an array of 2 float pointers from the 2 std vectors 
        float* dataPtrs[2];
        dataPtrs[0] = { vectoryL.data() };
        dataPtrs[1] = { vectoryR.data() };

        float* dataPtrsOut[2];
        dataPtrsOut[0] = { vectoryOut.data() };
        dataPtrsOut[1] = { vectoryOut.data() };



        //-Create the stereo AudioBuffer
        juce::AudioBuffer<float> bufferY = juce::AudioBuffer<float>(dataPtrs, 2, yKickTensor.sizes()[1]); //need to change last argument to let it be dynamic!
        bufferOut = juce::AudioBuffer<float>(dataPtrsOut, 2, yKickTensor.sizes()[1]);

        //-Print Wav
        juce::WavAudioFormat formatWav;
        std::unique_ptr<juce::AudioFormatWriter> writerY;

        writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(juce::File("C:/POLIMI/MAE_Capstone/DrumsDemix/drums_demix/wavs/testWavJuceKick.wav")),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));

        

        if (writerY != nullptr){
            writerY->writeFromAudioSampleBuffer (bufferY, 0, bufferY.getNumSamples());
            paintOut = true;
            repaint();
        }

       

        DBG("wav scritto!");
    
       

}




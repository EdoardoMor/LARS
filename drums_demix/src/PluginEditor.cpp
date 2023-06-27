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
#include <chrono>

//==============================================================================
DrumsDemixEditor::DrumsDemixEditor (DrumsDemixProcessor& p)
    : AudioProcessorEditor (&p), formatManager(), audioProcessor(p), state(Stopped),
    areaKick{}, areaSnare{}, areaToms{}, areaHihat{}, areaCymbals{}, areaFull{}

{

    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    setSize (1000, 500);

    //create system directory
    docsDir = juce::File::getSpecialLocation(juce::File::userMusicDirectory);
    filesDir = juce::File(docsDir.getFullPathName() + "/DrumsDemixFilesToDrop");
    filesDir.createDirectory();
    DBG("the files you separate are in: ");
    DBG(filesDir.getFullPathName());

    modelsDir = juce::File(juce::File::getSpecialLocation(juce::File::userDesktopDirectory).getFullPathName() + "/DrumsDemixUtils/DrumsDemixModels");
    DBG("the models are in: ");
    DBG(modelsDir.getFullPathName());

    imagesDir = juce::File(juce::File::getSpecialLocation(juce::File::userDesktopDirectory).getFullPathName() + "/DrumsDemixUtils/DrumsDemixImages");
    DBG("the images are in: ");
    DBG(imagesDir.getFullPathName());



    areaKick.setFilesDir(filesDir);
    areaSnare.setFilesDir(filesDir);
    areaToms.setFilesDir(filesDir);
    areaHihat.setFilesDir(filesDir);
    areaCymbals.setFilesDir(filesDir);

    thumbnailCache = new juce::AudioThumbnailCache(5);
    thumbnail = new juce::AudioThumbnail(512, formatManager, *thumbnailCache);

    thumbnailCacheKickOut = new juce::AudioThumbnailCache(5);
    thumbnailKickOut = new juce::AudioThumbnail(512, formatManager, *thumbnailCacheKickOut);

    thumbnailCacheSnareOut = new juce::AudioThumbnailCache(5);
    thumbnailSnareOut = new juce::AudioThumbnail(512, formatManager, *thumbnailCacheSnareOut);

    thumbnailCacheTomsOut = new juce::AudioThumbnailCache(5);
    thumbnailTomsOut = new juce::AudioThumbnail(512, formatManager, *thumbnailCacheSnareOut);

    thumbnailCacheHihatOut = new juce::AudioThumbnailCache(5);
    thumbnailHihatOut = new juce::AudioThumbnail(512, formatManager, *thumbnailCacheSnareOut);

    thumbnailCacheCymbalsOut = new juce::AudioThumbnailCache(5);
    thumbnailCymbalsOut = new juce::AudioThumbnail(512, formatManager, *thumbnailCacheSnareOut);


    //a text label to print some stuff
    //addAndMakeVisible(textLabel);

    /*
    auto downloadIcon = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/download.png"));
    auto playIcon = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/play.png"));
    auto stopIcon = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/stop.png"));
    auto separate = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/SEPARATE.png"));
    */

    auto downloadIcon = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/download.png") );
    auto playIcon = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/play.png") );
    auto stopIcon = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/stop.png") );
    auto separate = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/SEPARATE.png") );

    addAndMakeVisible(testButton);
    testButton.setImages(false, true, true, separate, 1.0, juce::Colour(), separate, 0.5, juce::Colour(), separate, 0.8, juce::Colour(), 0);
    //testButton.setButtonText("SEPARATE");
    testButton.setEnabled(false);
    testButton.addListener(this);

    addAndMakeVisible(downloadKickButton);
    downloadKickButton.setImages(false,true,true,downloadIcon,1.0,juce::Colour(),downloadIcon,0.5,juce::Colour(),downloadIcon,0.8,juce::Colour(),0);
    downloadKickButton.setEnabled(true);
    downloadKickButton.addListener(this);


    addAndMakeVisible(downloadSnareButton);
    downloadSnareButton.setImages(false, true, true, downloadIcon, 1.0, juce::Colour(), downloadIcon, 0.5, juce::Colour(), downloadIcon, 0.8, juce::Colour(), 0);
    downloadSnareButton.setEnabled(true);
    downloadSnareButton.addListener(this);


    addAndMakeVisible(downloadTomsButton);
    downloadTomsButton.setImages(false, true, true, downloadIcon, 1.0, juce::Colour(), downloadIcon, 0.5, juce::Colour(), downloadIcon, 0.8, juce::Colour(), 0);
    downloadTomsButton.setEnabled(true);
    downloadTomsButton.addListener(this);

    addAndMakeVisible(downloadHihatButton);
    downloadHihatButton.setImages(false, true, true, downloadIcon, 1.0, juce::Colour(), downloadIcon, 0.5, juce::Colour(), downloadIcon, 0.8, juce::Colour(), 0);
    downloadHihatButton.setEnabled(true);
    downloadHihatButton.addListener(this);

    addAndMakeVisible(downloadCymbalsButton);
    downloadCymbalsButton.setImages(false, true, true, downloadIcon, 1.0, juce::Colour(), downloadIcon, 0.5, juce::Colour(), downloadIcon, 0.8, juce::Colour(), 0);
    downloadCymbalsButton.setEnabled(true);
    downloadCymbalsButton.addListener(this);

    addAndMakeVisible(playButton);
    //playButton.setButtonText("PLAY");
    //playButton.setEnabled(false);
    playButton.setImages(false, true, true, playIcon, 1.0, juce::Colour(), playIcon, 0.5, juce::Colour(), playIcon, 0.8, juce::Colour(), 0);
    //playButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    playButton.addListener(this);

    addAndMakeVisible(stopButton);
    //stopButton.setButtonText("STOP");
    //stopButton.setEnabled(false);
    stopButton.setImages(false, true, true, stopIcon, 1.0, juce::Colour(), stopIcon, 0.5, juce::Colour(), stopIcon, 0.8, juce::Colour(), 0);
    //stopButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopButton.addListener(this);

    //FULL DRUMS
    addAndMakeVisible(areaFull);
    areaFull.addListener(this);
    areaFull.setAlpha(0);
    areaFull.setName("areaFull");
    

    //auto kitImage = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/kit.png"));
    auto kitImage = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/kit.png") );
    imageKit.setImage(kitImage, juce::RectanglePlacement::stretchToFit);
    addAndMakeVisible(imageKit);
      
    //KICK
    addAndMakeVisible(playKickButton);
    //playKickButton.setButtonText("PLAY");
    //playKickButton.setEnabled(false);
    //playKickButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    playKickButton.setImages(false, true, true, playIcon, 1.0, juce::Colour(), playIcon, 0.5, juce::Colour(), playIcon, 0.8, juce::Colour(), 0);
    playKickButton.addListener(this);

    addAndMakeVisible(stopKickButton);
    //stopKickButton.setButtonText("STOP");
    //stopKickButton.setEnabled(false);
    //stopKickButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopKickButton.setImages(false, true, true, stopIcon, 1.0, juce::Colour(), stopIcon, 0.5, juce::Colour(), stopIcon, 0.8, juce::Colour(), 0);
    stopKickButton.addListener(this);

    addAndMakeVisible(areaKick);
    areaKick.addListener(this);
    areaKick.setAlpha(0);
    areaKick.setName("areaKick");
    

    //auto kickImage = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/kick.png"));
    auto kickImage = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/kick.png") );
    imageKick.setImage(kickImage, juce::RectanglePlacement::stretchToFit);
    addAndMakeVisible(imageKick);
       
    //SNARE
    addAndMakeVisible(playSnareButton);
    //playSnareButton.setButtonText("PLAY");
    //playSnareButton.setEnabled(false);
    //playSnareButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    playSnareButton.setImages(false, true, true, playIcon, 1.0, juce::Colour(), playIcon, 0.5, juce::Colour(), playIcon, 0.8, juce::Colour(), 0);
    playSnareButton.addListener(this);

    addAndMakeVisible(stopSnareButton);
    //stopSnareButton.setButtonText("STOP");
    //stopSnareButton.setEnabled(false);
    //stopSnareButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopSnareButton.setImages(false, true, true, stopIcon, 1.0, juce::Colour(), stopIcon, 0.5, juce::Colour(), stopIcon, 0.8, juce::Colour(), 0);
    stopSnareButton.addListener(this);

    addAndMakeVisible(areaSnare);
    areaSnare.addListener(this);
    areaSnare.setAlpha(0);
    areaSnare.setName("areaSnare");
    

    //auto snareImage = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/snare.png"));
    auto snareImage = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/snare.png") );
    imageSnare.setImage(snareImage, juce::RectanglePlacement::stretchToFit);
    addAndMakeVisible(imageSnare);
     
    //TOMS
    addAndMakeVisible(playTomsButton);
    //playTomsButton.setButtonText("PLAY");
    //playTomsButton.setEnabled(false);
    //playTomsButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    playTomsButton.setImages(false, true, true, playIcon, 1.0, juce::Colour(), playIcon, 0.5, juce::Colour(), playIcon, 0.8, juce::Colour(), 0);
    playTomsButton.addListener(this);

    addAndMakeVisible(stopTomsButton);
    //stopTomsButton.setButtonText("STOP");
    //stopTomsButton.setEnabled(false);
    //stopTomsButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopTomsButton.setImages(false, true, true, stopIcon, 1.0, juce::Colour(), stopIcon, 0.5, juce::Colour(), stopIcon, 0.8, juce::Colour(), 0);
    stopTomsButton.addListener(this);

    addAndMakeVisible(areaToms);
    areaToms.addListener(this);
    areaToms.setAlpha(0);
    areaToms.setName("areaToms");
    

    //auto tomsImage = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/toms.png"));
    auto tomsImage = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/toms.png") );
    imageToms.setImage(tomsImage, juce::RectanglePlacement::stretchToFit);
    addAndMakeVisible(imageToms);

    //HIHAT
    addAndMakeVisible(playHihatButton);
    //playHihatButton.setButtonText("PLAY");
    //playHihatButton.setEnabled(false);
    //playHihatButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    playHihatButton.setImages(false, true, true, playIcon, 1.0, juce::Colour(), playIcon, 0.5, juce::Colour(), playIcon, 0.8, juce::Colour(), 0);
    playHihatButton.addListener(this);

    addAndMakeVisible(stopHihatButton);
    //stopHihatButton.setButtonText("STOP");
    //stopHihatButton.setEnabled(false);
    //stopHihatButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopHihatButton.setImages(false, true, true, stopIcon, 1.0, juce::Colour(), stopIcon, 0.5, juce::Colour(), stopIcon, 0.8, juce::Colour(), 0);
    stopHihatButton.addListener(this);

    addAndMakeVisible(areaHihat);
    areaHihat.addListener(this);
    areaHihat.setAlpha(0);
    areaHihat.setName("areaHihat");
    

    //auto hihatImage = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/hihat.png"));
    auto hihatImage = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/hihat.png") );
    imageHihat.setImage(hihatImage, juce::RectanglePlacement::stretchToFit);
    addAndMakeVisible(imageHihat);
            
    //CYMBALS
    addAndMakeVisible(playCymbalsButton);
    //playCymbalsButton.setButtonText("PLAY");
    //playCymbalsButton.setEnabled(false);
    //playCymbalsButton.setColour(juce::TextButton::buttonColourId, juce::Colours::green);
    playCymbalsButton.setImages(false, true, true, playIcon, 1.0, juce::Colour(), playIcon, 0.5, juce::Colour(), playIcon, 0.8, juce::Colour(), 0);
    playCymbalsButton.addListener(this);

    addAndMakeVisible(stopCymbalsButton);
    //stopCymbalsButton.setButtonText("STOP");
    //stopCymbalsButton.setEnabled(false);
    //stopCymbalsButton.setColour(juce::TextButton::buttonColourId, juce::Colours::red);
    stopCymbalsButton.setImages(false, true, true, stopIcon, 1.0, juce::Colour(), stopIcon, 0.5, juce::Colour(), stopIcon, 0.8, juce::Colour(), 0);
    stopCymbalsButton.addListener(this);

    addAndMakeVisible(areaCymbals);
    areaCymbals.addListener(this);
    areaCymbals.setAlpha(0);
    areaCymbals.setName("areaCymbals");
    

    //auto cymbalsImage = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/cymbals.png"));
    auto cymbalsImage = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/cymbals.png") );
    imageCymbals.setImage(cymbalsImage, juce::RectanglePlacement::stretchToFit);
    addAndMakeVisible(imageCymbals);
    
    //-----------------------------------------------------
    //auto browseIcon = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/browse.png"));
    auto browseIcon = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/browse.png") );

    addAndMakeVisible(openButton);
    openButton.setImages(false, true, true, browseIcon, 1.0, juce::Colour(), browseIcon, 0.5, juce::Colour(),browseIcon, 0.8, juce::Colour(), 0);
    openButton.addListener(this);

    formatManager.registerBasicFormats();
    audioProcessor.transportProcessorKick.addChangeListener(this);
    audioProcessor.transportProcessor.addChangeListener(this);
    //audioProcessor.transportProcessorKick.addChangeListener(this);
    audioProcessor.transportProcessorSnare.addChangeListener(this);
    audioProcessor.transportProcessorToms.addChangeListener(this);
    audioProcessor.transportProcessorHihat.addChangeListener(this);
    audioProcessor.transportProcessorCymbals.addChangeListener(this);
    
    //VISUALIZER
    thumbnail->addChangeListener(this);
    thumbnailKickOut->addChangeListener(this);
    thumbnailSnareOut->addChangeListener(this);
    thumbnailTomsOut->addChangeListener(this);
    thumbnailHihatOut->addChangeListener(this);
    thumbnailCymbalsOut->addChangeListener(this);

    
    
        

    try{
        //mymoduleKick=torch::jit::load("../src/scripted_modules/my_scripted_module_kick.pt");
        juce::String kickString = modelsDir.getFullPathName() + "/my_scripted_module_kick.pt";
        mymoduleKick = torch::jit::load(kickString.toStdString());
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }
    

    try{
        //mymoduleSnare=torch::jit::load("../src/scripted_modules/my_scripted_module_snare.pt");
        juce::String snareString = modelsDir.getFullPathName() + "/my_scripted_module_snare.pt";
        mymoduleSnare = torch::jit::load(snareString.toStdString());
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

    try{
        //mymoduleToms=torch::jit::load("../src/scripted_modules/my_scripted_module_toms.pt");
        juce::String tomsString = modelsDir.getFullPathName() + "/my_scripted_module_toms.pt";
        mymoduleToms = torch::jit::load(tomsString.toStdString());
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

    try{
        //mymoduleHihat=torch::jit::load("../src/scripted_modules/my_scripted_module_hihat.pt");
        juce::String hihatString = modelsDir.getFullPathName() + "/my_scripted_module_hihat.pt";
        mymoduleHihat = torch::jit::load(hihatString.toStdString());
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

    try{
        //mymoduleCymbals=torch::jit::load("../src/scripted_modules/my_scripted_module_cymbals.pt");
        juce::String cymbalsString = modelsDir.getFullPathName() + "/my_scripted_module_cymbals.pt";
        mymoduleCymbals = torch::jit::load(cymbalsString.toStdString());
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }
    startTimer(40);



}

DrumsDemixEditor::~DrumsDemixEditor()
{

    DBG("chiudo...");
    audioProcessor.transportProcessor.releaseResources();
    audioProcessor.transportProcessor.setSource(nullptr);
    delete thumbnail;
    delete thumbnailCache;

    audioProcessor.transportProcessorKick.releaseResources();
    audioProcessor.transportProcessorKick.setSource(nullptr);
    delete thumbnailKickOut;
    delete thumbnailCacheKickOut;

    audioProcessor.transportProcessorSnare.releaseResources();
    audioProcessor.transportProcessorSnare.setSource(nullptr);
    delete thumbnailSnareOut;
    delete thumbnailCacheSnareOut;

    audioProcessor.transportProcessorToms.releaseResources();
    audioProcessor.transportProcessorToms.setSource(nullptr);
    delete thumbnailTomsOut;
    delete thumbnailCacheTomsOut;

    audioProcessor.transportProcessorHihat.releaseResources();
    audioProcessor.transportProcessorHihat.setSource(nullptr);
    delete thumbnailHihatOut;
    delete thumbnailCacheHihatOut;

    audioProcessor.transportProcessorCymbals.releaseResources();
    audioProcessor.transportProcessorCymbals.setSource(nullptr);
    delete thumbnailCymbalsOut;
    delete thumbnailCacheCymbalsOut;

    //filesDir.deleteRecursively(false);
}

//==============================================================================
void DrumsDemixEditor::paint(juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    //g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
    g.setColour(juce::Colours::white);
    
    //Background image
    //background = juce::ImageCache::getFromFile(absolutePath.getChildFile("C:/Users/Riccardo/OneDrive - Politecnico di Milano/Documenti/GitHub/DrumsDemix/drums_demix/images/DRUMS DEMIX.png"));
    background = juce::ImageCache::getFromFile( juce::File(imagesDir.getFullPathName() + "/DRUMS DEMIX.png") );
    g.drawImageWithin(background, 0, 0, getWidth(), getHeight(), juce::RectanglePlacement::stretchToFit);
    
    /*
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
     */
   // g.setColour (juce::Colours::white);
   // g.setFont (15.0f);
   // g.drawFittedText ("Hello World!", getLocalBounds(), juce::Justification::centred, 1);
    
    //VISUALIZER

    //N.B: getWidth() - 220 = 780

    int thumbnailHeight = (getHeight() - 200) / 5;
    int thumbnailStartPoint = (getHeight() / 9) + 10;
    int buttonHeight = (getHeight() - 200) / 5;
    juce::Rectangle<int> thumbnailBounds (10 + buttonHeight, (getHeight() / 9)+10, getWidth() - 220- buttonHeight, thumbnailHeight);
    
           if (thumbnail->getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBounds, "Drop a file or load it");
           else 
           {
               paintIfFileLoaded(g, thumbnailBounds, *thumbnail, juce::Colour(200, 149, 127));
               paintCursorInput(g, thumbnailBounds, *thumbnail, juce::Colour(200, 149, 127));
           }

        
    juce::Rectangle<int> thumbnailBoundsKickOut (10 + buttonHeight,10+ thumbnailStartPoint + thumbnailHeight, getWidth() - 220 - buttonHeight, thumbnailHeight);
    
           if (thumbnailKickOut->getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBoundsKickOut, "Kick");
           else
           {
               paintIfFileLoaded(g, thumbnailBoundsKickOut, *thumbnailKickOut, juce::Colour(199, 128, 130));
               paintCursorKick(g, thumbnailBoundsKickOut, *thumbnailKickOut, juce::Colour(199, 128, 130));
           }
    
    juce::Rectangle<int> thumbnailBoundsSnareOut (10 + buttonHeight,20+ thumbnailStartPoint + thumbnailHeight*2, getWidth() - 220 - buttonHeight, thumbnailHeight);
    
           if (thumbnailSnareOut->getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBoundsSnareOut, "Snare");
           else 
           {
               paintIfFileLoaded(g, thumbnailBoundsSnareOut, *thumbnailSnareOut, juce::Colour(139, 188, 172));
               paintCursorSnare(g, thumbnailBoundsSnareOut, *thumbnailSnareOut, juce::Colour(139, 188, 172));
           }
    
    juce::Rectangle<int> thumbnailBoundsTomsOut (10 + buttonHeight,30+ thumbnailStartPoint + thumbnailHeight*3, getWidth() - 220 - buttonHeight, thumbnailHeight);
    
           if (thumbnailTomsOut->getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBoundsTomsOut, "Toms");
           else {
               paintIfFileLoaded(g, thumbnailBoundsTomsOut, *thumbnailTomsOut, juce::Colour(135, 139, 192));
               paintCursorToms(g, thumbnailBoundsTomsOut, *thumbnailTomsOut, juce::Colour(135, 139, 192));
           }
    
    juce::Rectangle<int> thumbnailBoundsHihatOut (10 + buttonHeight,40+ thumbnailStartPoint + thumbnailHeight*4, getWidth() - 220 - buttonHeight, thumbnailHeight);
    
           if (thumbnailHihatOut->getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBoundsHihatOut, "Hihat");
           else {
               paintIfFileLoaded(g, thumbnailBoundsHihatOut, *thumbnailHihatOut, juce::Colour(127, 181, 181));
               paintCursorHihat(g, thumbnailBoundsHihatOut, *thumbnailHihatOut, juce::Colour(127, 181, 181));
           }

    
    juce::Rectangle<int> thumbnailBoundsCymbalsOut (10 + buttonHeight,50+ thumbnailStartPoint + thumbnailHeight*5, getWidth() - 220 - buttonHeight, thumbnailHeight);
    
           if (thumbnailCymbalsOut->getNumChannels() == 0)
               paintIfNoFileLoaded (g, thumbnailBoundsCymbalsOut, "Cymbals");
           else {
               paintIfFileLoaded(g, thumbnailBoundsCymbalsOut, *thumbnailCymbalsOut, juce::Colour(180, 182, 145));
               paintCursorCymbals(g, thumbnailBoundsCymbalsOut, *thumbnailCymbalsOut, juce::Colour(180, 182, 145));
           }
    
}

void DrumsDemixEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    float rowHeight = getHeight()/5; 
    int buttonHeight = (getHeight() - 200) / 5;
    int thumbnailWidth = getWidth() - 220;
    int thumbnailHeight = (getHeight() - 200) / 5;
    int thumbnailStartPoint = (getHeight() / 9) + 10;
    

    testButton.setBounds(getWidth()/2,5, getWidth()/2, getHeight()/9);
    openButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4 + 10 + buttonHeight, getHeight() / 9 + 10, buttonHeight, buttonHeight);

    
    playButton.setBounds(getWidth() - 220 +20, getHeight() / 9 +10, buttonHeight, buttonHeight);
    stopButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4, getHeight() / 9 +10, buttonHeight, buttonHeight);
    imageKit.setBounds(5 , getHeight() / 9 +10, buttonHeight, buttonHeight);
    

    playKickButton.setBounds(getWidth() - 220 +20, 10 + thumbnailStartPoint + thumbnailHeight, buttonHeight, buttonHeight);
    stopKickButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4, 10 + thumbnailStartPoint + thumbnailHeight, buttonHeight, buttonHeight);
    imageKick.setBounds(5, 10 + thumbnailStartPoint + thumbnailHeight, buttonHeight, buttonHeight);
    areaKick.setBounds(10,10+ thumbnailStartPoint + thumbnailHeight, getWidth() - 220, thumbnailHeight);
    

    playSnareButton.setBounds(getWidth() - 220 +20, 20 + thumbnailStartPoint + thumbnailHeight*2, buttonHeight, buttonHeight);
    stopSnareButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4, 20 + thumbnailStartPoint + thumbnailHeight * 2, buttonHeight, buttonHeight);
    imageSnare.setBounds(5, 20 + thumbnailStartPoint + thumbnailHeight * 2, buttonHeight, buttonHeight);
    areaSnare.setBounds(10, 20 + thumbnailStartPoint + thumbnailHeight * 2, getWidth() - 220, thumbnailHeight);
    

    playTomsButton.setBounds(getWidth() - 220 +20, 30 + thumbnailStartPoint + thumbnailHeight*3, buttonHeight, buttonHeight);
    stopTomsButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4, 30 + thumbnailStartPoint + thumbnailHeight * 3, buttonHeight, buttonHeight);
    imageToms.setBounds(5, 30 + thumbnailStartPoint + thumbnailHeight * 3, buttonHeight, buttonHeight);
    areaToms.setBounds(10, 30 + thumbnailStartPoint + thumbnailHeight * 3, getWidth() - 220, thumbnailHeight);
    

    playHihatButton.setBounds(getWidth() - 220 +20, 40 + thumbnailStartPoint + thumbnailHeight*4, buttonHeight, buttonHeight);
    stopHihatButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4, 40 + thumbnailStartPoint + thumbnailHeight * 4, buttonHeight, buttonHeight);
    imageHihat.setBounds(5, 40 + thumbnailStartPoint + thumbnailHeight * 4, buttonHeight, buttonHeight);
    areaHihat.setBounds(10, 40 + thumbnailStartPoint + thumbnailHeight * 4, getWidth() - 220, thumbnailHeight);
    

    playCymbalsButton.setBounds(getWidth() - 220 +20, 50 + thumbnailStartPoint + thumbnailHeight*5, buttonHeight, buttonHeight);
    stopCymbalsButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4, 50 + thumbnailStartPoint + thumbnailHeight * 5, buttonHeight, buttonHeight);
    imageCymbals.setBounds(5, 50 + thumbnailStartPoint + thumbnailHeight * 5, buttonHeight, buttonHeight);
    areaCymbals.setBounds(10, 50 + thumbnailStartPoint + thumbnailHeight * 5, getWidth() - 220, thumbnailHeight);
    

    areaFull.setBounds(10, (getHeight() / 9) + 10, getWidth() - 220, thumbnailHeight);

    //textLabel.setBounds(10, 60 + thumbnailStartPoint + thumbnailHeight * 5, getWidth() - 220, thumbnailHeight);
    //textLabel.setFont(juce::Font(16.0f, juce::Font::bold)); 
    //textLabel.setColour(juce::Label::textColourId, juce::Colours::lightgreen);

    downloadKickButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4 + buttonHeight + 10, 10 + thumbnailStartPoint + thumbnailHeight, buttonHeight, buttonHeight);
    downloadSnareButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4 + buttonHeight + 10, 20 + thumbnailStartPoint + thumbnailHeight * 2, buttonHeight, buttonHeight);
    downloadTomsButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4 + buttonHeight + 10, 30 + thumbnailStartPoint + thumbnailHeight * 3, buttonHeight, buttonHeight);
    downloadHihatButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4 + buttonHeight + 10, 40 + thumbnailStartPoint + thumbnailHeight * 4, buttonHeight, buttonHeight);
    downloadCymbalsButton.setBounds(getWidth() - 220 + 10 + (getHeight() - 200) / 4 + buttonHeight + 10, 50 + thumbnailStartPoint + thumbnailHeight * 5, buttonHeight, buttonHeight);
    
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

    if (btn == &testButton) {


        auto begin = std::chrono::high_resolution_clock::now();
        //***TAKE THE INPUT FROM THE MIXED DRUMS FILE***


        //-From Wav to AudiofileBuffer


        Utils utils = Utils();
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
        
        // DECOMMENT IF USING ONLY THE KICK FOR QUICK DEBUGGING
        //CreateWavQuick(yKick);

        // DECOMMENT IF USING ALL THE DRUMS
        std::vector<at::Tensor> tensorList = {yKick, ySnare, yToms, yHihat, yCymbals};
        CreateWav(tensorList, inputFileName.dropLastCharacters(4));


        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

        DBG("TEMPO MISURATO (in ms): ");
        DBG(std::to_string(elapsed.count()));

        //textLabel.setText(std::to_string(elapsed.count()), juce::dontSendNotification);

        //playKickButton.setEnabled(true);
        //playSnareButton.setEnabled(true);
        //playTomsButton.setEnabled(true);
        //playHihatButton.setEnabled(true);
        //playCymbalsButton.setEnabled(true);
             

    }
    if (btn == &openButton) {

        juce::FileChooser chooser("Choose a Wav or Aiff File", juce::File::getSpecialLocation(juce::File::userDesktopDirectory), "*.wav;*.aiff;*.mp3");

        if (chooser.browseForFileToOpen())
        {
            //juce::File myFile;
            myFile = chooser.getResult();
            inputFileName = chooser.getResult().getFileName();

            areaKick.setInFile(inputFileName);
            areaSnare.setInFile(inputFileName);
            areaToms.setInFile(inputFileName);
            areaHihat.setInFile(inputFileName);
            areaCymbals.setInFile(inputFileName);



            juce::AudioFormatReader* reader = formatManager.createReaderFor(myFile);

            if (reader != nullptr)
            {

                std::unique_ptr<juce::AudioFormatReaderSource> tempSource(new juce::AudioFormatReaderSource(reader, true));

                audioProcessor.transportProcessor.setSource(tempSource.get());
                transportStateChanged(Stopped, "input");

                playSource.reset(tempSource.get());
                areaFull.setSrc(tempSource.release());


                DBG("IFopenbuttonclicked");

            }
            DBG("openbuttonclicked");
            testButton.setEnabled(true);
            //playButton.setEnabled(true);

            auto docsDir = juce::File::getSpecialLocation(juce::File::userMusicDirectory);

            DBG(docsDir.getFullPathName());
            DBG(juce::File::getSpecialLocation(juce::File::currentExecutableFile).getFullPathName());
            DBG(juce::File::getSpecialLocation(juce::File::currentApplicationFile).getFullPathName());
            DBG(juce::File::getSpecialLocation(juce::File::invokedExecutableFile).getFullPathName());
            DBG(juce::File::getSpecialLocation(juce::File::hostApplicationPath).getFullPathName());
            DBG(juce::File::getSpecialLocation(juce::File::tempDirectory).getFullPathName());


            /*
            auto parentDir = juce::File(docsDir.getFullPathName() + "/Che_Cartellona!");
            parentDir.createDirectory(); */


        }
        //VISUALIZER
        thumbnail->setSource(new juce::FileInputSource(myFile));
    }


    if (btn == &downloadKickButton) {

        juce::FileChooser chooser("Choose a Folder to save the .wav File", juce::File::getSpecialLocation(juce::File::userDesktopDirectory));

        if (chooser.browseForDirectory())
        {
            DBG(chooser.getResult().getFullPathName());
            CreateWavQuick(yKick, chooser.getResult().getFullPathName(),inputFileName.dropLastCharacters(4)+"_kick.wav");


        }

    }

    if (btn == &downloadSnareButton) {

        juce::FileChooser chooser("Choose a Folder to save the .wav File", juce::File::getSpecialLocation(juce::File::userDesktopDirectory));

        if (chooser.browseForDirectory())
        {
            DBG(chooser.getResult().getFullPathName());
            CreateWavQuick(ySnare, chooser.getResult().getFullPathName(), inputFileName.dropLastCharacters(4) + "_snare.wav");


        }

    }

    if (btn == &downloadTomsButton) {

        juce::FileChooser chooser("Choose a Folder to save the .wav File", juce::File::getSpecialLocation(juce::File::userDesktopDirectory));

        if (chooser.browseForDirectory())
        {
            DBG(chooser.getResult().getFullPathName());
            CreateWavQuick(yToms, chooser.getResult().getFullPathName(), inputFileName.dropLastCharacters(4) + "_toms.wav");


        }

    }

    if (btn == &downloadHihatButton) {

        juce::FileChooser chooser("Choose a Folder to save the .wav File", juce::File::getSpecialLocation(juce::File::userDesktopDirectory));

        if (chooser.browseForDirectory())
        {
            DBG(chooser.getResult().getFullPathName());
            CreateWavQuick(yHihat, chooser.getResult().getFullPathName(), inputFileName.dropLastCharacters(4) + "_hihats.wav");


        }

    }

    if (btn == &downloadCymbalsButton) {

        juce::FileChooser chooser("Choose a Folder to save the .wav File", juce::File::getSpecialLocation(juce::File::userDesktopDirectory));

        if (chooser.browseForDirectory())
        {
            DBG(chooser.getResult().getFullPathName());
            CreateWavQuick(yCymbals, chooser.getResult().getFullPathName(), inputFileName.dropLastCharacters(4) + "_cymbals.wav");


        }

    }

    if (btn == &playButton){

        audioProcessor.playInput = true;
        audioProcessor.playKick = false;
        audioProcessor.playSnare = false;
        audioProcessor.playToms = false;
        audioProcessor.playHihat = false;
        audioProcessor.playCymbals = false;


        transportStateChanged(Starting, "input");
        DBG("playbuttonclicked");
        //playButton.setEnabled(false);
        //stopButton.setEnabled(true);
        

    }
    if (btn == &stopButton){
        transportStateChanged(Stopping, "input");
        DBG("stopbuttonclicked");
        //playButton.setEnabled(true);
        //stopButton.setEnabled(false);

    }


    if (btn == &playKickButton) {
        audioProcessor.playInput = false;
        audioProcessor.playKick = true;
        audioProcessor.playSnare = false;
        audioProcessor.playToms = false;
        audioProcessor.playHihat = false;
        audioProcessor.playCymbals = false;
        transportStateChanged(Starting, "kick");
        DBG("playbuttonclicked");
        //playButton.setEnabled(false);
        //stopButton.setEnabled(true);


    }
    if (btn == &stopKickButton) {
        transportStateChanged(Stopping, "kick");
        DBG("stopbuttonclicked");
        //playButton.setEnabled(true);
        //stopButton.setEnabled(false);

    }

    if (btn == &playSnareButton) {
        //audioProcessor.transportProcessorSnare.setPosition(0.0);
        audioProcessor.playInput = false;
        audioProcessor.playKick = false;
        audioProcessor.playSnare = true;
        audioProcessor.playToms = false;
        audioProcessor.playHihat = false;
        audioProcessor.playCymbals = false;
        transportStateChanged(Starting, "snare");
        DBG("playbuttonclicked");
        //playButton.setEnabled(false);
        //stopButton.setEnabled(true);


    }
    if (btn == &stopSnareButton) {
        transportStateChanged(Stopping, "snare");
        DBG("stopbuttonclicked");
        //playButton.setEnabled(true);
        //stopButton.setEnabled(false);

    }

    if (btn == &playTomsButton) {
        audioProcessor.playInput = false;
        audioProcessor.playKick = false;
        audioProcessor.playSnare = false;
        audioProcessor.playToms = true;
        audioProcessor.playHihat = false;
        audioProcessor.playCymbals = false;
        transportStateChanged(Starting, "tom");
        DBG("playbuttonclicked");
        //playButton.setEnabled(false);
        //stopButton.setEnabled(true);


    }
    if (btn == &stopTomsButton) {
        transportStateChanged(Stopping, "tom");
        DBG("stopbuttonclicked");
        //playButton.setEnabled(true);
        //stopButton.setEnabled(false);

    }


    if (btn == &playHihatButton) {
        audioProcessor.playInput = false;
        audioProcessor.playKick = false;
        audioProcessor.playSnare = false;
        audioProcessor.playToms = false;
        audioProcessor.playHihat = true;
        audioProcessor.playCymbals = false;
        transportStateChanged(Starting, "hihat");
        DBG("playbuttonclicked");
        //playButton.setEnabled(false);
        //stopButton.setEnabled(true);


    }
    if (btn == &stopHihatButton) {
        transportStateChanged(Stopping, "hihat");
        DBG("stopbuttonclicked");
        //playButton.setEnabled(true);
        //stopButton.setEnabled(false);

    }

    if (btn == &playCymbalsButton) {
        audioProcessor.playInput = false;
        audioProcessor.playKick = false;
        audioProcessor.playSnare = false;
        audioProcessor.playToms = false;
        audioProcessor.playHihat = false;
        audioProcessor.playCymbals = true;
        transportStateChanged(Starting, "cymbals");
        DBG("playbuttonclicked");
        //playButton.setEnabled(false);
        //stopButton.setEnabled(true);


    }
    if (btn == &stopCymbalsButton) {
        transportStateChanged(Stopping, "cymbals");
        DBG("stopbuttonclicked");
        //playButton.setEnabled(true);
        //stopButton.setEnabled(false);

    }



}

void DrumsDemixEditor::transportStateChanged(TransportState newState, juce::String id)
{
    if (id == "input")
    {
        if (newState != state)
        {
            state = newState;

            switch (state)
            {
            case Stopped:
                audioProcessor.transportProcessor.setPosition(0.0);
                //playButton.setEnabled(true);
                //stopButton.setEnabled(false);
                break;
            case Starting:
                //stopButton.setEnabled(true);
                //playButton.setEnabled(false);
                audioProcessor.transportProcessor.start();
                break;
            case Playing:
                //stopButton.setEnabled(true);
                break;
            case Stopping:
                //stopButton.setEnabled(false);
                //playButton.setEnabled(true);
                audioProcessor.transportProcessor.stop();
                break;
            }
        }
    }

    if (id == "kick")
    {
        if (newState != state)
        {
            state = newState;

            switch (state)
            {
            case Stopped:
                audioProcessor.transportProcessorKick.setPosition(0.0);
                //playKickButton.setEnabled(true);
                //stopKickButton.setEnabled(false);
                break;
            case Starting:
                //stopKickButton.setEnabled(true);
                //playKickButton.setEnabled(false);
                audioProcessor.transportProcessorKick.start();
                break;
            case Playing:
                //stopKickButton.setEnabled(true);
                break;
            case Stopping:
                //stopKickButton.setEnabled(false);
                //playKickButton.setEnabled(true);
                audioProcessor.transportProcessorKick.stop();
                break;
            }
        }
    }

    if (id == "snare")
    {
        if (newState != state)
        {
            state = newState;

            switch (state)
            {
            case Stopped:
                audioProcessor.transportProcessorSnare.setPosition(0.0);
                //playSnareButton.setEnabled(true);
                //stopSnareButton.setEnabled(false);
                break;
            case Starting:
                //stopSnareButton.setEnabled(true);
                //playSnareButton.setEnabled(false);
                audioProcessor.transportProcessorSnare.start();
                break;
            case Playing:
                //stopSnareButton.setEnabled(true);
                break;
            case Stopping:
                //stopSnareButton.setEnabled(false);
                //playSnareButton.setEnabled(true);
                audioProcessor.transportProcessorSnare.stop();
                break;
            }
        }
    }

    if (id == "tom")
    {
        if (newState != state)
        {
            state = newState;

            switch (state)
            {
            case Stopped:
                audioProcessor.transportProcessorToms.setPosition(0.0);
                //playTomsButton.setEnabled(true);
                //stopTomsButton.setEnabled(false);
                break;
            case Starting:
                //stopTomsButton.setEnabled(true);
                //playTomsButton.setEnabled(false);
                audioProcessor.transportProcessorToms.start();
                break;
            case Playing:
                //stopTomsButton.setEnabled(true);
                break;
            case Stopping:
                //stopTomsButton.setEnabled(false);
                //playTomsButton.setEnabled(true);
                audioProcessor.transportProcessorToms.stop();
                break;
            }
        }
    }

    if (id == "hihat")
    {
        if (newState != state)
        {
            state = newState;

            switch (state)
            {
            case Stopped:
                audioProcessor.transportProcessorHihat.setPosition(0.0);
                //playHihatButton.setEnabled(true);
                //stopHihatButton.setEnabled(false);
                break;
            case Starting:
                //stopHihatButton.setEnabled(true);
                //playHihatButton.setEnabled(false);
                audioProcessor.transportProcessorHihat.start();
                break;
            case Playing:
                //stopHihatButton.setEnabled(true);
                break;
            case Stopping:
                //stopHihatButton.setEnabled(false);
                //playHihatButton.setEnabled(true);
                audioProcessor.transportProcessorHihat.stop();
                break;
            }
        }
    }

    if (id == "cymbals")
    {
        if (newState != state)
        {
            state = newState;

            switch (state)
            {
            case Stopped:
                audioProcessor.transportProcessorCymbals.setPosition(0.0);
                //playCymbalsButton.setEnabled(true);
                //stopCymbalsButton.setEnabled(false);
                break;
            case Starting:
                //stopCymbalsButton.setEnabled(true);
                //playCymbalsButton.setEnabled(false);
                audioProcessor.transportProcessorCymbals.start();
                break;
            case Playing:
                //stopCymbalsButton.setEnabled(true);
                break;
            case Stopping:
                //stopCymbalsButton.setEnabled(false);
                //playCymbalsButton.setEnabled(true);
                audioProcessor.transportProcessorCymbals.stop();
                break;
            }
        }
    }
}


void DrumsDemixEditor::displayOut(juce::AudioBuffer<float>& buffer, juce::AudioThumbnail& thumbnailOut)
{
    //juce::MemoryAudioSource input(buffer, true, false);
    //std::this_thread::sleep_for(std::chrono::milliseconds(10000));

    thumbnailOut.reset(buffer.getNumChannels(), 44100, buffer.getNumSamples());
    thumbnailOut.addBlock(0, buffer, 0, buffer.getNumSamples());
}


//VISUALIZER
void DrumsDemixEditor::changeListenerCallback (juce::ChangeBroadcaster* source)
  {
    if (source == thumbnail) { repaint(); }
    if (source == thumbnailKickOut) { repaint(); }
    if (source == thumbnailSnareOut) { repaint(); }
    if (source == thumbnailTomsOut) { repaint(); }
    if (source == thumbnailHihatOut) { repaint(); }
    if (source == thumbnailCymbalsOut) { repaint(); }
    if(source == &audioProcessor.transportProcessor)
    {

        if(audioProcessor.transportProcessor.isPlaying())
        {
        transportStateChanged(Playing, "input");
        }
        else
        {
            DBG("input reset");
        transportStateChanged(Stopped, "input");
        }
    }

    if (source == &audioProcessor.transportProcessorKick)
    {

        if (audioProcessor.transportProcessorKick.isPlaying())
        {
            transportStateChanged(Playing, "kick");
        }
        else
        {
            transportStateChanged(Stopped, "kick");
        }
    }


    if (source == &audioProcessor.transportProcessorSnare)
    {

        if (audioProcessor.transportProcessorSnare.isPlaying())
        {
            DBG("snare start");
            transportStateChanged(Playing, "snare");
        }
        else
        {
            DBG("snare reset");
            transportStateChanged(Stopped, "snare");
        }
    }


    if (source == &audioProcessor.transportProcessorToms)
    {

        if (audioProcessor.transportProcessorToms.isPlaying())
        {
            transportStateChanged(Playing, "tom");
        }
        else
        {
            transportStateChanged(Stopped, "tom");
        }
    }

    if (source == &audioProcessor.transportProcessorHihat)
    {


        if (audioProcessor.transportProcessorHihat.isPlaying())
        {
            transportStateChanged(Playing, "hihat");
        }
        else
        {
            transportStateChanged(Stopped, "hihat");
        }
    }

    if (source == &audioProcessor.transportProcessorCymbals)
    {

        if (audioProcessor.transportProcessorCymbals.isPlaying())
        {
            transportStateChanged(Playing, "cymbals");
        }
        else
        {
            transportStateChanged(Stopped, "cymbals");
        }
    }
  }

void DrumsDemixEditor::paintIfNoFileLoaded (juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, at::string phrase)
  {
      g.setColour(juce::Colour(46, 45, 45));
      g.fillRect (thumbnailBounds);
      g.setColour (juce::Colours::white);
      g.drawFittedText (phrase, thumbnailBounds, juce::Justification::centred, 1);
  }

void DrumsDemixEditor::paintIfFileLoaded (juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color)
 {
    float thumbnailHeight = (getHeight() - 200) / 5;
    float thumbnailStartPoint = (getHeight() / 9) + 10;
     g.setColour (juce::Colour(46, 45, 45));
     g.fillRect (thumbnailBounds);

     g.setColour (color);                               // [8]
     auto audioLength = (float)thumbnailWav.getTotalLength();

     thumbnailWav.drawChannels (g,                                      // [9]
                             thumbnailBounds,
                             0.0,                                    // start time
                             thumbnailWav.getTotalLength(),             // end time
                             1.0f);  // vertical zoom

     g.setColour(juce::Colours::lightgrey);

     //if (audioProcessor.playInput) {
     //    auto audioPosition = (float)audioProcessor.transportProcessor.getCurrentPosition();
     //    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
     //    g.drawLine(drawPosition, (float)(getHeight() / 9) + 10, drawPosition,
     //        (float)(getHeight() / 9) + 10 + thumbnailHeight, 1.0f);
     //}
     //if (audioProcessor.playKick) {
     //    auto audioPosition = (float)audioProcessor.transportProcessorKick.getCurrentPosition();
     //    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
     //    g.drawLine(drawPosition, (float)10 + thumbnailStartPoint + thumbnailHeight, drawPosition,
     //        (float)10 + thumbnailStartPoint + thumbnailHeight +thumbnailHeight, 1.0f);
     //}
     //if (audioProcessor.playSnare) {
     //    auto audioPosition = (float)audioProcessor.transportProcessorSnare.getCurrentPosition();
     //    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
     //    g.drawLine(drawPosition, (float)20 + thumbnailStartPoint + thumbnailHeight * 2, drawPosition,
     //        (float)20 + thumbnailStartPoint + thumbnailHeight * 2 +thumbnailHeight, 1.0f);
     //}
     //if (audioProcessor.playToms) {
     //    auto audioPosition = (float)audioProcessor.transportProcessorToms.getCurrentPosition();
     //    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
     //    g.drawLine(drawPosition, (float)30 + thumbnailStartPoint + thumbnailHeight * 3, drawPosition,
     //        (float)30 + thumbnailStartPoint + thumbnailHeight * 3 +thumbnailHeight, 1.0f);
     //}
     //if (audioProcessor.playHihat) {
     //    auto audioPosition = (float)audioProcessor.transportProcessorHihat.getCurrentPosition();
     //    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
     //    g.drawLine(drawPosition, (float)40 + thumbnailStartPoint + thumbnailHeight * 4, drawPosition,
     //        (float)40 + thumbnailStartPoint + thumbnailHeight * 4 + thumbnailHeight, 1.0f);
     //}
     //if (audioProcessor.playCymbals) {
     //    auto audioPosition = (float)audioProcessor.transportProcessorCymbals.getCurrentPosition();
     //    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
     //    g.drawLine(drawPosition, (float)50 + thumbnailStartPoint + thumbnailHeight * 5, drawPosition,
     //        (float)50 + thumbnailStartPoint + thumbnailHeight * 5 + thumbnailHeight, 1.0f);
     //}

 }

void DrumsDemixEditor::paintCursorInput(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color) {
    auto audioLength = (float)thumbnailWav.getTotalLength();
    float thumbnailHeight = (getHeight() - 200) / 5;
    float thumbnailStartPoint = (getHeight() / 9) + 10;
    g.setColour(juce::Colours::lightgrey);
    auto audioPosition = (float)audioProcessor.transportProcessor.getCurrentPosition();
    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
    g.drawLine(drawPosition, (float)(getHeight() / 9) + 10, drawPosition,
        (float)(getHeight() / 9) + 10 + thumbnailHeight, 1.0f);
}

void DrumsDemixEditor::paintCursorKick(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color) {
    auto audioLength = (float)thumbnailWav.getTotalLength();
    float thumbnailHeight = (getHeight() - 200) / 5;
    float thumbnailStartPoint = (getHeight() / 9) + 10;
    g.setColour(juce::Colours::lightgrey);
    auto audioPosition = (float)audioProcessor.transportProcessorKick.getCurrentPosition();
    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
    g.drawLine(drawPosition, (float)10 + thumbnailStartPoint + thumbnailHeight, drawPosition,
        (float)10 + thumbnailStartPoint + thumbnailHeight + thumbnailHeight, 1.0f);
}

void DrumsDemixEditor::paintCursorSnare(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color) {
    auto audioLength = (float)thumbnailWav.getTotalLength();
    float thumbnailHeight = (getHeight() - 200) / 5;
    float thumbnailStartPoint = (getHeight() / 9) + 10;
    g.setColour(juce::Colours::lightgrey);
    auto audioPosition = (float)audioProcessor.transportProcessorSnare.getCurrentPosition();
    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
    g.drawLine(drawPosition, (float)20 + thumbnailStartPoint + thumbnailHeight * 2, drawPosition,
        (float)20 + thumbnailStartPoint + thumbnailHeight * 2 + thumbnailHeight, 1.0f);
}

void DrumsDemixEditor::paintCursorToms(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color) {
    auto audioLength = (float)thumbnailWav.getTotalLength();
    float thumbnailHeight = (getHeight() - 200) / 5;
    float thumbnailStartPoint = (getHeight() / 9) + 10;
    g.setColour(juce::Colours::lightgrey);
    auto audioPosition = (float)audioProcessor.transportProcessorToms.getCurrentPosition();
    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
    g.drawLine(drawPosition, (float)30 + thumbnailStartPoint + thumbnailHeight * 3, drawPosition,
        (float)30 + thumbnailStartPoint + thumbnailHeight * 3 + thumbnailHeight, 1.0f);
}

void DrumsDemixEditor::paintCursorHihat(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color) {
    auto audioLength = (float)thumbnailWav.getTotalLength();
    float thumbnailHeight = (getHeight() - 200) / 5;
    float thumbnailStartPoint = (getHeight() / 9) + 10;
    g.setColour(juce::Colours::lightgrey);
    auto audioPosition = (float)audioProcessor.transportProcessorHihat.getCurrentPosition();
    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
    g.drawLine(drawPosition, (float)40 + thumbnailStartPoint + thumbnailHeight * 4, drawPosition,
        (float)40 + thumbnailStartPoint + thumbnailHeight * 4 + thumbnailHeight, 1.0f);
}

void DrumsDemixEditor::paintCursorCymbals(juce::Graphics& g, const juce::Rectangle<int>& thumbnailBounds, juce::AudioThumbnail& thumbnailWav, juce::Colour color) {
    auto audioLength = (float)thumbnailWav.getTotalLength();
    float thumbnailHeight = (getHeight() - 200) / 5;
    float thumbnailStartPoint = (getHeight() / 9) + 10;
    g.setColour(juce::Colours::lightgrey);
    auto audioPosition = (float)audioProcessor.transportProcessorCymbals.getCurrentPosition();
    auto drawPosition = (audioPosition / audioLength) * (float)thumbnailBounds.getWidth() + (float)thumbnailBounds.getX();
    g.drawLine(drawPosition, (float)50 + thumbnailStartPoint + thumbnailHeight * 5, drawPosition,
        (float)50 + thumbnailStartPoint + thumbnailHeight * 5 + thumbnailHeight, 1.0f);
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
        if ((juce::File(file).isAChildOf(filesDir.getFullPathName()))) { DBG("cercando di droppare un file dall'interno!"); };


        if (isInterestedInFileDrag(files) && !(juce::File(file).isAChildOf(filesDir.getFullPathName())))
        {
            loadFile(file);

        }
    }
    repaint();

}

void DrumsDemixEditor::loadFile(const juce::String& path)
{


    auto file = juce::File(path);
    inputFileName = file.getFileName();

    areaKick.setInFile(inputFileName);
    areaSnare.setInFile(inputFileName);
    areaToms.setInFile(inputFileName);
    areaHihat.setInFile(inputFileName);
    areaCymbals.setInFile(inputFileName);
    
    DBG(inputFileName);
    
    myFile = file;
    juce::AudioFormatReader* reader = formatManager.createReaderFor(file);
    if (reader != nullptr)
    {

        std::unique_ptr<juce::AudioFormatReaderSource> tempSource(new juce::AudioFormatReaderSource(reader, true));

        audioProcessor.transportProcessor.setSource(tempSource.get());
        transportStateChanged(Stopped, "input");

        playSource.reset(tempSource.get());
        areaFull.setSrc(tempSource.release());
        DBG("IFopenbuttonclicked");

    }
    DBG("openbuttonclicked");
    testButton.setEnabled(true);
    playButton.setEnabled(true);

    thumbnail->setSource(new juce::FileInputSource(file));

}

void DrumsDemixEditor::InferModels(std::vector<torch::jit::IValue> my_input, torch::Tensor phase, int size) 
{
    //c10::InferenceMode guard(true);
    DBG("Infering the Models...");
    Utils utils = Utils();
    //***INFER THE MODEL***


        //-Forward
        at::Tensor outputsKick = mymoduleKick.forward(my_input).toTensor();

        // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
         at::Tensor outputsSnare = mymoduleSnare.forward(my_input).toTensor();
         at::Tensor outputsToms = mymoduleToms.forward(my_input).toTensor();
         at::Tensor outputsHihat = mymoduleHihat.forward(my_input).toTensor();
         at::Tensor outputsCymbals = mymoduleCymbals.forward(my_input).toTensor();

        //-Need another dimension to do batch_istft
        outputsKick = torch::squeeze(outputsKick, 0);

        DBG("outputs sizes: ");
        DBG(outputsKick.sizes()[0]);
        DBG(outputsKick.sizes()[1]);
        DBG(outputsKick.sizes()[2]);
        //DBG(outputs.sizes()[3]);




        // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
         outputsSnare = torch::squeeze(outputsSnare, 0);
         outputsToms = torch::squeeze(outputsToms, 0);
         outputsHihat = torch::squeeze(outputsHihat, 0);
         outputsCymbals = torch::squeeze(outputsCymbals, 0);
        

        //-Compute ISTFT

        yKick = utils.batch_istft(outputsKick, phase, size);

        DBG("y tensor sizes: ");
        DBG(yKick.sizes()[0]);
        DBG(yKick.sizes()[1]);


        // COMMENTA PER AUMENTARE LA RUNTIME SPEED PER QUICK DEBUGGING
        
         ySnare = utils.batch_istft(outputsSnare, phase, size);
         yToms = utils.batch_istft(outputsToms, phase, size);
         yHihat = utils.batch_istft(outputsHihat, phase, size);
         yCymbals = utils.batch_istft(outputsCymbals, phase, size);
        


        /// RELOADARE I MODELLI E' UN MODO PER NON FAR CRASHARE AL SECONDO SEPARATE CONSECUTIVO, MA FORSE NON IL MIGLIOR MODO! (RALLENTA UN PO')

         
         try {
             //mymoduleKick=torch::jit::load("../src/scripted_modules/my_scripted_module_kick.pt");
             juce::String kickString = modelsDir.getFullPathName() + "/my_scripted_module_kick.pt";
             mymoduleKick = torch::jit::load(kickString.toStdString());
         }
         catch (const c10::Error& e) {
             DBG("error"); //indicate error to calling code
         }


         try {
             //mymoduleSnare=torch::jit::load("../src/scripted_modules/my_scripted_module_snare.pt");
             juce::String snareString = modelsDir.getFullPathName() + "/my_scripted_module_snare.pt";
             mymoduleSnare = torch::jit::load(snareString.toStdString());
         }
         catch (const c10::Error& e) {
             DBG("error"); //indicate error to calling code
         }

         try {
             //mymoduleToms=torch::jit::load("../src/scripted_modules/my_scripted_module_toms.pt");
             juce::String tomsString = modelsDir.getFullPathName() + "/my_scripted_module_toms.pt";
             mymoduleToms = torch::jit::load(tomsString.toStdString());
         }
         catch (const c10::Error& e) {
             DBG("error"); //indicate error to calling code
         }

         try {
             //mymoduleHihat=torch::jit::load("../src/scripted_modules/my_scripted_module_hihat.pt");
             juce::String hihatString = modelsDir.getFullPathName() + "/my_scripted_module_hihat.pt";
             mymoduleHihat = torch::jit::load(hihatString.toStdString());
         }
         catch (const c10::Error& e) {
             DBG("error"); //indicate error to calling code
         }

         try {
             //mymoduleCymbals=torch::jit::load("../src/scripted_modules/my_scripted_module_cymbals.pt");
             juce::String cymbalsString = modelsDir.getFullPathName() + "/my_scripted_module_cymbals.pt";
             mymoduleCymbals = torch::jit::load(cymbalsString.toStdString());
         }
         catch (const c10::Error& e) {
             DBG("error"); //indicate error to calling code
         }

         
        

}

void DrumsDemixEditor::CreateWav(std::vector<at::Tensor> tList, juce::String name)
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
        juce::AudioBuffer<float> bufferY = juce::AudioBuffer<float>(dataPtrs, 2, yInstr.sizes()[1]); 

        //-Create Source
        std::unique_ptr<juce::MemoryAudioSource> memSourcePtr (new juce::MemoryAudioSource (bufferY, true, false));

        //-Create Writer
        juce::WavAudioFormat formatWav;
        std::unique_ptr<juce::AudioFormatWriter> writerY;
        juce::File outFile;

        if(torch::equal(yInstr, yKick)) {

            outFile = juce::File(filesDir.getFullPathName()).getChildFile(name + "_kick.wav");
            DBG(outFile.getFullPathName());
            writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(outFile),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));
            if (writerY != nullptr)
                writerY->writeFromAudioSampleBuffer(bufferY, 0, bufferY.getNumSamples());



            audioProcessor.transportProcessorKick.setSource(memSourcePtr.get());
            transportStateChanged(Stopped, "kick");

            playSourceKick.reset(memSourcePtr.get());
            areaKick.setSrcInst(memSourcePtr.release());

            displayOut(bufferY,*thumbnailKickOut);
        }

        else if(torch::equal(yInstr, ySnare)) {

            outFile = juce::File(filesDir.getFullPathName()).getChildFile(name + "_snare.wav");
            writerY.reset(formatWav.createWriterFor(new juce::FileOutputStream(outFile),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));
            if (writerY != nullptr)
                writerY->writeFromAudioSampleBuffer(bufferY, 0, bufferY.getNumSamples());


            audioProcessor.transportProcessorSnare.setSource(memSourcePtr.get());
            transportStateChanged(Stopped, "snare");

            playSourceSnare.reset(memSourcePtr.get());
            areaSnare.setSrcInst(memSourcePtr.release());

            displayOut(bufferY, *thumbnailSnareOut);

        }

        else if(torch::equal(yInstr, yToms)){

            outFile = juce::File(filesDir.getFullPathName()).getChildFile(name + "_toms.wav");
            writerY.reset(formatWav.createWriterFor(new juce::FileOutputStream(outFile),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));
            if (writerY != nullptr)
                writerY->writeFromAudioSampleBuffer(bufferY, 0, bufferY.getNumSamples());


            audioProcessor.transportProcessorToms.setSource(memSourcePtr.get());
            transportStateChanged(Stopped, "tom");


            playSourceToms.reset(memSourcePtr.get());
            areaToms.setSrcInst(memSourcePtr.release());

            displayOut(bufferY, *thumbnailTomsOut);
        }

        else if(torch::equal(yInstr, yHihat)){

            outFile = juce::File(filesDir.getFullPathName()).getChildFile(name + "_hihat.wav");
            writerY.reset(formatWav.createWriterFor(new juce::FileOutputStream(outFile),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));

            if (writerY != nullptr)
                writerY->writeFromAudioSampleBuffer(bufferY, 0, bufferY.getNumSamples());


            audioProcessor.transportProcessorHihat.setSource(memSourcePtr.get());
            transportStateChanged(Stopped, "hihat");


            playSourceHihat.reset(memSourcePtr.get());
            areaHihat.setSrcInst(memSourcePtr.release());

            displayOut(bufferY, *thumbnailHihatOut);
        }

        else if(torch::equal(yInstr, yCymbals)){

            outFile = juce::File(filesDir.getFullPathName()).getChildFile(name + "_cymbals.wav");
            writerY.reset(formatWav.createWriterFor(new juce::FileOutputStream(outFile),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));
            if (writerY != nullptr)
                writerY->writeFromAudioSampleBuffer(bufferY, 0, bufferY.getNumSamples());


            audioProcessor.transportProcessorCymbals.setSource(memSourcePtr.get());
            transportStateChanged(Stopped, "cymbals");


            playSourceCymbals.reset(memSourcePtr.get());
            areaCymbals.setSrcInst(memSourcePtr.release());

            displayOut(bufferY, *thumbnailCymbalsOut);

        }

       

        DBG("wav scritto!");

    }
       

}

void DrumsDemixEditor::CreateWavQuick(torch::Tensor yDownloadTensor, juce::String path, juce::String name)
{


        DBG("y sizes: ");
        DBG(yDownloadTensor.sizes()[0]);
        DBG(yDownloadTensor.sizes()[1]);


        juce::File file = juce::File(path).getChildFile(name);
        DBG(file.getFullPathName());


         //-Split output tensor in Left & Right
        torch::autograd::variable_list ySplit = torch::split(yDownloadTensor, 1);
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
        juce::AudioBuffer<float> bufferY = juce::AudioBuffer<float>(dataPtrs, 2, yDownloadTensor.sizes()[1]); //need to change last argument to let it be dynamic!
        //bufferOut = juce::AudioBuffer<float>(dataPtrsOut, 2, yKickTensor.sizes()[1]);

        //-Print Wav
        juce::WavAudioFormat formatWav;
        std::unique_ptr<juce::AudioFormatWriter> writerY;

        writerY.reset (formatWav.createWriterFor(new juce::FileOutputStream(file),
                                        44100.0,
                                        bufferY.getNumChannels(),
                                        16,
                                        {},
                                        0));
            if (writerY != nullptr)
                writerY->writeFromAudioSampleBuffer(bufferY, 0, bufferY.getNumSamples());

            //std::unique_ptr<juce::MemoryAudioSource> memSourcePtr(new juce::MemoryAudioSource(bufferY, true, false));


            //audioProcessor.transportProcessorKick.setSource(memSourcePtr.get());
            //transportStateChanged(Stopped, "kick");


            //playSourceKick.reset(memSourcePtr.get());
            //areaKick.setSrcInst(memSourcePtr.release());

            ////displayOut(juce::File("../wavs/testWavJuceKick.wav"), thumbnailKickOut);

            //displayOut(bufferY,*thumbnailKickOut);
       

        DBG("wav scritto!");
    
       

}
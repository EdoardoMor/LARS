#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <cmath>
#include "PluginEditor.h"


class ClickableArea : public juce::TextButton
{
public:

    ClickableArea() : TextButton() {};


    void mouseDoubleClick(const juce::MouseEvent& event)
    {

        if ( (event.eventComponent)->getName() == "areaFull" && fullIsPresent){
            DBG("clickato sample:");
            DBG(makeConversion(event.x, srcFull->getTotalLength()));
            srcFull->setNextReadPosition(makeConversion(event.x, srcFull->getTotalLength()));

        }
        else if ( instIsPresent ) {
            DBG("clickato sample:");
            DBG(makeConversion(event.x, srcInst->getTotalLength()));
            srcInst->setNextReadPosition(makeConversion(event.x, srcInst->getTotalLength()));
        }

    }

    int makeConversion(int eventX, int totLen) {
        return std::floor(((float) ( eventX - 58 )/ 720.0 ) * totLen); //denominator must be equal to thumbnail's length
    }

    void setSrcInst(juce::MemoryAudioSource* sI){
        srcInst = sI;
        if (!instIsPresent) instIsPresent = true;
    }

    void setSrc(juce::AudioFormatReaderSource* sF) {
        srcFull = sF;
        if (!fullIsPresent) fullIsPresent = true;
    }

    void setFilesDir(juce::File fDir) {
        fileDir = fDir;
    }

    void setInFile(juce::String inF) {
        inputFileName = inF;
    }

    
    void mouseDrag(const juce::MouseEvent& e) override
    {
        if ((e.eventComponent)->getName() == "areaKick")
        {
            juce::StringArray path = (fileDir.getChildFile(inputFileName.dropLastCharacters(4) + "_kick.wav")).getFullPathName();
            Container.performExternalDragDropOfFiles(path, true);
        }
        if ((e.eventComponent)->getName() == "areaSnare")
        {
            juce::StringArray path = (fileDir.getChildFile(inputFileName.dropLastCharacters(4) + "_snare.wav")).getFullPathName();
            Container.performExternalDragDropOfFiles(path, true);
        }
        if ((e.eventComponent)->getName() == "areaToms")
        {
            juce::StringArray path = (fileDir.getChildFile(inputFileName.dropLastCharacters(4) + "_toms.wav")).getFullPathName();
            Container.performExternalDragDropOfFiles(path, true);
        }
        if ((e.eventComponent)->getName() == "areaHihat")
        {
            juce::StringArray path = (fileDir.getChildFile(inputFileName.dropLastCharacters(4) + "_hihat.wav")).getFullPathName();
            Container.performExternalDragDropOfFiles(path, true);
        }
        if ((e.eventComponent)->getName() == "areaCymbals")
        {
            juce::StringArray path = (fileDir.getChildFile(inputFileName.dropLastCharacters(4) + "_cymbals.wav")).getFullPathName();
            Container.performExternalDragDropOfFiles(path, true);
        }
    }


private:
    juce::DragAndDropContainer Container;



    juce::MemoryAudioSource* srcInst;
    juce::AudioFormatReaderSource* srcFull;

    juce::File fileDir;
    juce::String inputFileName;

    bool fullIsPresent = false;
    bool instIsPresent = false;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClickableArea);

};
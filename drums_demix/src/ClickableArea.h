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


    ClickableArea() : TextButton() {

    };

    void mouseDoubleClick(const juce::MouseEvent& event)
    {

        if ( (event.eventComponent)->getName() == "areaFull") {
            DBG("clickato sample:");
            DBG(makeConversion(event.x, srcFull->getTotalLength()));
            srcFull->setNextReadPosition(makeConversion(event.x, srcFull->getTotalLength()));

        }
        else {
            DBG("clickato sample:");
            DBG(makeConversion(event.x, srcInst->getTotalLength()));
            srcInst->setNextReadPosition(makeConversion(event.x, srcInst->getTotalLength()));
        }

    }

    int makeConversion(int eventX, int totLen) {
        return std::floor(((float) eventX / 780.0 ) * totLen); //!!! IL NUMERO AL DENOMINATORE DEVE ESSERE PARI ALLA LUNGHEZZA DELLE THUMBNAIL !!!
    }

    void setSrcInst(juce::MemoryAudioSource* sI){
        srcInst = sI;
    }

    void setSrc(juce::AudioFormatReaderSource* sF) {
        srcFull = sF;
    }


private:

    juce::MemoryAudioSource* srcInst;
    juce::AudioFormatReaderSource* srcFull;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ClickableArea);

};
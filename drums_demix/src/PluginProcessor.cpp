/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
DrumsDemixProcessor::DrumsDemixProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
#endif
,  baseFrequency{440}, 
     carrPhase{0}, 
     carrDPhase{0}, 

     //modFreq{0}, 
     modPhase{0}, 
     modDPhase{0}, 
    
     modIndex{1}, 
     modDepth{0}, 
    
   amp{0}, ampTarget{0}, dAmp{0.00001}, ampMax{0.25}
{
}

DrumsDemixProcessor::~DrumsDemixProcessor()
{
}

//==============================================================================
const juce::String DrumsDemixProcessor::getName() const
{
    return JucePlugin_Name;
}

bool DrumsDemixProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool DrumsDemixProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool DrumsDemixProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double DrumsDemixProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int DrumsDemixProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int DrumsDemixProcessor::getCurrentProgram()
{
    return 0;
}

void DrumsDemixProcessor::setCurrentProgram (int index)
{
}

const juce::String DrumsDemixProcessor::getProgramName (int index)
{
    return {};
}

void DrumsDemixProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void DrumsDemixProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    carrDPhase = getDPhase(baseFrequency, getSampleRate());
    modDPhase = getDPhase(baseFrequency*modIndex, getSampleRate());

}

void DrumsDemixProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool DrumsDemixProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void DrumsDemixProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    //////////// 
    // deal with MIDI 

     // transfer any pending notes into the midi messages and 
    // clear pending - these messages come from the addMidi function
    // which the UI might call to send notes from the piano widget
    if (midiToProcess.getNumEvents() > 0){
      midiMessages.addEvents(midiToProcess, midiToProcess.getFirstEventTime(), midiToProcess.getLastEventTime()+1, 0);
      midiToProcess.clear();
    }

    for (const auto metadata : midiMessages){
        auto message = metadata.getMessage();
        if (message.isNoteOn()){
            ampTarget = ampMax;
            baseFrequency = juce::MidiMessage::getMidiNoteInHertz(message.getNoteNumber());
            break;
        }
        if (message.isNoteOff()){
            ampTarget = 0;
        }
    }

    /////////////
    // end of deal with MIDI

    juce::ScopedNoDenormals noDenormals;
    auto totalNumInputChannels  = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear (i, 0, buffer.getNumSamples());

    // in case mod index changed
    // update mod dphase 
    modDPhase = getDPhase(baseFrequency*modIndex, getSampleRate());
    double mod{0};// output of modulator
    for (int channel = 0; channel < totalNumOutputChannels; ++channel){
        if (channel == 0){
            auto* channelData = buffer.getWritePointer (channel);
            int numSamples = buffer.getNumSamples();
            for (int sInd=0;sInd < numSamples; ++sInd){
                // we compute carrDPhase every sample
                // in FM synthesis

                // first deal with the output
                // of the modulator
                mod = std::sin(modPhase);
                mod *= modDepth * baseFrequency; // scale it 
                modPhase += modDPhase; 
                // now add mod output
                // to the base freq
                // and compute phase change 
                // on carrier 
                carrDPhase = getDPhase(baseFrequency + mod, getSampleRate());
                channelData[sInd] = (float) (std::sin(carrPhase) * amp);

                carrPhase += carrDPhase;

                if (amp > ampTarget) amp -= dAmp;
                if (amp < ampTarget) amp += dAmp;
            }
        }
    }

  
 
}

//==============================================================================
bool DrumsDemixProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* DrumsDemixProcessor::createEditor()
{
    return new DrumsDemixEditor (*this);
}

//==============================================================================
void DrumsDemixProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
}

void DrumsDemixProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DrumsDemixProcessor();
}

double DrumsDemixProcessor::getDPhase(double freq, double sampleRate)
{
    double two_pi = 3.1415927 * 2;
    return (two_pi / sampleRate) * freq;   
}
void DrumsDemixProcessor::updateFrequency(double newFreq)
{
    baseFrequency = newFreq; 
}

void DrumsDemixProcessor::updateFMParams(double _modIndex, double _modDepth)
{
    modIndex = _modIndex;
    modDepth = _modDepth;
}


void DrumsDemixProcessor::addMidi(juce::MidiMessage msg, int sampleOffset)
{
  midiToProcess.addEvent(msg, sampleOffset);
}


void DrumsDemixProcessor::setEnvLength(double envLenSecs)
{
    // calculate damp based on length
    if (envLenSecs == 0){// 
        // always on mode
        amp = ampMax;
        dAmp = 0;
    }
    else{
        double envLenSamples = envLenSecs * getSampleRate();
        dAmp = 1 / envLenSamples / 2; 
    }
}

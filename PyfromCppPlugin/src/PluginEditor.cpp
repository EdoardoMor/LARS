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

/*
    try{
        mymodule=torch::jit::load("../python_models/my_scripted_module.pt");
    }
    catch(const c10::Error& e) {
        DBG("error"); //indicate error to calling code
    }

    */
    

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

       /*
       std::vector<torch::jit::IValue> inputs;
       inputs.push_back(torch::rand({28*28}));

       at::Tensor outputs = mymodule.forward(inputs).toTensor();
       std::vector<float> v(outputs.data_ptr<float>(), outputs.data_ptr<float>() + outputs.numel()); //conversione da tensor a std vector
       DBG(v[0]);

       */
        DBG("vai!");

        CallPython("PythonFile2", "helloworld2");

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


void FMPluginEditor::CallPython(string PythonModuleName, string PythonFunctionName)
{
    char* funcname = new char[PythonFunctionName.length() + 1];
	strcpy_s(funcname, PythonFunctionName.length() + 1, PythonFunctionName.c_str());

	char* modname = new char[PythonModuleName.length() + 1];
	strcpy_s(modname, PythonModuleName.length() + 1, PythonModuleName.c_str());

	DBG("initialize the Python interpreter");

	Py_Initialize();

	TCHAR cwd[2048];
	GetCurrentDirectory(sizeof(cwd), cwd);

	DBG("Load the Python module");
	PyObject* my_module = PyImport_ImportModule(modname);

	
	PyErr_Print();

	DBG("Module found");
	DBG(" find function from Python module");

	DBG("Getting address of the funct in Python module");
	PyObject* my_function = PyObject_GetAttrString(my_module, funcname);

	PyErr_Print();

	DBG("Function found");
	DBG("call function from Python module");

	 
	PyObject* my_result = PyObject_CallObject(my_function, NULL);

	PyErr_Print();

	DBG("Your function has been called");

	// Undo all initializations made by Py_Initialize() and subsequent use of Python/C API functions, 
	// and destroy all sub-interpreters (see Py_NewInterpreter() below) that were created and not yet 
	// destroyed since the last call to Py_Initialize(). Ideally, this frees all memory allocated by the Python interpreter.
	// https://docs.python.org/3/c-api/init.html?highlight=py_finalize#c.Py_FinalizeEx
	Py_Finalize();

	delete[] funcname;
	delete[] modname;

}

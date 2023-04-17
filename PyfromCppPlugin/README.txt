PROCEDURE PER FARE PLUGIN CON JUCE CHE CHIAMA SCRIPT PYTHON (WINDOWS)

- da C:\Python310 copiare:
	- python3.dll
	- python3_d.dll
	- python310.dll
	- python310_d.dll

  e incollare nella cartella dove è presente il .exe che si sta creando, per esempio nel caso in cui
  si stia compilando una Debug x64 Standalone questi .dll andranno copiati in:
	C:\Testing_Codes\PyfromCppPlugin\build\pyfromcpp_plugin_artefacts\Debug\Standalone

- nel CmakeLists.txt assicurarsi che siano presenti queste linee di codice (oltre tutte le altre per far funzionare JUCE ofc):
  1)
	target_link_libraries(pyfromcpp_plugin
    		PRIVATE
        	# AudioPluginData           # If we'd created a binary data target, we'd link to it here
        		juce::juce_audio_utils
        		${PYTHON_LIBRARIES}

    		PUBLIC
        		juce::juce_recommended_config_flags
        		juce::juce_recommended_lto_flags
        		juce::juce_recommended_warning_flags)

  2)
	find_package(PythonLibs REQUIRED)

  3)
	include_directories(${PYTHON_INCLUDE_DIRS})

- in VS (es. Standalone): tasto destro su nomeprogetto_Standalone->proprietà->Directory di VC++(nel menù a sx) e poi:
	Editare "Directory di inclusione" aggiungendo "C:\Python310\include"
	Editare "Directory librerie" aggiungendo "C:\Python310\libs"

- #include <Python.h> all'inizio dello script


# Audio classification project

In this project a neutral network is build to recognize enviromental sound events, such as a dog bark or a doorbell ring. To build the network were used pythons libraries such as `keras`, `tensorflow`, `librosa`, `pytorch`, `matplotlib`, `pandas` among others.

Also some libraries to audio manipulation were developed so tasks as record, play and convert audio files can be accomplished.

## Introduction

### Audio processing
As this is a **enviromental audio analysis** project, we will be using known enviromentla audio databases for the training and tuning. Among the used datests are: URBANDOUND8K , MIVIA

I have some audio libs that are able to read, play, plot and extract featctures from `.wav` files
- `librosa` is used for *feature extraction*`
- `simpleaudio` is used for *playing and generating audio files*

The chosen feature extraction goes from firstly, getting the short fourier transform function 
### Data management


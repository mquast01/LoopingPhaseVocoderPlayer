#pragma once

#include <algorithm>
#include <iostream>
#include <complex>
#include <juce_dsp/juce_dsp.h>
#include "CircularBuffer.h"

# define M_PI 3.14159265358979323846  /* pi */

class PhaseVocoder
{
public: 
    using JuceWindow = typename juce::dsp::WindowingFunction<float>;
    using JuceWindowTypes = typename juce::dsp::WindowingFunction<float>::WindowingMethod;

    PhaseVocoder(int bufferSize_, double pitchRatio = 1.0f) : 
        bufferSize(bufferSize_), 
        analysisHopsize((int) ((bufferSize_ / 4) / pitchRatio)),
        synthesisHopsize(bufferSize_ / 4),
        frequencySpectrum(bufferSize_ * 2),
        outputBuffer(bufferSize_, 0),
        lastPhase(bufferSize_, 0),
        accumPhase(bufferSize_, 0),


        windowFunction(std::make_unique<juce::dsp::WindowingFunction<float>>(bufferSize, JuceWindowTypes::hamming)),
        forwardFFT(std::make_unique<juce::dsp::FFT>(log2(bufferSize_))) 
    {
        expectedPhase = expected(0, bufferSize_, bufferSize_);
    }
    int getSize() {
        return bufferSize;
    }

    int getAnalysisHopsize() { return analysisHopsize; }
    int getSynthesisHopsize() { return synthesisHopsize; }

    float* data() {
        return outputBuffer.data();
    }

    /*
    //TODO: implement circular buffer over vector
    void writeFrom (const float *sourceBuffer, int samplesToProcess) {
        for(int i = 0; i < samplesToProcess; i++) {
            analysisBuffer.writeSample(sourceBuffer[i]);
        }
    }
    */

    //Unwrap delta d
    void unwrapArray(float *in, int len) {
        float prev = in[0];
        for (int i = 1; i < len; i++) {
            float d = in[i] - prev;
            d = d > M_PI ? d - 2 * M_PI : (d < -M_PI ? d + 2 * M_PI : d);
            prev = in[i];
            in[i] = in[i-1] + d;
        }
    }

    void processBuffer(const float *sourceBuffer, int samplesToProcess) {
        float* freqPointer = frequencySpectrum.data();
        std::fill(frequencySpectrum.begin(), frequencySpectrum.end(), 0.0f);
        std::copy(sourceBuffer, sourceBuffer + samplesToProcess, frequencySpectrum.data());
        windowFunction->multiplyWithWindowingTable(freqPointer, samplesToProcess);
        forwardFFT->performRealOnlyForwardTransform(freqPointer);
        
        std::vector<float> currentPhase(samplesToProcess, 0);
        std::vector<float> currentMagn(samplesToProcess, 0);
        std::vector<float> deltaPhase(samplesToProcess, 0);   
        //Get frequency spectrum
        for (int i = 0; i < samplesToProcess; i++) {
            std::complex<float> temp;
            temp.real(frequencySpectrum[i * 2]);
            temp.imag(frequencySpectrum[i * 2 + 1]);
            currentPhase[i] = std::arg(temp);
            currentMagn[i] = std::abs(temp);
            
            deltaPhase[i] = currentPhase[i] - lastPhase[i];
            lastPhase[i] = currentPhase[i];

            deltaPhase[i] -= expectedPhase[i];
        }
        unwrapArray(deltaPhase.data(), samplesToProcess);
    
        //Rebuild for iFFT
        for (int i = 0; i < samplesToProcess; i++) {
            accumPhase[i] += static_cast<float>(deltaPhase[i] + expectedPhase[i]) * synthesisHopsize / analysisHopsize;

            float real = std::cos(accumPhase[i]) * currentMagn[i];
            float imag = std::sin(accumPhase[i]) * currentMagn[i];

            frequencySpectrum[i * 2] = real;
            frequencySpectrum[i * 2 + 1] = imag;
        }
        
        forwardFFT->performRealOnlyInverseTransform(freqPointer);
        windowFunction->multiplyWithWindowingTable(freqPointer, samplesToProcess);
        std::copy(freqPointer, freqPointer + samplesToProcess, outputBuffer.data());
    }

    std::vector<float> expected(float start, float end, int num) {
        std::vector<float> result;
        float step = (end - start) / (num - 1);

        for (int i = 0; i < num; ++i) {
            float k = start + i * step;
            result.push_back(k * 2 * M_PI * analysisHopsize / bufferSize);
        }

        return result;
    }
                
private:
    int bufferSize;
    int processBlockCounter = 0;

    int analysisHopsize;
    int synthesisHopsize;
    std::vector<float> frequencySpectrum;
    std::vector<float> outputBuffer;

    std::vector<float> lastPhase;
    std::vector<float> expectedPhase;
    std::vector<float> accumPhase;

    std::unique_ptr<juce::dsp::WindowingFunction< float >> windowFunction;
    std::unique_ptr<juce::dsp::FFT> forwardFFT;

JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(PhaseVocoder)
};



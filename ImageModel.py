from PyQt5 import QtWidgets, QtCore, uic, QtGui, QtPrintSupport
from pyqtgraph import PlotWidget, plot
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *   
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from os import path
import pyqtgraph as pg
import queue as Q
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageModel():

    """
    A class that represents the ImageModel
    """

    def __init__(self, imgPath: str,id):
        """
        :param imgPath: absolute path of the image
        """
        self.imgPath = imgPath
        self.img = cv2.imread(self.imgPath, flags=cv2.IMREAD_GRAYSCALE).T
        self.imgShape = self.img.shape
        self.fourier = np.fft.fft2(self.img)
        self.real = np.real(self.fourier)
        self.imaginary = np.imag(self.fourier)
        self.magnitude = np.abs(self.fourier)
        self.mag_spectrum = np.log10(self.magnitude)
        self.phase = np.angle(self.fourier)
        self.uniformMagnitude = np.ones(self.img.shape)
        self.uniformPhase = np.zeros(self.img.shape)
        self.component_list=[self.mag_spectrum,self.phase,self.real,self.imaginary]

    def mix(self, imageToBeMixed, magnitudeOrRealRatio, phaesOrImaginaryRatio, mode):
        """
        a function that takes ImageModel object mag ratio, phase ration and
        return the magnitude of ifft of the mix
        return type ---> 2D numpy array
        """
        w1 = magnitudeOrRealRatio
        w2 = phaesOrImaginaryRatio
        mixInverse = None

        if mode == 'MagnitudeAndPhase':
            print("Mixing Magnitude and Phase")
            M1 = self.magnitude
            M2 = imageToBeMixed.magnitude

            P1 = self.phase
            P2 = imageToBeMixed.phase

            magnitudeMix = w1*M1 + (1-w1)*M2
            phaseMix = (1-w2)*P1 + w2*P2

            combined = np.multiply(magnitudeMix, np.exp(1j * phaseMix))
            mixInverse = np.real(np.fft.ifft2(combined))
        
        elif mode == 'UniMagnitudeAndPhase':
            print("Mixing uniformMagnitude and Phase")
            M1 = self.uniformMagnitude

            P1 = self.phase
            P2 = imageToBeMixed.phase

            magnitudeMix = M1
            phaseMix = (1-w2)*P1 + w2*P2

            combined = np.multiply(magnitudeMix, np.exp(1j * phaseMix))
            mixInverse = np.real(np.fft.ifft2(combined))
        
        elif mode == 'PhaseAndMagnitude':
            print("Mixing Phase and Magnitude")
            M1 = self.magnitude
            M2 = imageToBeMixed.magnitude

            P1 = self.phase
            P2 = imageToBeMixed.phase

            magnitudeMix= (1-w2)*M1 + w2*M2
            phaseMix = w1*P1  + (1-w1)*P2 

            combined = np.multiply(magnitudeMix, np.exp(1j * phaseMix))
            mixInverse = np.real(np.fft.ifft2(combined))
       
        elif mode == 'PhaseAndUniMagnitude':
            M1 = self.uniformMagnitude 

            P1 = self.phase
            P2 = imageToBeMixed.phase

            magnitudeMix=M1
            phaseMix = w1*P1  + (1-w1)*P2 

            combined = np.multiply(magnitudeMix, np.exp(1j * phaseMix))
            mixInverse = np.real(np.fft.ifft2(combined))
        
        elif mode == 'UniPhaseAndMagnitude':
            M1 = self.magnitude
            M2 = imageToBeMixed.magnitude 

            P1 = self.uniformPhase
            

            magnitudeMix= (1-w2)*M1 + w2*M2
            phaseMix = P1 

            combined = np.multiply(magnitudeMix, np.exp(1j * phaseMix))
            mixInverse = np.real(np.fft.ifft2(combined))

        elif mode == 'MagnitudeAndUniPhase':
            print("Mixing Magnitude and UniPhase")
            M1 = self.magnitude
            M2 = imageToBeMixed.magnitude

            P1 = self.uniformPhase

            magnitudeMix = w1*M1 + (1-w1)*M2
            phaseMix = P1

            combined = np.multiply(magnitudeMix, np.exp(1j * phaseMix))
            mixInverse = np.real(np.fft.ifft2(combined))
                
        elif mode == 'RealAndImaginary':
            print("Mixing Real and Imaginary")
            R1 = self.real
            R2 = imageToBeMixed.real

            I1 = self.imaginary
            I2 = imageToBeMixed.imaginary

            realMix = w1*R1 + (1-w1)*R2
            imaginaryMix = (1-w2)*I1 + w2*I2

            combined = realMix + imaginaryMix * 1j
            mixInverse = np.real(np.fft.ifft2(combined))
                
        elif mode == 'ImaginaryAndReal':
            print("Mixing Imaginary and Real")
            R1 = self.real
            R2 = imageToBeMixed.real

            I1 = self.imaginary
            I2 = imageToBeMixed.imaginary

            realMix = (1-w2)*R1 + w2*R2
            imaginaryMix = w1*I1+(1-w1)*I2

            combined = realMix + imaginaryMix * 1j
            mixInverse = np.real(np.fft.ifft2(combined))
                  
        return abs(mixInverse)
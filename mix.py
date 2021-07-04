from PyQt5 import QtWidgets, QtCore, uic, QtGui, QtPrintSupport
from pyqtgraph import PlotWidget, plot
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *   
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from os import path
import pyqtgraph as pg
from ImageModel import ImageModel
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

import logging 


logging.basicConfig(level=logging.INFO, filename="logging.log", format='%(asctime)s:%(levelname)s:%(message)s', filemode='w') 

  
#Setting the threshold of logger to DEBUG 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MAIN_WINDOW,_=loadUiType(path.join(path.dirname(__file__),"ImageMixer.ui"))

class MainApp(QMainWindow,MAIN_WINDOW):
  
    def __init__(self,parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.menuBar()
        self.sliders=[self.horizontalSlider,self.horizontalSlider_2]       
        self.imageWindows=[self.viewFirstimg,self.viewSecondimg]
        self.images={}
        self.image_Component_Windows=[self.viewFirstComponent,self.viewSecondComponent]
        self.OutputWindows={"Output 1":self.viewOutput1,"Output 2":self.viewOutput2}
        self.Windows_List=[ self.viewFirstimg,self.viewSecondimg,self.viewOutput1,self.viewOutput2,self.viewFirstComponent,self.viewSecondComponent]
        self.ImageModels={}
        self.output_comboBox=[self.comboBox_4,self.comboBox_5]
        self.mix_comp=['Magnitude','Phase','Real','Imaginary','UniMagnitude','UniPhase']
        self.imagePath=['Task_3','Task_3']
        self.ratio=[0,0]
        self.width=[1,2]
        self.height=[1,2]
        self.component_list={}
        self.img_comboBox=[self.comboBox, self.comboBox2]
        self.mix_comboBox=[self.comboBox_3,self.comboBox_4,self.comboBox_5,self.comboBox_6,self.comboBox_7]
        self.image_view_congigrution()
        self.mix_comboBox_connection()
        self.sliders_view_congigrution()

        logger.info("The Application started successfully")
   
    def image_view_congigrution(self):
        for i in range (len(self.Windows_List)):
                self.Windows_List[i].ui.histogram.hide()
                self.Windows_List[i].ui.roiBtn.hide()
                self.Windows_List[i].ui.menuBtn.hide()
                self.Windows_List[i].ui.roiPlot.hide()
       
    def sliders_view_congigrution(self):
        for i in range (2):
            self.sliders[i].setMinimum(0)
            self.sliders[i].setMaximum(100) 
            self.sliders[i].setSingleStep(10)
            self.sliders[i].valueChanged.connect(self.valuechange) 
    
    def  mix_comboBox_connection(self):
        for i in range (5):
            self.mix_comboBox[i].activated.connect(self.valuechange)


    def menuBar(self):
        self.openFirstImg.triggered.connect(lambda:self.browse(0))
        self.openSecondImg.triggered.connect(lambda:self.browse(1))
    
    def show_popup(self, message):
        msg = QMessageBox()
        msg.setWindowTitle("Warning")
        msg.setText(message)
        msg.setIcon(QMessageBox.Warning)
        x = msg.exec_()
    
    def browse(self,number):
        logger.info("Browsing the files...")
        fileName = QtWidgets.QFileDialog.getOpenFileName(None,"QFileDialog.getOpenFileName()", "","Image files (*.jpg)")
        self.imagePath[number]=fileName[0]
        print(self.imagePath[number])
        im = Image.open(self.imagePath[number])
        width, height = im.size
        self.width[number]=width
        self.height[number]=height
        if number==1:
                if ((self.width[number]!=self.width[number-1])&(self.height[number]!=self.height[number-1])):
                    logger.warning("The two images must be the same size!")
                    self.show_popup("The two images must be the same size!")
                    self.browse(1)
                else: 
                    pass
        self.images['Image'+str(number+1)]=cv2.imread(self.imagePath[number],0) 
        self.updateCombosChanged(self.images['Image'+str(number+1)].T,number)
        self.displayImage(self.images['Image'+str(number+1)].T,self.imageWindows[number])
        self.ImageModels['Image'+str(number+1)]=ImageModel(self.imagePath[number],number)
        self.img_comboBox[number].currentIndexChanged.connect(lambda:self.ComboBox_function(number))
        logger.info(f"An image has been added successfully")

    def updateCombosChanged(self,data,number):
        fft = np.fft.fft2(data)
        fShift = np.fft.fftshift(fft)
        Magnitude = 20 * np.log(np.abs(fShift))
        phase = np.angle(fShift)
        real = 20 * np.log(np.real(fShift))
        imaginary = np.imag(fShift)
        self.component_list['Image'+str(number+1)+'_list']=[Magnitude,phase,real,imaginary]

    def displayImage(self, data, widget):
        """
        Display the given data
        :param data: 2d numpy array
        :param widget: ImageView object
        :return:
        """
        widget.setImage(data)
        widget.ui.roiPlot.hide()  

    def ComboBox_function(self,number):
          for i in range (4):
            if self.img_comboBox[number].currentText()==self.mix_comp[i]:
                 self.displayImage(self.component_list['Image'+str(number+1)+'_list'][i].T,self.image_Component_Windows[number])
  
    def valuechange(self):
        for i in range (2):
            self.ratio[i]=self.sliders[i].value()/ 100.0
        mode=self.comboBox_6.currentText()+str('And')+self.comboBox_7.currentText()
        self.combobox_update (self.comboBox_6.currentText(),self.comboBox_7.currentText())
        print(mode)

        # if self.comboBox_6.currentText() != 'Select Component' and self.comboBox_7.currentText() != ('Select A Component' | 'Choose Component') :
        if self.comboBox_6.currentText() != 'Select Component' and self.comboBox_7.currentText() != ('Select A Component') :
            logger.info('Current mode is {} and {}' . format(self.comboBox_6.currentText() , self.comboBox_7.currentText()))
            logger.info('mixing {} of {} with {} of {} ' . format(self.sliders[0].value() , self.comboBox_6.currentText() ,self.sliders[1].value() ,  self.comboBox_7.currentText()) )
            
        mixInverse=self.ImageModels[self.comboBox_4.currentText()].mix(self.ImageModels[self.comboBox_5.currentText()],self.ratio[0],self.ratio[1],mode)
        logger.info('Component 1 is {} , component 2 is {}' . format(self.comboBox_4.currentText() , self.comboBox_5.currentText()))
        self.displayImage(mixInverse, self.OutputWindows[self.comboBox_3.currentText()])
       
    def combobox_update (self,comp1,comp2):
        self.comboBox_7.clear()
        self.comboBox_7.addItem("Select A Component")

        if  comp1=='Magnitude':
            self.comboBox_7.addItem("Phase")
            self.comboBox_7.addItem("UniPhase")
            self.comboBox_7.setCurrentText(comp2)
            
        elif  comp1=='Phase':
            self.comboBox_7.addItem("Magnitude")
            self.comboBox_7.addItem("UniMagnitude")
            self.comboBox_7.setCurrentText(comp2)
            
        elif  comp1=='UniMagnitude':
            self.comboBox_7.addItem("Phase")
            self.comboBox_7.setCurrentText(comp2)
              
        elif  comp1=='UniPhase':
            self.comboBox_7.addItem("Magnitude")
            self.comboBox_7.setCurrentText(comp2)
            
        elif  comp1=='Real':
            self.comboBox_7.addItem("Imaginary")
            self.comboBox_7.setCurrentText(comp2)
             
        elif  comp1=='Imaginary':
            self.comboBox_7.addItem("Real")
            self.comboBox_7.setCurrentText(comp2)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__=='__main__':
    main()
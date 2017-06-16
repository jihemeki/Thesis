from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.properties import *
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
import cv2
import numpy as np
import sudokuS
import time
import os
import sys
import datetime
import imageS


def fillUp(text):
    if text == '':
        text = '0'
    return text

class tInput(TextInput):
    multiline = False
    font_size = NumericProperty('40sp')
    input_filter = ObjectProperty('int', allownone=True)
    def insert_text(self, substring, from_undo=False):
        if len(self.text) < 1:
            s = substring
            return super(tInput, self).insert_text(s, from_undo=from_undo)

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class Sudoku(BoxLayout):
    cameraStatus = 0
    grid = ''
    a1 = ObjectProperty(None)
    a2 = ObjectProperty(None)
    a3 = ObjectProperty(None)
    a4 = ObjectProperty(None)
    a5 = ObjectProperty(None)
    a6 = ObjectProperty(None)
    a7 = ObjectProperty(None)
    a8 = ObjectProperty(None)
    a9 = ObjectProperty(None)
    b1 = ObjectProperty(None)
    b2 = ObjectProperty(None)
    b3 = ObjectProperty(None)
    b4 = ObjectProperty(None)
    b5 = ObjectProperty(None)
    b6 = ObjectProperty(None)
    b7 = ObjectProperty(None)
    b8 = ObjectProperty(None)
    b9 = ObjectProperty(None)
    c1 = ObjectProperty(None)
    c2 = ObjectProperty(None)
    c3 = ObjectProperty(None)
    c4 = ObjectProperty(None)
    c5 = ObjectProperty(None)
    c6 = ObjectProperty(None)
    c7 = ObjectProperty(None)
    c8 = ObjectProperty(None)
    c9 = ObjectProperty(None)
    d1 = ObjectProperty(None)
    d2 = ObjectProperty(None)
    d3 = ObjectProperty(None)
    d4 = ObjectProperty(None)
    d5 = ObjectProperty(None)
    d6 = ObjectProperty(None)
    d7 = ObjectProperty(None)
    d8 = ObjectProperty(None)
    d9 = ObjectProperty(None)
    e1 = ObjectProperty(None)
    e2 = ObjectProperty(None)
    e3 = ObjectProperty(None)
    e4 = ObjectProperty(None)
    e5 = ObjectProperty(None)
    e6 = ObjectProperty(None)
    e7 = ObjectProperty(None)
    e8 = ObjectProperty(None)
    e9 = ObjectProperty(None)
    f1 = ObjectProperty(None)
    f2 = ObjectProperty(None)
    f3 = ObjectProperty(None)
    f4 = ObjectProperty(None)
    f5 = ObjectProperty(None)
    f6 = ObjectProperty(None)
    f7 = ObjectProperty(None)
    f8 = ObjectProperty(None)
    f9 = ObjectProperty(None)
    g1 = ObjectProperty(None)
    g2 = ObjectProperty(None)
    g3 = ObjectProperty(None)
    g4 = ObjectProperty(None)
    g5 = ObjectProperty(None)
    g6 = ObjectProperty(None)
    g7 = ObjectProperty(None)
    g8 = ObjectProperty(None)
    g9 = ObjectProperty(None)
    h1 = ObjectProperty(None)
    h2 = ObjectProperty(None)
    h3 = ObjectProperty(None)
    h4 = ObjectProperty(None)
    h5 = ObjectProperty(None)
    h6 = ObjectProperty(None)
    h7 = ObjectProperty(None)
    h8 = ObjectProperty(None)
    h9 = ObjectProperty(None)
    i1 = ObjectProperty(None)
    i2 = ObjectProperty(None)
    i3 = ObjectProperty(None)
    i4 = ObjectProperty(None)
    i5 = ObjectProperty(None)
    i6 = ObjectProperty(None)
    i7 = ObjectProperty(None)
    i8 = ObjectProperty(None)
    i9 = ObjectProperty(None)

    label = ObjectProperty(None)

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def dismiss_popup(self):
        self._popup.dismiss()

    def load(self, path, filename):
        gr = self.getGrid()
        for k in gr.keys():
            gr[k].background_color = (1,1,1,1) #SetBGcolor
	try:
            with open(os.path.join(path, filename[0])) as stream: 
                grid = stream.read()
	        if filename[0].endswith("txt"):
                     try:
                         grid = sudokuS.grid_values(grid)
     
                         for k, v in grid.items():
                             if v not in '123456789':
                                 grid[k] = ''
   
                         gr = self.getGrid()
                         for k in gr.keys():
                             gr[k].text = grid[k]
     
                         self.label.text = 'Loaded successfully.'
                     except AssertionError:
                         self.label.text = 'The content of input file is invalid.' #txt

	        elif filename[0].endswith(('png','jpg','jpeg')):
		    img = cv2.imread(os.path.join(path, filename[0])) 
		    reader = imageS.OCRmodelClass()
		    status = imageS.solverStatusClass()
   	    	    puzzle = imageS.puzzleStatusClass()
     	   	    image = imageS.imageClass(img)
		    flag = 1
	            while status.puzzleFound == False:
                        image.captureImage(status)
                        if status.restart == True:
			    flag = 0
                            break
     	            while status.puzzleRead == False and status.puzzleFound == True:
                        image.captureImage(status)
                        image.perspective()
	                image.warp()
                        reader.OCR(status,image,puzzle)
          	        if status.restart == True:
			    flag = 0		   
              	            break
            	        elif np.array_equal(puzzle.SDKarray,puzzle.last):
                	    status.beginSolver = True
                	    status.puzzleRead = True
            	        else:
                	    puzzle.last = np.copy(puzzle.SDKarray)

		    if flag == 0:
		        self.label.text = "Cannot recognize the Sudoku"
		    else:
		        grid = puzzle.getSDKString()
                        grid = sudokuS.grid_values(grid)
       
                        for k, v in grid.items():
                            if v not in '123456789':
                                grid[k] = ''
    
                        gr = self.getGrid()
                        for k in gr.keys():
                           gr[k].text = grid[k]

                        self.label.text = 'Loaded successfully.' #image

	        else:
		    print "Fuckkkk" #FileTypeErr

            self.dismiss_popup()
	except IndexError:
	    self.dismiss_popup()
	    self.label.text = 'Please choose a file.'

    def getGrid(self):
        dict =  {'A1': self.a1, 'A2': self.a2, 'A3': self.a3, 'A4': self.a4, 'A5': self.a5, 'A6': self.a6, 'A7': self.a7, 'A8': self.a8, 'A9': self.a9,
                 'B1': self.b1, 'B2': self.b2, 'B3': self.b3, 'B4': self.b4, 'B5': self.b5, 'B6': self.b6, 'B7': self.b7, 'B8': self.b8, 'B9': self.b9,
                 'C1': self.c1, 'C2': self.c2, 'C3': self.c3, 'C4': self.c4, 'C5': self.c5, 'C6': self.c6, 'C7': self.c7, 'C8': self.c8, 'C9': self.c9,
                 'D1': self.d1, 'D2': self.d2, 'D3': self.d3, 'D4': self.d4, 'D5': self.d5, 'D6': self.d6, 'D7': self.d7, 'D8': self.d8, 'D9': self.d9,
                 'E1': self.e1, 'E2': self.e2, 'E3': self.e3, 'E4': self.e4, 'E5': self.e5, 'E6': self.e6, 'E7': self.e7, 'E8': self.e8, 'E9': self.e9,
                 'F1': self.f1, 'F2': self.f2, 'F3': self.f3, 'F4': self.f4, 'F5': self.f5, 'F6': self.f6, 'F7': self.f7, 'F8': self.f8, 'F9': self.f9,
                 'G1': self.g1, 'G2': self.g2, 'G3': self.g3, 'G4': self.g4, 'G5': self.g5, 'G6': self.g6, 'G7': self.g7, 'G8': self.g8, 'G9': self.g9,
                 'H1': self.h1, 'H2': self.h2, 'H3': self.h3, 'H4': self.h4, 'H5': self.h5, 'H6': self.h6, 'H7': self.h7, 'H8': self.h8, 'H9': self.h9,
                 'I1': self.i1, 'I2': self.i2, 'I3': self.i3, 'I4': self.i4, 'I5': self.i5, 'I6': self.i6, 'I7': self.i7, 'I8': self.i8, 'I9': self.i9
                 }
        return dict

    def getStringGrid(self):
        self.grid = fillUp(self.a1.text) + fillUp(self.a2.text) + fillUp(self.a3.text) + fillUp(self.a4.text) + fillUp(self.a5.text) + fillUp(self.a6.text) + fillUp(self.a7.text) + fillUp(self.a8.text) + fillUp(self.a9.text)
        self.grid = self.grid + fillUp(self.b1.text) + fillUp(self.b2.text) + fillUp(self.b3.text) + fillUp(self.b4.text) + fillUp(self.b5.text) + fillUp(self.b6.text) + fillUp(self.b7.text) + fillUp(self.b8.text) + fillUp(self.b9.text)
        self.grid = self.grid + fillUp(self.c1.text) + fillUp(self.c2.text) + fillUp(self.c3.text) + fillUp(self.c4.text) + fillUp(self.c5.text) + fillUp(self.c6.text) + fillUp(self.c7.text) + fillUp(self.c8.text) + fillUp(self.c9.text)
        self.grid = self.grid + fillUp(self.d1.text) + fillUp(self.d2.text) + fillUp(self.d3.text) + fillUp(self.d4.text) + fillUp(self.d5.text) + fillUp(self.d6.text) + fillUp(self.d7.text) + fillUp(self.d8.text) + fillUp(self.d9.text)
        self.grid = self.grid + fillUp(self.e1.text) + fillUp(self.e2.text) + fillUp(self.e3.text) + fillUp(self.e4.text) + fillUp(self.e5.text) + fillUp(self.e6.text) + fillUp(self.e7.text) + fillUp(self.e8.text) + fillUp(self.e9.text)
        self.grid = self.grid + fillUp(self.f1.text) + fillUp(self.f2.text) + fillUp(self.f3.text) + fillUp(self.f4.text) + fillUp(self.f5.text) + fillUp(self.f6.text) + fillUp(self.f7.text) + fillUp(self.f8.text) + fillUp(self.f9.text)
        self.grid = self.grid + fillUp(self.g1.text) + fillUp(self.g2.text) + fillUp(self.g3.text) + fillUp(self.g4.text) + fillUp(self.g5.text) + fillUp(self.g6.text) + fillUp(self.g7.text) + fillUp(self.g8.text) + fillUp(self.g9.text)
        self.grid = self.grid + fillUp(self.h1.text) + fillUp(self.h2.text) + fillUp(self.h3.text) + fillUp(self.h4.text) + fillUp(self.h5.text) + fillUp(self.h6.text) + fillUp(self.h7.text) + fillUp(self.h8.text) + fillUp(self.h9.text)
        self.grid = self.grid + fillUp(self.i1.text) + fillUp(self.i2.text) + fillUp(self.i3.text) + fillUp(self.i4.text) + fillUp(self.i5.text) + fillUp(self.i6.text) + fillUp(self.i7.text) + fillUp(self.i8.text) + fillUp(self.i9.text)


    def check(self):
        gr = self.getGrid()
        self.getStringGrid()
        flag = 1
        v1 = ''
        v2 = ''
        for k1 in gr.keys():
            for k2 in sudokuS.peers[k1]:
                if (gr[k1].text == gr[k2].text) and (gr[k1].text != ''):
                    v1, v2 = k1, k2
                    flag = 0
                    break
            if flag == 0:
                break

        if flag == 1:
            if sudokuS.solve(self.grid) == False:
                self.label.text = 'This Sudoku does not have solution'
                for k in gr.keys():
                    gr[k].background_color = (225, 0.5, 0.5, 0.65)
            else:
                self.label.text = 'This Sudoku is valid and has solution'
                for k in gr.keys():
                    gr[k].background_color = (0.6, 225, 1, 1)
        else:
            for k in gr.keys():
                gr[k].background_color = (1, 1, 1, 1)
            gr[v1].background_color = (225, 0.5, 0.5, 0.65)
            gr[v2].background_color = (225, 0.5, 0.5, 0.65)
            self.label.text = 'This Sudoku input is invalid.'


    def hint(self):
        gr = self.getGrid()
        self.getStringGrid()
        flag = 1
        v1 = ''
        v2 = ''
        for k1 in gr.keys():
            for k2 in sudokuS.peers[k1]:
                if (gr[k1].text == gr[k2].text) and (gr[k1].text != ''):
                    v1, v2 = k1, k2
                    flag = 0
                    break
            if flag == 0:
                break

        if flag == 1: # SolvedGrid = sudokuS.solve(self.grid)
			# if SolvedGrid == False
            if sudokuS.solve(self.grid) == False:
                self.label.text = 'This Sudoku does not have solution'
                for k in gr.keys():
                    gr[k].background_color = (225, 0.5, 0.5, 0.65)
            else: #hint
                hintCell = ''
                SolvedGrid = sudokuS.solve(self.grid)
                for k in gr.keys():
                    if gr[k].text == '':
                        hintCell = k
                        break
                if hintCell == '':
                    self.label.text = 'This Sudoku has been solved'
                    for k in gr.keys():
                            gr[k].background_color = (0.5, 0.5, 255, 0.8)
                else:
                    for k in gr.keys():
                        gr[k].background_color = (1, 1, 1, 1)
                    gr[hintCell].background_color = (255,255,0,0.65)
                    gr[hintCell].text = SolvedGrid[hintCell]
                    self.label.text = 'Hint!'
        else:
            for k in gr.keys():
                gr[k].background_color = (1, 1, 1, 1)
            gr[v1].background_color = (225, 0.5, 0.5, 0.65)
            gr[v2].background_color = (225, 0.5, 0.5, 0.65)
            self.label.text = 'This Sudoku input is invalid.'

    def solve(self):
        gr = self.getGrid()
        self.getStringGrid()
        flag = 1
        v1 = ''
        v2 = ''
        for k1 in gr.keys():
            for k2 in sudokuS.peers[k1]:
                if (gr[k1].text == gr[k2].text) and (gr[k1].text != ''):
                    v1,v2 = k1,k2
                    flag = 0
                    break
            if flag == 0:
                break

        if flag == 1:
            if sudokuS.solve(self.grid) == False:
                self.label.text = 'This Sudoku does not have solution'
                for k in gr.keys():
                    gr[k].background_color = (225, 0.5, 0.5, 0.65)
            else:
                t1 = time.clock()
                SolvedGrid = sudokuS.solve(self.grid)
                t2 = time.clock()

                for k in gr.keys():
                    gr[k].background_color = (1, 1, 1, 1)
                for k in gr.keys():
                    if gr[k].text == '':
                        gr[k].background_color = (0.5, 0.5, 255, 0.8)
                # Assign back
                for k1, v1 in gr.items():
                    for k2, v2 in SolvedGrid.items():
                        if k1 == k2:
                            gr[k1].text = SolvedGrid[k2]
                            break
                t = t2 - t1
                self.label.text = 'Time:' + str(t)
        else:
            for k in gr.keys():
                gr[k].background_color = (1, 1, 1, 1)
            gr[v1].background_color = (225, 0.5, 0.5, 0.65)
            gr[v2].background_color = (225, 0.5, 0.5, 0.65)
            self.label.text = 'This Sudoku input is invalid.'


    def save(self):
	self.getStringGrid()
	f = open("Sudoku/"+str(datetime.datetime.now())+".txt","w")
	f.write(self.grid)
	f.close()
	self._popup.dismiss()

    def confirm(self):
        content = BoxLayout(orientation='horizontal',spacing = 10)

	dissmissB = Button(text='Cancel',font_size=30)
	dissmissB.bind(on_press = lambda *args: self._popup.dismiss())

	confirmB = Button(text='Ok',font_size=30)
	confirmB.bind(on_press = lambda *args: self.save())

	content.add_widget(dissmissB)
	content.add_widget(confirmB)

        self._popup = Popup(title="Confirmation", content=content,
                            size_hint=(None, None), size=(300,200))

        self._popup.open()


    def camera(self):
	self.cameraStatus = 1
	print self.cameraStatus
	App.get_running_app().stop()
	        
        
    def clear(self):
        gr = self.getGrid()
        for k in gr.keys():
            gr[k].text = ''
            gr[k].background_color = (1,1,1,1)
        self.label.text = 'Notification.'
	

class LuuApp(App):
    def build(self):
        return Sudoku()
	
def main():
    a = LuuApp()
    a.run()	

if __name__ == '__main__': main()







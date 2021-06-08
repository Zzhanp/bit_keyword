import sys
#from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTextEdit, QGroupBox, QPushButton, QLabel, QTableWidgetItem, QTableWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
#from PyQt5.QtCore import QEventLoop, QTimer
from Ui_keyword import *
from PyQt5 import QtCore, QtGui
from hanlptest import *
import pandas as pd
import numpy as np
import csv
import xlwt

class EmittingStr(QtCore.QObject):
   textWritten = QtCore.pyqtSignal(str) #定义一个发送str的信号
   def write(self, text):
      self.textWritten.emit(str(text))

class MyWindow(QMainWindow, Ui_MainWindow):
   def __init__(self, parent=None):
      super(MyWindow, self).__init__(parent)
      self.setupUi(self)
      sys.stdout = EmittingStr(textWritten=self.outputWritten)
      sys.stderr = EmittingStr(textWritten=self.outputWritten)
   def read(self): 
      file_name,ok=QFileDialog.getOpenFileName(self,'读取','./') 
      if ok : 
         f=open(file_name,'r') 
         self.textBrowser_file.append(file_name) 

   def keyword(self):
      key_num = self.numlineEdit.text()
      file_path = self.textBrowser_file.toPlainText().replace('/','\\')
      key_result = extract(file_path, key_num )
      #self.textBrowser_result.append(str(key_result))
   
   def wordcloud(self):
      txtname = self.textBrowser_file.toPlainText().replace('/','\\')
      truepath =txtname.split("\\")[-1]
      title1 = truepath.split('.')[0]
      savepath = './word_cloud'
      imgName = os.path.join(savepath, title1 + '.png')
      img = QtGui.QPixmap(imgName).scaled(self.wordimage.width(), self.wordimage.height())
      self.wordimage.setPixmap(img)

   def history(self):
      path_openfile_name = 'output.xlsx'
      if len(path_openfile_name) > 0:
         input_table = pd.read_excel(path_openfile_name)
         input_table_rows = input_table.shape[0]
         input_table_colunms = input_table.shape[1]
         input_table_header = input_table.columns.values.tolist()

        ###===========读取表格，转换表格，============================================
        ###======================给csvWidget设置行列表头============================

         self.csvWidget.setColumnCount(input_table_colunms)
         self.csvWidget.setRowCount(input_table_rows)
         self.csvWidget.setHorizontalHeaderLabels(input_table_header)

        ###======================给csvWidget设置行列表头============================

        ###================遍历表格每个元素，同时添加到csvWidget中========================
         for i in range(input_table_rows):
            input_table_rows_values = input_table.iloc[[i]]
            input_table_rows_values_array = np.array(input_table_rows_values)
            input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
            for j in range(input_table_colunms):
               input_table_items_list = input_table_rows_values_list[j]

        ###==============将遍历的元素添加到csvWidget中并显示=======================

               input_table_items = str(input_table_items_list)
               newItem = QTableWidgetItem(input_table_items) 
               newItem.setTextAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
               self.csvWidget.setItem(i, j, newItem)
               self.csvWidget.resizeColumnsToContents()   #将列调整到跟内容大小相匹配
               self.csvWidget.resizeRowsToContents() 

      ###================遍历表格每个元素，同时添加到csvWidget中========================
      else:
         self.centralWidget.show()
   
   def dehistory(self):
      del_data()

   

   def outputWritten(self, text):
      cursor = self.textBrowser_result.textCursor()
      cursor.movePosition(QtGui.QTextCursor.End)
      cursor.insertText(text)
      QApplication.processEvents()
      self.textBrowser_result.setTextCursor(cursor)
      self.textBrowser_result.ensureCursorVisible()



class ControlBoard(QMainWindow, Ui_MainWindow):
   def __init__(self):
      super(ControlBoard, self).__init__()
      self.setupUi(self)
    # 下面将输出重定向到textBrowser中
      sys.stdout = EmittingStr(textWritten=self.outputWritten)
      sys.stderr = EmittingStr(textWritten=self.outputWritten)

'''def csv_to_xlsx():
    csv = pd.read_csv('output.csv', header=None, usecols=[0,1],index_col=None)
    csv.to_excel('output.xlsx', sheet_name='data',header=None,index=None)'''
   
if __name__ == '__main__':
   app = QApplication(sys.argv)
   myWin = MyWindow()
   myWin.show()
   sys.exit(app.exec_())

#v1.0 增加history功能
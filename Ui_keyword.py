# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\my code\python\keyword9.0\keyword1.0\keyword.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(957, 611)
        MainWindow.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(0, 0, 951, 51))
        self.groupBox_2.setStyleSheet("background-color: rgb(34, 133, 255);")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(13, 11, 201, 31))
        self.label_3.setStyleSheet("font: 87 16pt \"可口可乐在乎体 楷体\";\n"
"color: rgb(255, 255, 255);")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(220, 20, 211, 21))
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);\n"
"font: 9pt \"微软雅黑\";")
        self.label_4.setObjectName("label_4")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_3.setGeometry(QtCore.QRect(890, 10, 16, 16))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_4.setGeometry(QtCore.QRect(910, 10, 16, 16))
        self.pushButton_4.setObjectName("pushButton_4")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(140, 50, 781, 531))
        self.stackedWidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_0 = QtWidgets.QWidget()
        self.page_0.setObjectName("page_0")
        self.textBrowser_result = QtWidgets.QTextBrowser(self.page_0)
        self.textBrowser_result.setGeometry(QtCore.QRect(130, 290, 511, 231))
        self.textBrowser_result.setObjectName("textBrowser_result")
        self.label = QtWidgets.QLabel(self.page_0)
        self.label.setGeometry(QtCore.QRect(130, 30, 121, 41))
        self.label.setStyleSheet("background-color: rgb(39, 133, 255);\n"
"font: 10pt \"微软雅黑\";")
        self.label.setObjectName("label")
        self.extrapushButton = QtWidgets.QPushButton(self.page_0)
        self.extrapushButton.setGeometry(QtCore.QRect(490, 50, 120, 120))
        self.extrapushButton.setStyleSheet("QPushButton{\n"
"    color:White;\n"
"    border-style:solid;\n"
"    border-radius:60px;\n"
"    font-family:微软雅黑;\n"
"    background:#2285FF;\n"
"}\n"
"QPushButton:hover{\n"
"    background:#00aaFF;\n"
"}\n"
"QPushButton:pressed{\n"
"    background:#00aaFF;\n"
"}")
        self.extrapushButton.setObjectName("extrapushButton")
        self.label_2 = QtWidgets.QLabel(self.page_0)
        self.label_2.setGeometry(QtCore.QRect(130, 230, 121, 41))
        self.label_2.setStyleSheet("font: 9pt \"微软雅黑\";")
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.page_0)
        self.label_5.setGeometry(QtCore.QRect(130, 180, 111, 31))
        self.label_5.setStyleSheet("font: 12pt \"微软雅黑\";")
        self.label_5.setObjectName("label_5")
        self.numlineEdit = QtWidgets.QLineEdit(self.page_0)
        self.numlineEdit.setGeometry(QtCore.QRect(260, 190, 31, 20))
        self.numlineEdit.setObjectName("numlineEdit")
        self.toolButton = QtWidgets.QToolButton(self.page_0)
        self.toolButton.setGeometry(QtCore.QRect(330, 100, 41, 41))
        self.toolButton.setObjectName("toolButton")
        self.textBrowser_file = QtWidgets.QTextBrowser(self.page_0)
        self.textBrowser_file.setGeometry(QtCore.QRect(130, 90, 181, 51))
        self.textBrowser_file.setObjectName("textBrowser_file")
        self.clearButton = QtWidgets.QPushButton(self.page_0)
        self.clearButton.setGeometry(QtCore.QRect(10, 490, 75, 23))
        self.clearButton.setStyleSheet("background-color: rgb(34, 133, 255);\n"
"color: rgb(255, 255, 255);\n"
"font: 9pt \"微软雅黑\";")
        self.clearButton.setObjectName("clearButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.page_0)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 450, 75, 23))
        self.pushButton_2.setStyleSheet("font: 9pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(34, 133, 255);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.wordButton = QtWidgets.QPushButton(self.page_0)
        self.wordButton.setGeometry(QtCore.QRect(10, 410, 75, 23))
        self.wordButton.setStyleSheet("font: 9pt \"微软雅黑\";\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(34, 133, 255);")
        self.wordButton.setObjectName("wordButton")
        self.stackedWidget.addWidget(self.page_0)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.wordimage = QtWidgets.QLabel(self.page_2)
        self.wordimage.setGeometry(QtCore.QRect(20, 40, 701, 471))
        self.wordimage.setText("")
        self.wordimage.setObjectName("wordimage")
        self.label_6 = QtWidgets.QLabel(self.page_2)
        self.label_6.setGeometry(QtCore.QRect(10, 0, 51, 41))
        self.label_6.setObjectName("label_6")
        self.stackedWidget.addWidget(self.page_2)
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.historyButton = QtWidgets.QPushButton(self.page)
        self.historyButton.setGeometry(QtCore.QRect(0, 20, 93, 28))
        self.historyButton.setObjectName("historyButton")
        self.csvWidget = QtWidgets.QTableWidget(self.page)
        self.csvWidget.setGeometry(QtCore.QRect(0, 60, 791, 461))
        self.csvWidget.setObjectName("csvWidget")
        self.csvWidget.setColumnCount(0)
        self.csvWidget.setRowCount(0)
        self.dehistoryButton = QtWidgets.QPushButton(self.page)
        self.dehistoryButton.setGeometry(QtCore.QRect(110, 20, 93, 28))
        self.dehistoryButton.setObjectName("dehistoryButton")
        self.stackedWidget.addWidget(self.page)
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setObjectName("page_1")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.page_1)
        self.textBrowser_2.setGeometry(QtCore.QRect(0, 0, 741, 521))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.stackedWidget.addWidget(self.page_1)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(0, 50, 124, 521))
        self.listWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.listWidget.setAutoFillBackground(False)
        self.listWidget.setStyleSheet("QListWidget {\n"
"    font: 12pt \"微软雅黑\";\n"
"    min-width: 120px;\n"
"    max-width: 120px;\n"
"    color: Black;\n"
"    background: #F5F5F5;\n"
"}\n"
"\n"
"QListWidget::Item:selected {\n"
"    background: lightGray;\n"
"    border-left: 5px solid #2285ff;\n"
"}\n"
"HistoryPanel:hover {\n"
"    background: rgb(52, 52, 52);\n"
"}\n"
"QListWidget{text-align:right}")
        self.listWidget.setLineWidth(0)
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget.addItem(item)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.listWidget.setCurrentRow(-1)
        self.pushButton_3.clicked.connect(MainWindow.showMinimized)
        self.pushButton_4.clicked.connect(MainWindow.close)
        self.listWidget.currentRowChanged['int'].connect(self.stackedWidget.setCurrentIndex)
        self.clearButton.clicked.connect(self.textBrowser_result.clear)
        self.extrapushButton.clicked.connect(MainWindow.keyword)
        self.toolButton.clicked.connect(self.textBrowser_file.clear)
        self.toolButton.clicked.connect(MainWindow.read)
        self.wordButton.clicked.connect(MainWindow.wordcloud)
        self.historyButton.clicked.connect(MainWindow.history)
        self.dehistoryButton.clicked.connect(self.csvWidget.clear)
        self.dehistoryButton.clicked.connect(MainWindow.dehistory)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "关键词提取系统"))
        self.label_4.setText(_translate("MainWindow", "Designed by 谭佐捷，张展鹏"))
        self.pushButton_3.setText(_translate("MainWindow", "-"))
        self.pushButton_4.setText(_translate("MainWindow", "X"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">选择文件</span></p></body></html>"))
        self.extrapushButton.setText(_translate("MainWindow", "提取关键词"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:16pt;\">提取结果</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "关键词个数"))
        self.toolButton.setText(_translate("MainWindow", "..."))
        self.clearButton.setText(_translate("MainWindow", "清除结果"))
        self.pushButton_2.setText(_translate("MainWindow", "BUG反馈"))
        self.wordButton.setText(_translate("MainWindow", "生成词云"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">词云</span></p></body></html>"))
        self.historyButton.setText(_translate("MainWindow", "历史"))
        self.dehistoryButton.setText(_translate("MainWindow", "删除记录"))
        self.textBrowser_2.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">使用说明</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:14pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">1.点文件选择(三个点)选择要提取的文件。</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">2.输入要提取的关键词个数(一般是5)。</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">3.点击提取关键词按钮。</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt;\">4.如果觉得输出框内容太多，可以点击左侧的清除按钮。</span></p></body></html>"))
        self.listWidget.setSortingEnabled(False)
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "首页"))
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", "词云"))
        item = self.listWidget.item(2)
        item.setText(_translate("MainWindow", "历史"))
        item = self.listWidget.item(3)
        item.setText(_translate("MainWindow", "关于"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
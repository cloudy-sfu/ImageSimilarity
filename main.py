import sys
from threading import Event
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
from cv_pre_process import *
from datetime import datetime
import pickle
from siamese_cnn import Siamese


class MyWindow(QWidget):
    def __init__(self):
        super(QWidget, self).__init__()
        # Variables
        windows_user = os.environ['USERNAME']
        self.current_folder = {}
        normal_style = QFont("Calibri", 11)
        title_style = QFont("Calibri", 14, 75)
        comment_style = QFont("Calibri Light", 10)
        code_style = QFont("Consolas", 9)
        self.model_test_console = None
        self.model_test_pause = False
        self.model_single_test_console = None
        self.model_valid_console = None
        self.model_valid_pause = False
        self.model_round_test_console = None
        self.model_round_test_pause = False
        # Window Setting
        self.setFont(normal_style)
        self.setPalette(QPalette())
        self.resize(1000, 580)
        self.center()
        self.setWindowTitle(windows_user + "@Trademarks Similarity Detector")
        # Reporter Section
        self.label9 = QLabel("Reporter", self)
        self.label9.setFont(title_style)
        self.label9.resize(100, 30)
        self.label9.move(740, 10)
        # RS Line 1
        self.box1 = QTextBrowser(self)
        self.box1.setFont(code_style)
        self.box1.resize(240, 520)
        self.box1.move(740, 40)
        # Batch Test Section
        self.label1 = QLabel("Batch Test", self)
        self.label1.setFont(title_style)
        self.label1.resize(120, 30)
        self.label1.move(20, 10)
        # BTS Line 1
        self.label2 = QLabel("Trademarks of Group 1", self)
        self.label2.resize(170, 30)
        self.label2.move(20, 40)
        self.button1 = QPushButton("Select Folder", self)
        self.button1.resize(110, 30)
        self.button1.move(200, 40)
        self.button1.clicked.connect(lambda: self.select_folder('test_1'))
        self.label3 = QLabel("Trademarks of Group 2", self)
        self.label3.resize(170, 30)
        self.label3.move(320, 40)
        self.button2 = QPushButton("Select Folder", self)
        self.button2.resize(110, 30)
        self.button2.move(500, 40)
        self.button2.clicked.connect(lambda: self.select_folder('test_2'))
        # BTS Line 2
        self.label8 = QLabel("* The filenames of a pair of images to be determined should be the same.\n"
                             "  If broken down, delete 'desktop.ini' in the folder.", self)
        self.label8.setFont(comment_style)
        self.label8.resize(510, 50)
        self.label8.move(20, 70)
        # BTS Line 3
        self.label4 = QLabel("Model to Use", self)
        self.label4.resize(100, 30)
        self.label4.move(20, 120)
        self.button3 = QPushButton("Open", self)
        self.button3.resize(60, 30)
        self.button3.move(130, 120)
        self.button3.clicked.connect(lambda: self.select_file('model', "Model (*.h5)"))
        self.label5 = QLabel("Execute Model Test", self)
        self.label5.resize(150, 30)
        self.label5.move(200, 120)
        self.button4 = QPushButton("Start", self)
        self.button4.resize(50, 30)
        self.button4.move(350, 120)
        self.button4.clicked.connect(lambda: self.model_test('start'))
        self.button5 = QPushButton("Pause/Continue", self)
        self.button5.resize(130, 30)
        self.button5.move(470, 120)
        self.button5.clicked.connect(lambda: self.model_test('pause'))
        self.button6 = QPushButton("Stop", self)
        self.button6.resize(50, 30)
        self.button6.move(410, 120)
        self.button6.clicked.connect(lambda: self.model_test('stop'))
        self.label6 = QLabel("Loading Process", self)
        self.label6.resize(120, 30)
        self.label6.move(20, 150)
        self.process1 = QProgressBar(self)
        self.process1.resize(350, 20)
        self.process1.move(150, 155)
        self.label7 = QLabel("Status of Model - Not Started", self)
        self.label7.resize(230, 30)
        self.label7.move(500, 150)
        # Single Test Section
        self.label12 = QLabel("Single Test", self)
        self.label12.setFont(title_style)
        self.label12.resize(120, 30)
        self.label12.move(20, 180)
        # STS Line 2
        self.label10 = QLabel("Trademarks 1", self)
        self.label10.resize(170, 30)
        self.label10.move(20, 210)
        self.button7 = QPushButton("Select File", self)
        self.button7.resize(110, 30)
        self.button7.move(200, 210)
        self.button7.clicked.connect(lambda: self.select_file('test_single_1',
                                                              " ;;BMP (*.bmp);;PNG (*.png)"))
        self.label11 = QLabel("Trademarks 2", self)
        self.label11.resize(170, 30)
        self.label11.move(320, 210)
        self.button8 = QPushButton("Select File", self)
        self.button8.resize(110, 30)
        self.button8.move(500, 210)
        self.button8.clicked.connect(lambda: self.select_file('test_single_2',
                                                              "JPEG (*.jpg);;BMP (*.bmp);;PNG (*.png)"))
        # STS Line 3
        self.label13 = QLabel("Model to Use", self)
        self.label13.resize(100, 30)
        self.label13.move(20, 240)
        self.button9 = QPushButton("Open", self)
        self.button9.resize(60, 30)
        self.button9.move(130, 240)
        self.button9.clicked.connect(lambda: self.select_file('model', "Model (*.h5)"))
        self.label14 = QLabel("Execute Model Test", self)
        self.label14.resize(150, 30)
        self.label14.move(200, 240)
        self.button10 = QPushButton("Start", self)
        self.button10.resize(50, 30)
        self.button10.move(350, 240)
        self.button10.clicked.connect(self.model_single_test)
        self.label15 = QLabel("Status of Model - Not Started", self)
        self.label15.resize(230, 30)
        self.label15.move(500, 240)
        # Validation Section
        self.label16 = QLabel("Model Estimation", self)
        self.label16.setFont(title_style)
        self.label16.resize(170, 30)
        self.label16.move(20, 270)
        # VS Line 1
        self.label17 = QLabel("Trademarks of Group 1", self)
        self.label17.resize(170, 30)
        self.label17.move(20, 300)
        self.button11 = QPushButton("Select Folder", self)
        self.button11.resize(110, 30)
        self.button11.move(200, 300)
        self.button11.clicked.connect(lambda: self.select_folder('valid_1'))
        self.label18 = QLabel("Trademarks of Group 2", self)
        self.label18.resize(170, 30)
        self.label18.move(320, 300)
        self.button12 = QPushButton("Select Folder", self)
        self.button12.resize(110, 30)
        self.button12.move(500, 300)
        self.button12.clicked.connect(lambda: self.select_folder('valid_2'))

        # VS Line 2
        self.label19 = QLabel("* The filenames of a pair of images to be determined should be the same.\n"
                              "  If broken down, delete 'desktop.ini' in the folder.", self)
        self.label19.setFont(comment_style)
        self.label19.resize(510, 50)
        self.label19.move(20, 330)
        # VS Line 3
        self.label20 = QLabel("Pending File", self)
        self.label20.resize(100, 30)
        self.label20.move(20, 380)
        self.button13 = QPushButton("Open", self)
        self.button13.resize(60, 30)
        self.button13.move(130, 380)
        self.button13.clicked.connect(lambda: self.select_file('model', "Model (*.h5)"))
        self.label21 = QLabel("Execute Estimation", self)
        self.label21.resize(150, 30)
        self.label21.move(200, 380)
        self.button14 = QPushButton("Start", self)
        self.button14.resize(50, 30)
        self.button14.move(350, 380)
        self.button14.clicked.connect(lambda: self.model_valid('start'))
        self.button15 = QPushButton("Pause/Continue", self)
        self.button15.resize(130, 30)
        self.button15.move(470, 380)
        self.button15.clicked.connect(lambda: self.model_valid('pause'))
        self.button16 = QPushButton("Stop", self)
        self.button16.resize(50, 30)
        self.button16.move(410, 380)
        self.button16.clicked.connect(lambda: self.model_valid('stop'))
        self.label22 = QLabel("Loading Process", self)
        self.label22.resize(120, 30)
        self.label22.move(20, 410)
        self.process2 = QProgressBar(self)
        self.process2.resize(350, 20)
        self.process2.move(150, 415)
        self.label23 = QLabel("Status of Model - Not Started", self)
        self.label23.resize(230, 30)
        self.label23.move(500, 410)
        # Searching Similarity Section
        self.label24 = QLabel("Searching Similar", self)
        self.label24.setFont(title_style)
        self.label24.resize(170, 30)
        self.label24.move(20, 440)
        # SSS Line 1
        self.label25 = QLabel("New Trademark", self)
        self.label25.resize(170, 30)
        self.label25.move(20, 470)
        self.button17 = QPushButton("Select File", self)
        self.button17.resize(110, 30)
        self.button17.move(200, 470)
        self.button17.clicked.connect(lambda: self.select_file('test_round_1',
                                                               " ;;BMP (*.bmp);;PNG (*.png)"))
        self.label26 = QLabel("Gallery of Trademarks", self)
        self.label26.resize(170, 30)
        self.label26.move(320, 470)
        self.button18 = QPushButton("Select Folder", self)
        self.button18.resize(110, 30)
        self.button18.move(500, 470)
        self.button18.clicked.connect(lambda: self.select_folder('test_round_2'))
        # SSS Line 2
        self.label27 = QLabel("Model to Use", self)
        self.label27.resize(100, 30)
        self.label27.move(20, 500)
        self.button19 = QPushButton("Open", self)
        self.button19.resize(60, 30)
        self.button19.move(130, 500)
        self.button19.clicked.connect(lambda: self.select_file('model', "Model (*.h5)"))
        self.label28 = QLabel("Execute Model Test", self)
        self.label28.resize(150, 30)
        self.label28.move(200, 500)
        self.button20 = QPushButton("Start", self)
        self.button20.resize(50, 30)
        self.button20.move(350, 500)
        self.button20.clicked.connect(lambda: self.model_round_test('start'))
        self.button21 = QPushButton("Pause/Continue", self)
        self.button21.resize(130, 30)
        self.button21.move(470, 500)
        self.button21.clicked.connect(lambda: self.model_round_test('pause'))
        self.button22 = QPushButton("Stop", self)
        self.button22.resize(50, 30)
        self.button22.move(410, 500)
        self.button22.clicked.connect(lambda: self.model_round_test('stop'))
        self.label29 = QLabel("Loading Process", self)
        self.label29.resize(120, 30)
        self.label29.move(20, 530)
        self.process3 = QProgressBar(self)
        self.process3.resize(350, 20)
        self.process3.move(150, 535)
        self.label30 = QLabel("Status of Model - Not Started", self)
        self.label30.resize(230, 30)
        self.label30.move(500, 530)

    def select_folder(self, name):
        home_path = os.environ['HOMEPATH'] + r"\Desktop"
        self.current_folder[name] = \
            QFileDialog.getExistingDirectory(directory=home_path)

    def select_file(self, name, type_):
        home_path = os.environ['HOMEPATH'] + r"\Desktop"
        self.current_folder[name] = \
            QFileDialog.getOpenFileName(directory=home_path, filter=type_)[0]

    def model_test(self, method):
        if method == "start":
            self.label7.setText("Status of Model - Loading Data")
            self.model_test_pause = False
            self.model_test_console = ModelAction(obj1=self.current_folder['test_1'],
                                                  obj2=self.current_folder['test_2'],
                                                  model_fp=self.current_folder['model'],
                                                  method="test_batch")
            self.model_test_console.start()
            self.model_test_console.signal1.connect(self.data_progress_test)
            self.model_test_console.signal2.connect(self.test_reporter)
            self.model_test_console.signal3.connect(self.model_status_batch_test)
        elif method == 'pause' and self.model_test_console and (not self.model_test_pause):
            self.model_test_console.pause()
            self.model_test_pause = True
        elif method == 'pause' and self.model_test_console and self.model_test_pause:
            self.model_test_console.resume()
            self.model_test_pause = False
        elif method == 'stop' and self.model_test_console:
            self.model_test_console.pause()
            self.model_test_console = None
            self.model_test_pause = False
            self.process1.setValue(0)
            self.label7.setText("Status of Model - Abandon")

    def model_single_test(self):
        self.label15.setText("Status of Model - Loading Data")
        self.model_single_test_console = ModelAction(obj1=self.current_folder['test_single_1'],
                                                     obj2=self.current_folder['test_single_2'],
                                                     model_fp=self.current_folder['model'],
                                                     method="single_test")
        self.model_single_test_console.start()
        self.model_single_test_console.signal2.connect(self.test_reporter)
        self.model_single_test_console.signal3.connect(self.model_status_single_test)

    def model_valid(self, method):
        if method == "start":
            self.label23.setText("Status of Model - Loading Data")
            self.model_valid_pause = False
            self.model_valid_console = ModelAction(obj1=self.current_folder['valid_1'],
                                                   obj2=self.current_folder['valid_2'],
                                                   model_fp=self.current_folder['model'],
                                                   method="valid")
            self.model_valid_console.start()
            self.model_valid_console.signal1.connect(self.data_progress_valid)
            self.model_valid_console.signal2.connect(self.test_reporter)
            self.model_valid_console.signal3.connect(self.model_status_valid)
        elif method == 'pause' and self.model_valid_console and (not self.model_valid_pause):
            self.model_valid_console.pause()
            self.model_valid_pause = True
        elif method == 'pause' and self.model_valid_console and self.model_valid_pause:
            self.model_valid_console.resume()
            self.model_valid_pause = False
        elif method == 'stop' and self.model_valid_console:
            self.model_valid_console.pause()
            self.model_valid_console = None
            self.model_valid_pause = False
            self.process2.setValue(0)
            self.label23.setText("Status of Model - Abandon")

    def model_round_test(self, method):
        if method == "start":
            self.label30.setText("Status of Model - Loading Data")
            self.model_round_test_pause = False
            self.model_round_test_console = ModelAction(obj1=self.current_folder['test_round_1'],
                                                        obj2=self.current_folder['test_round_2'],
                                                        model_fp=self.current_folder['model'],
                                                        method="round_test")
            self.model_round_test_console.start()
            self.model_round_test_console.signal1.connect(self.data_progress_round_test)
            self.model_round_test_console.signal2.connect(self.test_reporter)
            self.model_round_test_console.signal3.connect(self.model_status_round_test)
        elif method == 'pause' and self.model_round_test_console and (not self.model_round_test_pause):
            self.model_round_test_console.pause()
            self.model_round_test_pause = True
        elif method == 'pause' and self.model_round_test_consoletest_console and self.model_round_test_pause:
            self.model_round_test_consoletest_console.resume()
            self.model_round_test_pause = False
        elif method == 'stop' and self.model_round_test_console:
            self.model_round_test_console.pause()
            self.model_round_test_console = None
            self.model_round_test_pause = False
            self.process3.setValue(0)
            self.label30.setText("Status of Model - Abandon")

    def data_progress_test(self, msg):
        self.process1.setValue(msg)

    def data_progress_valid(self, msg):
        self.process2.setValue(msg)

    def data_progress_round_test(self, msg):
        self.process3.setValue(msg)

    def model_status_batch_test(self, msg):
        self.label7.setText(msg)

    def model_status_single_test(self, msg):
        self.label15.setText(msg)

    def model_status_valid(self, msg):
        self.label23.setText(msg)

    def model_status_round_test(self, msg):
        self.label30.setText(msg)

    def test_reporter(self, msg):
        self.box1.append(msg)

    def center(self):
        fg = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        fg.moveCenter(cp)
        self.move(fg.topLeft())


class ModelAction(QThread):
    signal1 = pyqtSignal(float)
    signal2 = pyqtSignal(str)
    signal3 = pyqtSignal(str)

    def __init__(self, obj1=None, obj2=None, model_fp=None, method=None):
        super().__init__()
        self.flag = Event()
        self.flag.set()
        self.obj1 = obj1
        self.obj2 = obj2
        self.model_fp = model_fp
        self.method = method

    def run(self):
        if self.method == "batch_test":
            self._batch_test()
        elif self.method == "single_test":
            self._batch_test()
        elif self.method == "valid":
            self._valid()
        elif self.method == "round_test":
            self._round_test()

    def _round_test(self):
        test_pkl_path = os.environ['TEMP'] + datetime.now().strftime(r"\valid_%Y_%m_%d_%H_%M_%S.pkl")
        process = load_data_round_test(self.obj1, self.obj2, test_pkl_path)
        for pct in process:
            self.flag.wait()
            self.signal1.emit(pct)
        with open(test_pkl_path, 'rb') as fp:
            saver = pickle.load(fp)
            x_test, n = saver['X'], saver['n']
        self.signal3.emit("Status of Model - Running...")
        siamese = Siamese()
        self.flag.wait()
        siamese.load_model(self.model_fp)
        self.flag.wait()
        y_test = siamese.test(x_test)
        for fn, label in zip(os.listdir(self.obj2), y_test):
            self.flag.wait()
            if label == 1:
                state = str(fn.split(".")[0]) + " Similar to new trademark."
                self.signal2.emit(state)
        self.signal3.emit("Status of Model - Done")

    def _valid(self):
        valid_pkl_path = os.environ['TEMP'] + datetime.now().strftime(r"\valid_%Y_%m_%d_%H_%M_%S.pkl")
        process = load_data_valid(self.obj1, self.obj2, valid_pkl_path)
        for pct in process:
            self.flag.wait()
            self.signal1.emit(pct)
        with open(valid_pkl_path, 'rb') as fp:
            saver = pickle.load(fp)
            x_valid, y_valid, n = saver['X'], saver['Y'], saver['n']
        self.signal3.emit("Status of Model - Running...")
        siamese = Siamese()
        self.flag.wait()
        siamese.load_model(self.model_fp)
        self.flag.wait()
        c_mat = siamese.valid(x_valid, y_valid)
        valid_reporter = "TN=%d, FN=%d, FP=%d, TP=%d" % (c_mat[0, 0], c_mat[0, 1], c_mat[1, 0], c_mat[1, 1])
        self.signal2.emit(valid_reporter)
        self.signal3.emit("Status of Mode - Done")

    def _single_test(self):
        x_test = load_data_single_test(self.obj1, self.obj2)
        self.signal3.emit("Status of Model - Running...")
        siamese = Siamese()
        siamese.load_model(self.model_fp)
        y_test = siamese.test(x_test)
        if y_test == 0:
            state = " Different"
        elif y_test == 1:
            state = " Same"
        else:
            state = " Can't Judge"
        self.signal2.emit(state)
        self.signal3.emit("Status of Model - Done")

    def _batch_test(self):
        test_pkl_path = os.environ['TEMP'] + datetime.now().strftime(r"\test_%Y_%m_%d_%H_%M_%S.pkl")
        process = load_data_batch_test(self.obj1, self.obj2, test_pkl_path)
        for pct in process:
            self.flag.wait()
            self.signal1.emit(pct)
        with open(test_pkl_path, 'rb') as fp:
            saver = pickle.load(fp)
            x_test, n = saver['X'], saver['n']
        self.signal3.emit("Status of Model - Running...")
        siamese = Siamese()
        self.flag.wait()
        siamese.load_model(self.model_fp)
        self.flag.wait()
        y_test = siamese.test(x_test)
        for fn, label in zip(os.listdir(self.obj1), y_test):
            self.flag.wait()
            if label == 0:
                state = " Different"
            elif label == 1:
                state = " Same"
            else:
                state = " Can't Judge"
            state = str(fn.split(".")[0]) + state
            self.signal2.emit(state)
        self.signal3.emit("Status of Model - Done")

    def pause(self):
        self.flag.clear()

    def resume(self):
        self.flag.set()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myw = MyWindow()
    myw.show()
    sys.exit(app.exec_())

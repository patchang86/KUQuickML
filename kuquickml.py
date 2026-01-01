print("실행 중입니다... PC 환경에 따라 최대 1분 가량 소요될 수 있습니다. 프로그램이 실행 중인 동안 본 콘솔 창을 닫지 마십시오.")
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QColorDialog, QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget, QPushButton, QDialog, QLabel, QComboBox, QHBoxLayout, QFileDialog, QAction, QMenu,
                             QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QCheckBox, QSpinBox, QFrame,
                             QButtonGroup, QRadioButton, QListWidgetItem, QGridLayout, QDialogButtonBox, QListWidget,
                             QInputDialog, QLineEdit, QDoubleSpinBox,QToolTip,QSplitter)
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QDoubleValidator, QFont
from PyQt5.QtCore import Qt, QCoreApplication
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from dialogs.data_scaling import DataScaler
from dialogs.column_role_dialog import ColumnRoleDialog
from dialogs.label_mapping_dialog import LabelMappingDialog
from dialogs.sample_selection_dialog import SampleSelectionDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
import warnings
import mplcursors
from sklearn.svm import SVR
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import seaborn as sns
import joblib
def resource_path(relative_path):
    """ PyInstaller로 패키징할 때 파일 경로를 반환하는 함수 """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_dir = os.path.abspath(".")
        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'MaxAbsScaler': MaxAbsScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer()
        }

        self.latest_model = None
        self.model_features = []
        self.scaler = None
        self.dimensionality_reduction_model = None
        self.models = {}
        self.model_reducers = {}
        self.initUI()
    def initUI(self):
        self.setGeometry(300, 300, 1000, 700)
        self.setWindowTitle('KUQuickML')
        self.setWindowIcon(QIcon('icon.png'))

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.mainTab = QWidget()
        self.scaledDataTab = QWidget()
        self.predictionTab = QWidget()
        self.previousModelPredictionTab = QWidget()  # 새 탭 추가

        self.tabs.addTab(self.mainTab, "Main")
        self.tabs.addTab(self.scaledDataTab, "Scaled Data")

        self.tabs.addTab(self.predictionTab, "Prediction")

        self.setupMainTab()
        self.setupScaledDataTab()
        self.setupPredictionTab()
        self.setupPreviousModelPredictionTab()  # 새 탭 설정 함수

        self.show()

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        fileMenu = menubar.addMenu('1. File')
        loadAction = QAction('Load Data', self)
        loadAction.triggered.connect(self.loadCsv)
        fileMenu.addAction(loadAction)

        saveModelAction = QAction('Save Model', self)
        saveModelAction.triggered.connect(self.saveModelDialog)
        fileMenu.addAction(saveModelAction)


        dataScalingMenu = QMenu('2. Data Scaling', self)
        menubar.addMenu(dataScalingMenu)
        self.dataScaler = DataScaler(self.csvViewer, self)

        datasplitMenu = QMenu('3. Test set Split',self)
        datasplitAction = QAction('Random selection', self)
        datasplitAction.triggered.connect(self.addDataSplitTab)  # 이벤트 연결
        datasplitMenu.addAction(datasplitAction)
        menubar.addMenu(datasplitMenu)

        algorithmMenu = QMenu('4. Algorithm', self)
        knnAction = QAction('KNN', self)
        knnAction.triggered.connect(self.addKnnTab)
        algorithmMenu.addAction(knnAction)
        mlpAction = QAction('Multi-Layer Perceptron', self)
        mlpAction.triggered.connect(self.addMLPTab)  # 이벤트 연결
        algorithmMenu.addAction(mlpAction)
        rfAction = QAction('Random Forest', self)
        rfAction.triggered.connect(self.addRFTab)
        algorithmMenu.addAction(rfAction)
        svmAction = QAction('Support Vector Machine',self)
        svmAction.triggered.connect(self.addSVMTab)
        algorithmMenu.addAction(svmAction)
        menubar.addMenu(algorithmMenu)

        predictionMenu = menubar.addMenu('5. Prediction')
        predictionMenu.triggered.connect(lambda: self.tabs.setCurrentWidget(self.predictionTab))


        loadPreviousModelAction = QAction('Load Previous Model', self)
        loadPreviousModelAction.triggered.connect(self.loadPreviousModel)
        predictionMenu.addAction(loadPreviousModelAction)

        for name, scaler in self.scalers.items():
            scalerAction = QAction(name, self)
            scalerAction.triggered.connect(lambda ch, s=scaler, n=name: self.applyScaler(s, n))
            dataScalingMenu.addAction(scalerAction)

        exitAction = QAction('&Exit', self)
        exitAction.triggered.connect(self.exitApp)
        fileMenu.addAction(exitAction)

    def useCurrentModel(self):
        self.tabs.setCurrentWidget(self.predictionTab)

    def loadPreviousModel(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Saved Model", "",
            "Joblib Files (*.joblib);;All Files (*)", options=options
        )
        if not filename:
            return

        try:
            loaded_bundle = joblib.load(filename)

            if isinstance(loaded_bundle, dict):
                model = loaded_bundle.get("model", None)
                scaler = loaded_bundle.get("scaler", None)
                reducer = loaded_bundle.get("reducer", None)
                feature_names = loaded_bundle.get("feature_names", None)
                label_mapping = loaded_bundle.get("label_mapping", None)
            else:
                model = loaded_bundle
                scaler = None
                reducer = None
                feature_names = None
                label_mapping = None

            # loadUnknownSample에서 사용하기 위해 저장
            self.loaded_bundle = loaded_bundle

            model_name = os.path.basename(filename).replace(".joblib", "")
            self.models[model_name] = model
            if reducer:
                self.model_reducers[model_name] = reducer
            if scaler:
                self.model_scalers = getattr(self, "model_scalers", {})
                self.model_scalers[model_name] = scaler

            self.feature_names = feature_names
            self.label_mapping = label_mapping

            scaler_name = type(scaler).__name__ if scaler else "None"
            reducer_name = type(reducer).__name__ if reducer else "None"

            self.prediction_status.setText(
                f"Model loaded: {model_name}\n"
                f"Scaler used: {scaler_name}\n"
                f"Reducer used: {reducer_name}"
            )

            QMessageBox.information(
                self, "Model Loaded",
                f"Model '{model_name}' loaded successfully.\n"
                f"Scaler used: {scaler_name}\nReducer used: {reducer_name}"
            )

            self.tabs.setCurrentWidget(self.predictionTab)

        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load model:\n{e}")

    def setupPreviousModelPredictionTab(self):
        # 이곳에서 탭의 레이아웃 및 다른 UI 구성 요소를 설정합니다.
        layout = QVBoxLayout()
        someLabel = QLabel("This is the Previous Model Prediction Tab")
        layout.addWidget(someLabel)
        self.previousModelPredictionTab.setLayout(layout)

    def saveModel(self, model_name, filename):
        model = self.models.get(model_name)
        reducer = self.model_reducers.get(model_name)
        # ✅ fit된 scaler를 우선 사용
        scaler = getattr(self, "fitted_scaler", getattr(self, "scaler", None))
        feature_names = getattr(self, "feature_names_used", None)
        label_mapping = getattr(self.csvViewer, "label_mapping", None)

        bundle = {
            "model": model,
            "scaler": scaler,
            "reducer": reducer,
            "feature_names": feature_names,
            "label_mapping": label_mapping,
        }

        try:
            joblib.dump(bundle, filename)
            QMessageBox.information(
                self,
                "Model Saved",
                f"Model '{model_name}' saved successfully.\n"
                f"Scaler: {type(scaler).__name__ if scaler else 'None'}\n"
                f"Reducer: {type(reducer).__name__ if reducer else 'None'}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save model:\n{e}")

    def saveModelDialog(self):
        model_choice, ok = QInputDialog.getItem(self, "Select Model to Save",
                                                "Choose a model to save:",
                                                list(self.models.keys()), 0, False)
        if ok and model_choice:
            scaler_choices = list(self.scalers.keys()) + ['None']  # Include 'None' for models trained without scaling
            scaler_choice, ok = QInputDialog.getItem(self, "Select Scaler",
                                                     "Choose the scaling method used:",
                                                     scaler_choices, 0, False)
            if ok and scaler_choice:
                options = QFileDialog.Options()
                suggested_name = f"{model_choice}_{scaler_choice}"
                filename, _ = QFileDialog.getSaveFileName(self, "Save Model",
                                                          suggested_name,
                                                          "Joblib Files (*.joblib);;All Files (*)",
                                                          options=options)
                if filename:
                    self.saveModel(model_choice, filename)

    def applyScaler(self, scaler, scaler_name):

        fitted_scaler = self.dataScaler.apply_scaling(scaler)
        self.scaler = fitted_scaler
        self.scaler_name = scaler_name

        print(f"[Scaler Applied] {scaler_name} has been set as current scaler.")

        if hasattr(self, "scalerStatusLabel"):
            self.scalerStatusLabel.setText(f"Current Scaling Method: {scaler_name}")

    def addKnnTab(self):
        self.knnTab = QWidget()
        self.tabs.addTab(self.knnTab, "KNN")
        self.tabs.setCurrentWidget(self.knnTab)
        self.setupKnnTab()

    def addMLPTab(self):
        self.MLPTab = QWidget()
        self.tabs.addTab(self.MLPTab, "MLP")
        self.tabs.setCurrentWidget(self.MLPTab)
        self.setupMLPTab()

    def addRFTab(self):
        self.RFTab = QWidget()
        self.tabs.addTab(self.RFTab, "RF")
        self.tabs.setCurrentWidget(self.RFTab)
        self.setupRFTab()

    def addSVMTab(self):
        self.SVMTab = QWidget()
        self.tabs.addTab(self.SVMTab, "SVM")
        self.tabs.setCurrentWidget(self.SVMTab)
        self.setupSVMTab()

    def addDataSplitTab(self):
        self.dataSplitTab = QWidget()
        self.tabs.addTab(self.dataSplitTab, "Data Split")
        self.tabs.setCurrentWidget(self.dataSplitTab)
        self.setupDataSplitTab()

    def setupSVMTab(self):
        layout = QVBoxLayout()
        desc_style = "color: #555; font-size: 10pt; margin-bottom: 4px;"

        # --- 상단: SVM 개요 설명 ---
        svm_overview = QLabel(
            "<h3>⚙️ Support Vector Machine (SVM)</h3>"
            "<p>SVM은 주어진 데이터를 분류하거나 회귀할 때, 클래스 간의 경계를 최적으로 구분하는 초평면을 찾는 알고리즘입니다.<br>"
            "커널 함수를 사용해 비선형적인 데이터도 고차원 공간에서 분리할 수 있습니다.</p>"
        )
        svm_overview.setWordWrap(True)
        layout.addWidget(svm_overview)

        # --- 파라미터 박스 ---
        groupBox = QFrame()
        groupBox.setFrameShape(QFrame.Box)
        groupBox.setFrameShadow(QFrame.Sunken)
        groupBoxLayout = QVBoxLayout(groupBox)

        # SVM Type
        frame_type = QFrame()
        frame_type.setFrameShape(QFrame.Box)
        frame_type.setFrameShadow(QFrame.Sunken)
        frame_type_layout = QVBoxLayout(frame_type)
        label = QLabel("Select SVM Type:")
        self.svm_type = QComboBox()
        self.svm_type.addItems(["One-vs-Rest SVM", "One-vs-One SVM"])
        desc = QLabel("SVM의 분류 전략을 선택합니다.<br>"
                      "<b>One-vs-Rest</b>: 한 클래스를 나머지 전부와 비교 (빠름)<br>"
                      "<b>One-vs-One</b>: 클래스 간 모든 조합을 학습 (정확도↑)")
        desc.setStyleSheet(desc_style)
        frame_type_layout.addWidget(label)
        frame_type_layout.addWidget(self.svm_type)
        frame_type_layout.addWidget(desc)
        groupBoxLayout.addWidget(frame_type)

        # Kernel Type
        frame_kernel = QFrame()
        frame_kernel.setFrameShape(QFrame.Box)
        frame_kernel.setFrameShadow(QFrame.Sunken)
        frame_kernel_layout = QVBoxLayout(frame_kernel)
        label = QLabel("Select Kernel Type:")
        self.kernel_type = QComboBox()
        self.kernel_type.addItems(["linear", "poly", "rbf", "sigmoid"])
        desc = QLabel("커널은 입력 데이터를 고차원 공간으로 변환하는 함수입니다.<br>"
                      "<b>linear</b>: 선형 경계, 빠름<br>"
                      "<b>poly</b>: 다항식 커널<br>"
                      "<b>rbf</b>: 가우시안 기반, 비선형에 강함<br>"
                      "<b>sigmoid</b>: 신경망 유사 특성")
        desc.setStyleSheet(desc_style)
        frame_kernel_layout.addWidget(label)
        frame_kernel_layout.addWidget(self.kernel_type)
        frame_kernel_layout.addWidget(desc)
        groupBoxLayout.addWidget(frame_kernel)

        # C Value
        frame_c = QFrame()
        frame_c.setFrameShape(QFrame.Box)
        frame_c.setFrameShadow(QFrame.Sunken)
        frame_c_layout = QVBoxLayout(frame_c)
        label = QLabel("C Value:")
        self.c_value = QDoubleSpinBox()
        self.c_value.setRange(0.01, 100.0)
        self.c_value.setValue(1.0)
        self.c_value.setSingleStep(0.01)
        desc = QLabel("C 값은 오류 허용 정도를 조절하는 규제(regularization) 강도입니다.<br>"
                      "작으면 일반화 ↑ (느리지만 안정), 크면 훈련 정확도 ↑ (과적합 위험).")
        desc.setStyleSheet(desc_style)
        frame_c_layout.addWidget(label)
        frame_c_layout.addWidget(self.c_value)
        frame_c_layout.addWidget(desc)
        groupBoxLayout.addWidget(frame_c)

        # Random State
        frame_random = QFrame()
        frame_random.setFrameShape(QFrame.Box)
        frame_random.setFrameShadow(QFrame.Sunken)
        frame_random_layout = QVBoxLayout(frame_random)
        label = QLabel("Random State:")
        self.random_state_input = QSpinBox()
        self.random_state_input.setRange(0, 999999)
        self.random_state_input.setValue(42)
        desc = QLabel("무작위 초기화를 제어하는 시드 값입니다.<br>같은 결과를 재현하려면 동일한 값을 유지하세요.")
        desc.setStyleSheet(desc_style)
        frame_random_layout.addWidget(label)
        frame_random_layout.addWidget(self.random_state_input)
        frame_random_layout.addWidget(desc)
        groupBoxLayout.addWidget(frame_random)

        # Dimensionality reduction
        frame_reducer = QFrame()
        frame_reducer.setFrameShape(QFrame.Box)
        frame_reducer.setFrameShadow(QFrame.Sunken)
        frame_reducer_layout = QVBoxLayout(frame_reducer)
        label = QLabel("Select Dimensionality Reduction Method:")
        self.pcaCheckBox = QCheckBox("PCA")
        self.ldaCheckBox = QCheckBox("LDA")
        self.ncaCheckBox = QCheckBox("NCA")
        self.noneCheckBox = QCheckBox("None")

        self.pcaCheckBox.setChecked(True)  # PCA를 기본값으로 설정
        self.dimensionalityGroup = QButtonGroup()
        for checkbox in [self.pcaCheckBox, self.ldaCheckBox, self.ncaCheckBox, self.noneCheckBox]:
            frame_reducer_layout.addWidget(checkbox)
            self.dimensionalityGroup.addButton(checkbox)
        self.dimensionalityGroup.setExclusive(True)
        desc = QLabel("차원 축소는 데이터를 저차원 공간으로 투영하여 계산 효율과 시각화를 돕습니다.<br>"
                      "<b>PCA</b>: 주성분 분석 (일반적)<br>"
                      "<b>LDA</b>: 클래스 간 분리 최적화<br>"
                      "<b>NCA</b>: 거리 기반 분류에 적합<br>"
                      "<b>None</b>: 차원 축소 미적용")
        desc.setStyleSheet(desc_style)
        frame_reducer_layout.addWidget(desc)
        groupBoxLayout.addWidget(frame_reducer)

        groupBoxLayout.setContentsMargins(5, 5, 5, 5)
        groupBoxLayout.setSpacing(15)
        layout.addWidget(groupBox)

        # --- 버튼 ---
        buttons_layout = QHBoxLayout()
        self.createSVMModelButton = QPushButton("Create SVM Classification Model")
        self.createSVMModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createSVMModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createSVMModelButton.clicked.connect(self.createSVMModel)
        buttons_layout.addWidget(self.createSVMModelButton)

        self.createSVMRegressionModelButton = QPushButton("Create SVM Regression Model")
        self.createSVMRegressionModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createSVMRegressionModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createSVMRegressionModelButton.clicked.connect(self.createSVMRegressionModel)
        buttons_layout.addWidget(self.createSVMRegressionModelButton)
        layout.addLayout(buttons_layout)

        # --- 오른쪽: 폰트 설정 ---
        right_side_layout = QVBoxLayout()
        right_side_layout.setAlignment(Qt.AlignTop)
        right_side_layout.setSizeConstraint(QVBoxLayout.SetFixedSize)

        font_box = QFrame()
        font_box.setFrameShape(QFrame.Box)
        font_box.setFrameShadow(QFrame.Sunken)
        font_layout = QVBoxLayout(font_box)

        font_label = QLabel("Font settings:")
        font_type_label = QLabel("Font type:")
        self.fontTypeComboBox = QComboBox()
        self.fontTypeComboBox.addItems(["Arial", "Calibri", "Times New Roman", "Verdana"])

        font_size_label = QLabel("Font size:")
        self.fontSizeInput = QSpinBox()
        self.fontSizeInput.setRange(6, 32)
        self.fontSizeInput.setValue(12)

        font_layout.addWidget(font_label)
        font_layout.addWidget(font_type_label)
        font_layout.addWidget(self.fontTypeComboBox)
        font_layout.addWidget(font_size_label)
        font_layout.addWidget(self.fontSizeInput)

        right_side_layout.addWidget(font_box)

        # --- 전체 배치 ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(right_side_layout)

        self.SVMTab.setLayout(main_layout)
        self.tabs.setCurrentWidget(self.SVMTab)

    def createSVMModel(self):
        loading = QProgressDialog("모델 생성 중입니다...\n\n"
                                  "컴퓨터 사양에 따라 몇 분 정도 소요될 수 있습니다.",
                                  None, 0, 0, self)
        loading.setWindowTitle("SVM 모델 생성 중")
        loading.setWindowModality(Qt.ApplicationModal)
        loading.setMinimumWidth(420)
        loading.setAutoClose(False)
        loading.setAutoReset(False)
        loading.show()
        QApplication.processEvents()


        if not self.checkDataSplit():
            return

        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()



        self.model_features = X_train.columns.tolist()
        self.model_features.remove('Sample')

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        # 스케일 적용
        # Apply the scaler
        if self.scaler is not None:
            X_train_scaled = self.scaler.transform(X_train_numeric)
            X_test_scaled = self.scaler.transform(X_test_numeric)
            self.fitted_scaler = self.scaler

            if hasattr(X_train_numeric, "columns"):
                X_train_numeric = pd.DataFrame(
                    X_train_scaled,
                    columns=X_train_numeric.columns,
                    index=X_train_numeric.index
                )
                X_test_numeric = pd.DataFrame(
                    X_test_scaled,
                    columns=X_test_numeric.columns,
                    index=X_test_numeric.index
                )
            else:
                X_train_numeric = X_train_scaled
                X_test_numeric = X_test_scaled

        # 선택된 차원 축소 방법 적용
        selected_method = self.getSelectedDimReductionMethod()
        if selected_method:
            method_name, reducer = selected_method
            reducer.fit(X_train_numeric, y_train)  # 학습 데이터에 대해 fit
            X_train_reduced = reducer.transform(X_train_numeric)
            X_test_reduced = reducer.transform(X_test_numeric)


        kernel = self.kernel_type.currentText()
        c_value = self.c_value.value()

        if self.svm_type.currentText() == "One-vs-One SVM":
            svc_model = OneVsOneClassifier(SVC(kernel=kernel, C=c_value))
        else:
            svc_model = OneVsRestClassifier(SVC(kernel=kernel, C=c_value))

        svc_model.fit(X_train_reduced, y_train)
        y_pred_train = svc_model.predict(X_train_reduced)
        y_pred_test = svc_model.predict(X_test_reduced)
        cm = confusion_matrix(y_test, y_pred_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
            precision = np.nan_to_num(precision)  # Convert NaNs to zero

        # Automatically generate index and columns based on cm shape
        labels = [f"Class {i}" for i in range(cm.shape[0])]
        cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in labels],
                             columns=[f"Predicted {label}" for label in labels])
        cm_df['Prediction Accuracy (%)'] = precision
        self.showConfusionMatrix(cm_df)
        overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)

        self.plotScatterWithDecisionBoundary(X_train_reduced, y_train, X_test_reduced, y_pred_test, svc_model,
                                             f"SVM Scatter Plot with Decision Boundary\nTest accuracy = {overall_accuracy:.3f}")
        if kernel == 'linear' and not (
                self.pcaCheckBox.isChecked() or self.ldaCheckBox.isChecked() or self.ncaCheckBox.isChecked()):
            self.showImportantCoefficients(X_train_numeric, svc_model)

        self.plotObservedVsPredicted(y_train, y_pred_train, y_test, y_pred_test,
                                     "SVC Observed vs Predicted")

        self.models['SV Classification'] = svc_model
        if reducer:
            self.model_reducers['SV Classification'] = reducer

        QTimer.singleShot(300, loading.close)

    def createSVMRegressionModel(self):
        if not self.checkDataSplit():
            return

        # 데이터 로드
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()


        if hasattr(self.csvViewer, "label_column"):
            label_col = self.csvViewer.label_column
            if label_col in X_train.columns:
                X_train = X_train.drop(columns=[label_col], errors="ignore")
                X_test = X_test.drop(columns=[label_col], errors="ignore")

        # 문자열형 컬럼 제거 (예: name)
        X_train_numeric = X_train.select_dtypes(include=["number"])
        X_test_numeric = X_test.select_dtypes(include=["number"])
        self.model_features = X_train.columns.tolist()
        self.model_features.remove('Sample')

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        # 스케일 적용
        # Apply the scaler
        if self.scaler is not None:
            X_train_scaled = self.scaler.transform(X_train_numeric)
            X_test_scaled = self.scaler.transform(X_test_numeric)
            self.fitted_scaler = self.scaler

            if hasattr(X_train_numeric, "columns"):
                X_train_numeric = pd.DataFrame(
                    X_train_scaled,
                    columns=X_train_numeric.columns,
                    index=X_train_numeric.index
                )
                X_test_numeric = pd.DataFrame(
                    X_test_scaled,
                    columns=X_test_numeric.columns,
                    index=X_test_numeric.index
                )
            else:
                X_train_numeric = X_train_scaled
                X_test_numeric = X_test_scaled
        selected_method = self.getSelectedDimReductionMethod()
        if selected_method:
            method_name, reducer = selected_method
            reducer.fit(X_train_numeric, y_train)
            X_train_reduced = reducer.transform(X_train_numeric)
            X_test_reduced = reducer.transform(X_test_numeric)

        kernel = self.kernel_type.currentText()
        c_value = self.c_value.value()

        # SVR 모델 생성 및 학습
        svr_model = SVR(kernel=kernel, C=c_value)
        svr_model.fit(X_train_reduced, y_train)  # 축소된 데이터 사용

        # 예측 수행
        y_pred_train = svr_model.predict(X_train_reduced)
        y_pred_test = svr_model.predict(X_test_reduced)

        # 성능 지표 계산
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        # 결과 시각화
      #  self.plotScatterWithDecisionBoundary(
       #     X_train_reduced, y_train, X_test_reduced, y_pred_test, svr_model,
       #     "Training and Test Data with Decision Boundary"
      #  )

        # 선형 커널이고 차원 축소가 없을 때만 중요 피처 표시
        if kernel == 'linear' and not (
                self.pcaCheckBox.isChecked() or self.ldaCheckBox.isChecked() or self.ncaCheckBox.isChecked()):
            self.showImportantCoefficients(X_train_numeric, svr_model)

        # 예측 값과 실제 값 비교
        self.plotObservedVsPredicted(
            y_train, y_pred_train, y_test, y_pred_test, "SVR Observed vs Predicted"
        )

        # 모델 저장
        self.models['SV Regression'] = svr_model
        if reducer:
            self.model_reducers['SV Regression'] = reducer

    def plotScatterWithDecisionBoundary(self, X_train_reduced, y_train, X_test_reduced, y_pred_test, model, title):
        plt.figure(figsize=(10, 8))

        # 결정 경계 그리기 위한 메쉬 그리드 생성
        x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
        y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        # 예측 수행
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 결정 경계 시각화
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

        # 레이블별 색상 설정
        unique_labels = np.unique(np.concatenate((y_train, y_pred_test)))
        label_colors = {
            label: "#{:02x}{:02x}{:02x}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for label in unique_labels
        }

        # 훈련 및 테스트 데이터 시각화
        scatter_train_handles = []
        scatter_test_handles = []

        for label in unique_labels:
            train_indices = np.where(y_train == label)[0]
            test_indices = np.where(y_pred_test == label)[0]

            if len(train_indices) > 0:
                handle = plt.scatter(
                    X_train_reduced[train_indices, 0], X_train_reduced[train_indices, 1],
                    c=label_colors[label], marker='o', alpha=0.5,
                    label=f'Train Class {label}'
                )
                scatter_train_handles.append(handle)

            if len(test_indices) > 0:
                handle = plt.scatter(
                    X_test_reduced[test_indices, 0], X_test_reduced[test_indices, 1],
                    c=label_colors[label], marker='x', alpha=0.5,
                    label=f'Test Class {label}'
                )
                scatter_test_handles.append(handle)


        legend_elements = scatter_train_handles + scatter_test_handles

        # 범례 설정 및 드래그 가능하게 설정
        legend = plt.legend(handles=legend_elements, loc='best', title="Classes", frameon=True)
        legend.set_draggable(True)

        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(title)
        plt.show()

    def showImportantCoefficients(self, X_train_numeric, model):
        coef = model.estimators_[0].coef_ if hasattr(model, 'estimators_') else model.coef_
        importance = pd.DataFrame(coef.T, index=X_train_numeric.columns, columns=["Importance"])
        importance["Absolute Importance"] = importance["Importance"].abs()
        importance = importance.sort_values(by="Absolute Importance", ascending=False)

        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Importances")
        dialog.setGeometry(100, 100, 600, 400)

        dialog_layout = QVBoxLayout(dialog)

        table = QTableWidget(dialog)
        table.setRowCount(len(importance))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Feature", "Importance"])

        for i, (feature, importance_value) in enumerate(importance["Importance"].items()):
            table.setItem(i, 0, QTableWidgetItem(str(feature)))
            table.setItem(i, 1, QTableWidgetItem(f"{importance_value:.4f}"))

        table.resizeColumnsToContents()
        dialog_layout.addWidget(table)

        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()

    def setupMainTab(self):
        self.mainLayout = QVBoxLayout()

        # 가이드 위젯 생성
        self.guideWidget = QWidget()
        guideLayout = QVBoxLayout(self.guideWidget)

        guideTitle = QLabel("Welcome to KUQuickML")
        guideTitle.setFont(QFont('Arial', 18, QFont.Bold))
        guideTitle.setAlignment(Qt.AlignCenter)
        guideLayout.addWidget(guideTitle)

        guideText = QLabel(
            "이 프로그램은 머신러닝 초보자를 위한 GUI 툴입니다.\n\n"
            "① CSV 파일을 불러옵니다. (각 열은 Feature(x값), Sample명, Label(y값) 값을 지닙니다.)\n"
            "  label 값이 numerical 하지 않다면 임의의 숫자값을 배정합니다. ex) 0, 1, 2 \n"
            "② Data Scaling을 진행합니다.\n"
            "③ 데이터를 Train/Test 세트로 분할합니다.\n"
            "④ 알고리즘(KNN, MLP, RF, SVM)을 선택해 모델을 학습합니다.\n"
            "⑤ 모델을 저장하거나 Unknown Sample을 예측할 수 있습니다.\n"
            "csv 파일의 형식은 하단의 예시 참조\n"
        )
        guideText.setAlignment(Qt.AlignLeft)
        guideText.setWordWrap(True)
        guideLayout.addWidget(guideText)

        # 예시 CSV 표시
        sampleTable = QTableWidget()
        sampleTable.setRowCount(4)
        sampleTable.setColumnCount(6)
        sampleTable.setHorizontalHeaderLabels(["Sample","sepal.length", "sepal.width", "petal.length", "petal.width","variety"])
        sample_data = [
            [1, 5.1, 3.5, 1.4, 0.2, "Setosa"],
            [2, 6.2, 3.4, 5.4, 1.5, "Virginica"],
            [3, 5.8, 2.7, 5.1, 1.5, "Versicolor"],
            [4, 4.9, 3.0, 1.4, 1.2, "Setosa"]
        ]
        for i, row in enumerate(sample_data):
            for j, val in enumerate(row):
                sampleTable.setItem(i, j, QTableWidgetItem(str(val)))
        sampleTable.resizeColumnsToContents()
        guideLayout.addWidget(sampleTable)

        self.mainLayout.addWidget(self.guideWidget)

        # 실제 CSV 로드 후 보여질 뷰어
        self.csvViewer = CsvViewer()
        self.csvViewer.hide()  # 처음에는 숨김
        self.mainLayout.addWidget(self.csvViewer)

        self.mainTab.setLayout(self.mainLayout)

    def setupScaledDataTab(self):
        layout = QVBoxLayout()
        guide_frame = QFrame()
        guide_layout = QVBoxLayout(guide_frame)
        guide_label = QLabel(
            "<h3>📊 Data Scaling Guide</h3>"
            "<p>스케일링은 각 feature의 값 범위를 조정하여 모델 학습 성능을 높입니다.<br>"
            "데이터 특성과 모델 종류에 따라 적절한 스케일러를 선택하세요.</p>"
            "<ul>"
            "<li><b>StandardScaler</b>: 평균 0, 표준편차 1로 정규화. 대부분의 ML 모델에서 기본적으로 적합.<br>"
            "‣ 장점: 정규분포 데이터에 효과적.<br>"
            "‣ 단점: 이상치(outlier)에 민감.</li><br>"
            "<li><b>MinMaxScaler</b>: [0, 1] 범위로 스케일링.<br>"
            "‣ 장점: Neural Network 등에서 빠른 수렴 유도.<br>"
            "‣ 단점: 이상치에 매우 민감.</li><br>"
            "<li><b>RobustScaler</b>: 중앙값과 IQR(사분위 범위)을 기준으로 변환.<br>"
            "‣ 장점: 이상치가 많은 데이터에 안정적.<br>"
            "‣ 단점: 분포가 정규형에 가깝다면 오히려 precision 감소.</li><br>"
            "<li><b>MaxAbsScaler</b>: 각 feature의 최대 절댓값을 1로 맞춤.<br>"
            "‣ 장점: 희소 행렬(sparse data) 유지.<br>"
            "‣ 단점: 음수/양수 비율이 큰 데이터에는 부적합.</li><br>"
            "<li><b>Normalizer</b>: 각 샘플 벡터의 길이를 1로 맞춤.<br>"
            "‣ 장점: 텍스트 벡터나 거리 기반 모델(KNN)에 적합.<br>"
            "‣ 단점: 전체 feature 간 분포는 보정하지 않음.</li>"
            "</ul>"
        )
        guide_label.setWordWrap(True)
        guide_layout.addWidget(guide_label)
        guide_frame.setFrameShape(QFrame.Box)
        guide_frame.setStyleSheet("background-color: #fafafa; padding: 8px; border: 1px solid #ccc;")

        layout.addWidget(guide_frame)

        # 현재 스케일링 메서드 표시 라벨 추가
        self.scalerStatusLabel = QLabel("Current Scaling Method: None")
        self.scalerStatusLabel.setStyleSheet("font-weight: bold; color: darkgreen;")
        layout.addWidget(self.scalerStatusLabel)

        self.scaledDataWidget = QTableWidget()
        layout.addWidget(self.scaledDataWidget)
        self.scaledDataTab.setLayout(layout)

    def setupMLPTab(self):
        layout = QVBoxLayout()
        desc_style = "color: #555; font-size: 10pt; margin-bottom: 4px;"

        # 각 항목을 네모칸(QFrame)으로 묶는 함수
        def wrap_in_box(widget_list):
            frame = QFrame()
            frame.setFrameShape(QFrame.Box)
            frame.setFrameShadow(QFrame.Sunken)
            inner_layout = QVBoxLayout(frame)
            for w in widget_list:
                inner_layout.addWidget(w)
            inner_layout.setContentsMargins(8, 5, 8, 5)
            inner_layout.setSpacing(5)
            frame.setLayout(inner_layout)
            return frame

        # 왼쪽 컬럼
        left_col = QVBoxLayout()

        # Hidden Layer 설정
        hidden_layer_label = QLabel("Hidden Layer Size (comma separated):")
        self.hidden_layer_input = QLineEdit()
        self.hidden_layer_input.setPlaceholderText("50,50")
        self.hidden_layer_input.setFixedWidth(300)
        desc = QLabel(
            "은닉층 구조를 지정합니다. <br>(예: 100,50,30 → 각각 뉴런 100개, 50개, 30개로 이루어진 3개의 은닉층). <br>뉴런과 층이 많을수록 복잡한 패턴을 학습하지만 과적합 위험이 있습니다.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([hidden_layer_label, self.hidden_layer_input, desc]))

        # Alpha 설정
        alpha_label = QLabel("Alpha (Regularization strength):")
        self.alpha_input = QLineEdit()
        self.alpha_input.setPlaceholderText("0.0001")
        self.alpha_input.setFixedWidth(300)
        desc = QLabel("가중치의 크기를 제한해 과적합을 방지합니다. <br>값이 클수록 모델이 단순해지고, 작을수록 복잡해집니다.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([alpha_label, self.alpha_input, desc]))

        # Max Iteration
        max_iter_label = QLabel("Max Iterations:")
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 999999)
        self.max_iter_input.setValue(1000)
        self.max_iter_input.setFixedWidth(300)
        desc = QLabel("최대 학습 반복 횟수입니다. 수렴하지 않을 경우 값을 높이세요.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([max_iter_label, self.max_iter_input, desc]))

        # Random State
        random_state_label = QLabel("Random State:")
        self.random_state_input = QSpinBox()
        self.random_state_input.setRange(0, 999999)
        self.random_state_input.setValue(42)
        self.random_state_input.setFixedWidth(300)
        desc = QLabel("무작위 초기화 시드를 고정합니다. 같은 결과를 재현하려면 같은 값을 유지하세요.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([random_state_label, self.random_state_input, desc]))

        # 오른쪽 컬럼
        right_col = QVBoxLayout()

        # Solver
        solver_label = QLabel("Solver:")
        self.solver_input = QComboBox()
        self.solver_input.addItems(['adam', 'sgd', 'lbfgs'])
        self.solver_input.setCurrentText('adam')
        self.solver_input.setFixedWidth(300)
        desc = QLabel(
            "가중치 최적화 알고리즘입니다. <br>'adam': 안정적 <br>'lbfgs': 적은 데이터셋에 적합 <br> 'sgd': 대규모 데이터셋에 적합, 최적화 과정 조정 가능")
        desc.setStyleSheet(desc_style)
        right_col.addWidget(wrap_in_box([solver_label, self.solver_input, desc]))

        # Activation
        activation_label = QLabel("Activation Function:")
        self.activation_input = QComboBox()
        self.activation_input.addItems(['identity', 'logistic', 'tanh', 'relu'])
        self.activation_input.setCurrentText('relu')
        self.activation_input.setFixedWidth(300)
        desc = QLabel(
            "활성화 함수는 뉴런의 출력 형태를 결정합니다. <br>'relu': 가장 일반적이고 안정적 <br> 'tanh': 높은 학습 안정성 <br> 'logistic':이진 분류 출력층 또는 작은 네트워크에서 사용 <br> 'identity': regression에 적합 ")
        desc.setStyleSheet(desc_style)
        right_col.addWidget(wrap_in_box([activation_label, self.activation_input, desc]))

        # Learning Rate
        learning_rate_label = QLabel("Learning Rate (learning_rate_init):")
        self.learning_rate_input = QLineEdit()
        self.learning_rate_input.setPlaceholderText("0.001")
        self.learning_rate_input.setFixedWidth(300)
        desc = QLabel("가중치 업데이트 속도를 조절합니다. 너무 크면 불안정, 너무 작으면 학습이 느립니다.")
        desc.setStyleSheet(desc_style)
        right_col.addWidget(wrap_in_box([learning_rate_label, self.learning_rate_input, desc]))
        # Font 설정 영역 (UI에 표시)
        font_box = QFrame()
        font_box.setFrameShape(QFrame.Box)
        font_box.setFrameShadow(QFrame.Sunken)
        font_box.setFixedHeight(80)  # 전체 박스 높이 제한
        font_layout = QVBoxLayout(font_box)
        font_layout.setContentsMargins(4, 2, 4, 2)  # 여백 최소화
        font_layout.setSpacing(1)  # 위젯 간격 최소화

        font_label = QLabel("Font settings:")
        font_label.setStyleSheet("font-size: 8pt; margin-bottom: 0px;")  # 폰트 작게
        font_layout.addWidget(font_label)

        font_row = QHBoxLayout()
        font_row.setSpacing(4)

        # Font type
        self.fontTypeComboBox = QComboBox()
        self.fontTypeComboBox.addItems(["Arial", "Calibri", "Times New Roman", "Verdana"])
        self.fontTypeComboBox.setFixedHeight(20)
        self.fontTypeComboBox.setFixedWidth(120)

        # Font size
        font_size_label = QLabel("Size:")
        font_size_label.setStyleSheet("font-size: 8pt; margin-right: 2px;")
        self.fontSizeInput = QSpinBox()
        self.fontSizeInput.setRange(6, 32)
        self.fontSizeInput.setValue(12)
        self.fontSizeInput.setFixedHeight(20)
        self.fontSizeInput.setFixedWidth(50)

        font_row.addWidget(self.fontTypeComboBox)
        font_row.addWidget(font_size_label)
        font_row.addWidget(self.fontSizeInput)

        font_layout.addLayout(font_row)
        layout.addWidget(font_box)

        # 버튼 영역
        buttons_layout = QHBoxLayout()
        self.createMLPClassModelButton = QPushButton("Create MLP Classification Model")
        self.createMLPClassModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createMLPClassModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createMLPClassModelButton.clicked.connect(self.createMLPClassificationModel)
        buttons_layout.addWidget(self.createMLPClassModelButton)

        self.createMLPRegModelButton = QPushButton("Create MLP Regressionression Model")
        self.createMLPRegModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createMLPRegModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createMLPRegModelButton.clicked.connect(self.createMLPRegressionModel)
        buttons_layout.addWidget(self.createMLPRegModelButton)

        # 좌우 배치
        main_columns = QHBoxLayout()
        main_columns.addLayout(left_col)
        main_columns.addSpacing(20)
        main_columns.addLayout(right_col)

        layout.addLayout(main_columns)
        layout.addSpacing(15)
        layout.addLayout(buttons_layout)

        self.MLPTab.setLayout(layout)
        self.tabs.addTab(self.MLPTab, "MLP")
        self.tabs.setCurrentWidget(self.MLPTab)



    def createMLPClassificationModel(self):
        if not self.checkDataSplit():
            return
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        self.model_features = X_train.columns.tolist()
        self.model_features.remove('Sample')

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        # Apply the scaler
        if self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train_numeric)
            X_test_scaled = self.scaler.transform(X_test_numeric)
            self.fitted_scaler = self.scaler

            if hasattr(X_train_numeric, "columns"):
                X_train_numeric = pd.DataFrame(
                    X_train_scaled,
                    columns=X_train_numeric.columns,
                    index=X_train_numeric.index
                )
                X_test_numeric = pd.DataFrame(
                    X_test_scaled,
                    columns=X_test_numeric.columns,
                    index=X_test_numeric.index
                )
            else:
                X_train_numeric = X_train_scaled
                X_test_numeric = X_test_scaled

        # Hidden layer size 설정
        hidden_layer_input_text = self.hidden_layer_input.text()
        if not hidden_layer_input_text:
            hidden_layers = (50, 50)
        else:
            hidden_layers = tuple(map(int, hidden_layer_input_text.split(',')))

        # Alpha 값 설정
        alpha_input_text = self.alpha_input.text()
        if not alpha_input_text:
            alpha = 0.0001
        else:
            alpha = float(alpha_input_text)
        # Learning rate 설정
        lr_input_text = self.learning_rate_input.text()
        if not lr_input_text:
            learning_rate = 0.001
        else:
            learning_rate = float(lr_input_text)

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=int(self.max_iter_input.value()),
            random_state=int(self.random_state_input.value()),
            alpha=alpha,
            solver=self.solver_input.currentText(),
            activation=self.activation_input.currentText(),
            learning_rate_init=learning_rate
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            mlp.fit(X_train_numeric, y_train)

        if mlp.n_iter_ == mlp.max_iter:
            QMessageBox.warning(self, "Iteration Warning", "Maximum iterations reached. Consider increasing max_iter.")

        y_pred_train = mlp.predict(X_train_numeric)
        y_pred_test = mlp.predict(X_test_numeric)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        self.latest_model = (mlp, None)  # Save the latest model
        cm = confusion_matrix(y_test, y_pred_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
            precision = np.nan_to_num(precision)  # Convert NaNs to zero

        # Automatically generate index and columns based on cm shape
        labels = [f"Class {i}" for i in range(cm.shape[0])]
        cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in labels],
                             columns=[f"Predicted {label}" for label in labels])
        cm_df['Prediction Accuracy (%)'] = precision
        self.showConfusionMatrix(cm_df)

        self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)
        perm_importance = permutation_importance(mlp, X_test_numeric, y_test, n_repeats=10, random_state=42)
        sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
        feature_importances = [(X_train_numeric.columns[idx], perm_importance.importances_mean[idx]) for idx in sorted_idx]

        self.showMLPFeatureImportances(feature_importances)
        self.models['MLP Classification'] = mlp
    def createMLPRegressionModel(self):
        if not self.checkDataSplit():
            return
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()
        self.model_features = X_train.columns.tolist()
        self.model_features.remove('Sample')

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        # Apply the scaler
        if self.scaler is not None:

            X_train_scaled = self.scaler.fit_transform(X_train_numeric)
            X_test_scaled = self.scaler.transform(X_test_numeric)
            self.fitted_scaler = self.scaler

            if hasattr(X_train_numeric, "columns"):
                X_train_numeric = pd.DataFrame(
                    X_train_scaled,
                    columns=X_train_numeric.columns,
                    index=X_train_numeric.index
                )
                X_test_numeric = pd.DataFrame(
                    X_test_scaled,
                    columns=X_test_numeric.columns,
                    index=X_test_numeric.index
                )
            else:
                X_train_numeric = X_train_scaled
                X_test_numeric = X_test_scaled

        # Hidden layer size 설정
        hidden_layer_input_text = self.hidden_layer_input.text()
        if not hidden_layer_input_text:
            hidden_layers = (50, 50)
        else:
            hidden_layers = tuple(map(int, hidden_layer_input_text.split(',')))

        # Alpha 값 설정
        alpha_input_text = self.alpha_input.text()
        if not alpha_input_text:
            alpha = 0.0001
        else:
            alpha = float(alpha_input_text)
        # Learning rate 설정
        lr_input_text = self.learning_rate_input.text()
        if not lr_input_text:
            learning_rate = 0.001
        else:
            learning_rate = float(lr_input_text)

        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=int(self.max_iter_input.value()),
            random_state=int(self.random_state_input.value()),
            alpha=alpha,
            solver=self.solver_input.currentText(),
            activation=self.activation_input.currentText(),
            learning_rate_init=learning_rate
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            mlp.fit(X_train_numeric, y_train)

        if mlp.n_iter_ == mlp.max_iter:
            QMessageBox.warning(self, "Iteration Warning", "Maximum iterations reached. Consider increasing max_iter.")

        y_pred_train = mlp.predict(X_train_numeric)
        y_pred_test = mlp.predict(X_test_numeric)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        self.latest_model = (mlp, None)  # Save the latest model

        self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)
        perm_importance = permutation_importance(mlp, X_test_numeric, y_test, n_repeats=10, random_state=42)
        sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
        # Feature 이름 안전 처리
        if hasattr(X_train_numeric, "columns"):
            feature_names = X_train_numeric.columns
        else:
            feature_names = [f"Feature {i}" for i in range(X_train_numeric.shape[1])]

        feature_importances = [
            (feature_names[idx], perm_importance.importances_mean[idx])
            for idx in sorted_idx
        ]

        self.showMLPFeatureImportances(feature_importances)
        self.models['MLP Regression'] = mlp
    def showMLPFeatureImportances(self, feature_importances):
        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Importances")
        dialog.setGeometry(100, 100, 600, 400)  # 창 크기를 조금 더 크게 설정

        dialog_layout = QVBoxLayout(dialog)

        #info_label = QLabel(
            #"Important Features\n\n"
            #"MLP는 인공신경망 특유의 복잡성으로 일반적으로 어떤 특성이 예측에 중요한 역할을 하는지 명확하게 해석하기 어렵습니다.\n\n"
            #"Ku Machine Learning은 permutation feature importance를 측정합니다.\n\n"
            #"이는 features 중 하나만을 랜덤으로 섞어버렸을 때 모델의 정확도 점수가 얼마나 낮아지는가를 통해 계산하는 방식입니다.\n\n"
            #"수치가 높을 수록 모델 생성에 더 큰 영향을 미쳤다고 할 수 있습니다. 마이너스 값은 모델 생성에 부정적인 영향을 주었을 가능성을 시사합니다.\n\n"
            #"<a href=\"https://scikit-learn.org/stable/modules/permutation_importance.html#id2\">상세는 이하의 링크를 참고해주십시오.</a>"
        #)
        #info_label.setOpenExternalLinks(True)
        #info_label.setWordWrap(True)  # 텍스트가 창 크기에 맞게 줄 바꿈 되도록 설정
        #dialog_layout.addWidget(info_label)

        table = QTableWidget(dialog)
        table.setRowCount(len(feature_importances))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Feature", "Importance"])

        for i, (feature, importance) in enumerate(feature_importances):
            table.setItem(i, 0, QTableWidgetItem(str(feature)))
            table.setItem(i, 1, QTableWidgetItem(f"{importance:.4f}"))

        table.resizeColumnsToContents()
        dialog_layout.addWidget(table)

        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()

    def setupRFTab(self):
        layout = QVBoxLayout()

        desc_style = "color: #555; font-size: 10pt; margin-bottom: 4px;"

        # --- 상단: Random Forest 설명 추가 ---
        rf_overview = QLabel(
            "<h3>🌲 Random Forest (RF)</h3>"
            "<p>Random Forest는 여러 개의 의사결정트리를 학습시켜 예측을 수행하는 앙상블 학습 기법입니다.<br>"
            "과적합 위험이 낮고, 분류(Classification)와 회귀(Regression) 문제 모두에서 우수한 성능을 보입니다.</p>"
        )
        rf_overview.setWordWrap(True)
        layout.addWidget(rf_overview)

        # --- 파라미터 그룹박스 ---
        groupBox = QFrame()
        groupBox.setFrameShape(QFrame.Box)
        groupBox.setFrameShadow(QFrame.Sunken)
        groupBoxLayout = QVBoxLayout(groupBox)

        params = [
            ("Max Depth:", 20, 1, 99999, "트리의 최대 깊이를 제한합니다.<br>값이 크면 모델이 복잡해지고, 작으면 단순해집니다."),
            ("N Estimators:", 20, 1, 99999, "생성할 트리의 개수입니다.<br>많을수록 안정적인 결과를 얻지만 학습 시간이 길어집니다."),
            ("Min Samples Leaf:", 1, 1, 99999, "각 리프 노드에 있어야 하는 최소 샘플 수입니다.<br>값이 크면 모델이 단순해지고 과적합이 줄어듭니다."),
            ("Min Samples Split:", 2, 2, 99999, "노드를 분할하기 위한 최소 샘플 수입니다.<br>값이 크면 트리의 깊이가 줄어들어 단순한 모델이 됩니다."),
            ("Random State:", 0, 0, 99999, "무작위성 제어를 위한 시드(seed) 값입니다.<br>같은 값을 사용하면 항상 동일한 결과를 재현할 수 있습니다.")
        ]

        self.param_inputs = {}
        for label, default, min_val, max_val, explanation in params:
            frame = QFrame()
            frame.setFrameShape(QFrame.Box)
            frame.setFrameShadow(QFrame.Sunken)
            frame_layout = QVBoxLayout(frame)

            lbl = QLabel(f"<b>{label}</b>")
            spinbox = QSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default)
            spinbox.setFixedWidth(300)

            desc = QLabel(explanation)
            desc.setWordWrap(True)
            desc.setStyleSheet(desc_style)

            frame_layout.addWidget(lbl)
            frame_layout.addWidget(spinbox)
            frame_layout.addWidget(desc)
            groupBoxLayout.addWidget(frame)

            self.param_inputs[label] = spinbox

        groupBoxLayout.setContentsMargins(5, 5, 5, 5)
        groupBoxLayout.setSpacing(15)
        layout.addWidget(groupBox)

        # --- 버튼 영역 ---
        buttons_layout = QHBoxLayout()
        self.createRFClassModelButton = QPushButton("Create RF Classification Model")
        self.createRFClassModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createRFClassModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createRFClassModelButton.clicked.connect(self.createRFClassificationModel)
        buttons_layout.addWidget(self.createRFClassModelButton)

        self.createRFRegModelButton = QPushButton("Create RF Regression Model")
        self.createRFRegModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createRFRegModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createRFRegModelButton.clicked.connect(self.createRFRegressionModel)
        buttons_layout.addWidget(self.createRFRegModelButton)
        layout.addLayout(buttons_layout)

        # --- 오른쪽: 폰트 설정 ---
        right_side_layout = QVBoxLayout()
        right_side_layout.setAlignment(Qt.AlignTop)

        font_box = QFrame()
        font_box.setFrameShape(QFrame.Box)
        font_box.setFrameShadow(QFrame.Sunken)
        font_layout = QVBoxLayout(font_box)

        font_label = QLabel("Font settings:")
        font_type_label = QLabel("Font type:")
        self.fontTypeComboBox = QComboBox()
        self.fontTypeComboBox.addItems(["Arial", "Calibri", "Times New Roman", "Verdana"])

        font_size_label = QLabel("Font size:")
        self.fontSizeInput = QSpinBox()
        self.fontSizeInput.setRange(6, 32)
        self.fontSizeInput.setValue(12)

        font_layout.addWidget(font_label)
        font_layout.addWidget(font_type_label)
        font_layout.addWidget(self.fontTypeComboBox)
        font_layout.addWidget(font_size_label)
        font_layout.addWidget(self.fontSizeInput)

        right_side_layout.addWidget(font_box)

        # --- 전체 병합 ---
        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(right_side_layout)

        self.RFTab.setLayout(main_layout)

    def createRFClassificationModel(self):
        if not self.checkDataSplit():
            return
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        self.model_features = X_train.columns.tolist()
        self.model_features.remove('Sample')

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        # Apply the scaler
        if self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train_numeric)
            X_test_scaled = self.scaler.transform(X_test_numeric)
            self.fitted_scaler = self.scaler

            if hasattr(X_train_numeric, "columns"):
                X_train_numeric = pd.DataFrame(
                    X_train_scaled,
                    columns=X_train_numeric.columns,
                    index=X_train_numeric.index
                )
                X_test_numeric = pd.DataFrame(
                    X_test_scaled,
                    columns=X_test_numeric.columns,
                    index=X_test_numeric.index
                )
            else:
                X_train_numeric = X_train_scaled
                X_test_numeric = X_test_scaled

        rf_clf = RandomForestClassifier(
            n_estimators=self.param_inputs["N Estimators:"].value(),
            max_depth=self.param_inputs["Max Depth:"].value(),
            min_samples_leaf=self.param_inputs["Min Samples Leaf:"].value(),
            min_samples_split=self.param_inputs["Min Samples Split:"].value(),
            random_state=0,
            n_jobs=-1
        )

        rf_clf.fit(X_train_numeric, y_train)
        y_pred_train = rf_clf.predict(X_train_numeric)
        y_pred_test = rf_clf.predict(X_test_numeric)
        accuracy = accuracy_score(y_test, y_pred_test)

        # Confusion matrix
        unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
        true = [f'true_{label}' for label in unique_labels]
        pred = [f'pred_{label}' for label in unique_labels]
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(cm, index=true, columns=pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
            precision = np.nan_to_num(precision)  # Convert NaNs to zero

        cm_df['Prediction Accuracy (%)'] = precision

        # R2 scores
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        # Variable importances
        characteristics = X_train_numeric.columns
        importances = rf_clf.feature_importances_
        variable_importances = sorted(zip(characteristics, importances), key=lambda x: x[1], reverse=True)

        self.showRFClfResults(accuracy, cm_df, r2_train, r2_test, variable_importances)
        self.plotObservedVsPredicted(y_train, y_pred_train, y_test, y_pred_test,
                                     "RF Classification: Observed vs Predicted")
        self.models['RF Classification'] = rf_clf
    def createRFRegressionModel(self):
        if not self.checkDataSplit():
            return
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        self.model_features = X_train.columns.tolist()
        self.model_features.remove('Sample')

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        # Apply the scaler
        if self.scaler is not None:
            X_train_scaled = self.scaler.fit_transform(X_train_numeric)
            X_test_scaled = self.scaler.transform(X_test_numeric)
            self.fitted_scaler = self.scaler

            if hasattr(X_train_numeric, "columns"):
                X_train_numeric = pd.DataFrame(
                    X_train_scaled,
                    columns=X_train_numeric.columns,
                    index=X_train_numeric.index
                )
                X_test_numeric = pd.DataFrame(
                    X_test_scaled,
                    columns=X_test_numeric.columns,
                    index=X_test_numeric.index
                )
            else:
                X_train_numeric = X_train_scaled
                X_test_numeric = X_test_scaled

        rf_regr = RandomForestRegressor(
            n_estimators=self.param_inputs["N Estimators:"].value(),
            max_depth=self.param_inputs["Max Depth:"].value(),
            min_samples_leaf=self.param_inputs["Min Samples Leaf:"].value(),
            min_samples_split=self.param_inputs["Min Samples Split:"].value(),
            random_state=0,
            n_jobs=-1
        )

        rf_regr.fit(X_train_numeric, y_train)
        y_pred_train = rf_regr.predict(X_train_numeric)
        y_pred_test = rf_regr.predict(X_test_numeric)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        # Variable importances
        characteristics = X_train_numeric.columns
        importances = rf_regr.feature_importances_
        variable_importances = sorted(zip(characteristics, importances), key=lambda x: x[1], reverse=True)

        self.showRFRegResults(r2_train, r2_test, mse_test, rmse_test, variable_importances)
        self.plotObservedVsPredicted(y_train, y_pred_train, y_test, y_pred_test, "RF Regressionression: Observed vs Predicted")
        self.models['RF Regression'] = rf_regr
    def showRFRegResults(self, r2_train, r2_test, mse_test, rmse_test, variable_importances):
        dialog = QDialog(self)
        dialog.setWindowTitle("Random Forest Regression Results")
        dialog.setGeometry(100, 100, 800, 600)
        dialog_layout = QVBoxLayout(dialog)

        # R2 and MSE scores
        results_label = QLabel(
                               f"Training Set R2 Score: {r2_train:.3f}\n"
                               f"Test Set R2 Score: {r2_test:.3f}\n"
                               f"Test Set MSE: {mse_test:.3f}\n"
                               f"Test Set RMSE: {rmse_test:.3f}")
        dialog_layout.addWidget(results_label)

        # Variable importances
        var_importance_label = QLabel("Variable Importances:")
        dialog_layout.addWidget(var_importance_label)

        var_importance_table = QTableWidget(dialog)
        var_importance_table.setRowCount(len(variable_importances))
        var_importance_table.setColumnCount(2)
        var_importance_table.setHorizontalHeaderLabels(["Variable", "Importance"])

        for i, (var, imp) in enumerate(variable_importances):
            var_importance_table.setItem(i, 0, QTableWidgetItem(var))
            var_importance_table.setItem(i, 1, QTableWidgetItem(f"{imp:.3f}"))

        var_importance_table.resizeColumnsToContents()
        dialog_layout.addWidget(var_importance_table)

        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()
    def showRFClfResults(self, accuracy, cm_df, r2_train, r2_test, variable_importances):
        dialog = QDialog(self)
        dialog.setWindowTitle("Random Forest Results")
        dialog.setGeometry(100, 100, 800, 600)
        dialog_layout = QVBoxLayout(dialog)

        # Accuracy and R2 scores
        results_label = QLabel(f"Confusion Matrix Accuracy (y test vs y pred test: {accuracy:.4f}\n"
                               f"Training Set R2 Score: {r2_train:.3f}\n"
                               f"Test Set R2 Score: {r2_test:.3f}\n")
        dialog_layout.addWidget(results_label)

        # Confusion matrix table
        cm_table = QTableWidget(dialog)
        cm_table.setRowCount(cm_df.shape[0])
        cm_table.setColumnCount(cm_df.shape[1])
        cm_table.setHorizontalHeaderLabels(cm_df.columns)
        cm_table.setVerticalHeaderLabels(cm_df.index)

        for i in range(cm_df.shape[0]):
            for j in range(cm_df.shape[1]):
                item = QTableWidgetItem(f"{cm_df.iloc[i, j]:.3f}")
                cm_table.setItem(i, j, item)

        cm_table.resizeColumnsToContents()
        dialog_layout.addWidget(cm_table)

        # Variable importances
        var_importance_label = QLabel("Variable Importances:")
        dialog_layout.addWidget(var_importance_label)

        var_importance_table = QTableWidget(dialog)
        var_importance_table.setRowCount(len(variable_importances))
        var_importance_table.setColumnCount(2)
        var_importance_table.setHorizontalHeaderLabels(["Variable", "Importance"])

        for i, (var, imp) in enumerate(variable_importances):
            var_importance_table.setItem(i, 0, QTableWidgetItem(var))
            var_importance_table.setItem(i, 1, QTableWidgetItem(f"{imp:.3f}"))

        var_importance_table.resizeColumnsToContents()
        dialog_layout.addWidget(var_importance_table)

        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()

    def setupPredictionTab(self):
        layout = QVBoxLayout()

        # 상단 안내 문구
        title_label = QLabel("Prediction with Loaded Model")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        info_label = QLabel(
            "1. Load a trained model (File → Load Previous Model)\n"
            "2. Load an unknown CSV file (features only)\n"
            "3. Scaling and dimensionality reduction will be automatically applied\n"
            "4. Predictions will be displayed below."
        )
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)

        # 모델 로드 버튼
        load_model_button = QPushButton("Load Saved Model")
        load_model_button.setFont(QFont('Arial', 12, QFont.Bold))
        load_model_button.setStyleSheet("QPushButton { padding: 8px; border-radius: 8px; border: 2px solid #000000; }")
        load_model_button.clicked.connect(self.loadPreviousModel)
        layout.addWidget(load_model_button)

        # Unknown CSV 로드 버튼
        load_unknown_button = QPushButton("Load Unknown Data (CSV)")
        load_unknown_button.setFont(QFont('Arial', 12, QFont.Bold))
        load_unknown_button.setStyleSheet(
            "QPushButton { padding: 8px; border-radius: 8px; border: 2px solid #000000; }")
        load_unknown_button.clicked.connect(self.loadUnknownSample)
        layout.addWidget(load_unknown_button)

        # 예측 결과 테이블
        self.prediction_table = QTableWidget()
        layout.addWidget(self.prediction_table)

        # 상태 메시지
        self.prediction_status = QLabel("Ready for prediction.")
        self.prediction_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_status)

        self.predictionTab.setLayout(layout)

    def loadUnknownSample(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Unknown Sample CSV File", "",
            "CSV Files (*.csv);;All Files (*)", options=options
        )
        if not filename:
            return

        try:
            # ① unknown CSV 로드
            self.unknown_data = pd.read_csv(filename)
            numeric_columns = self.unknown_data.select_dtypes(include=[np.number]).columns
            data_to_scale = self.unknown_data[numeric_columns].copy()

            # ② 로드된 모델 확인
            if not hasattr(self, "loaded_bundle"):
                QMessageBox.warning(self, "Error", "Please load a trained model first.")
                return

            bundle = self.loaded_bundle
            model = bundle.get("model")
            scaler = bundle.get("scaler")
            reducer = bundle.get("reducer")
            feature_names = bundle.get("feature_names")
            label_mapping = bundle.get("label_mapping")

            if model is None:
                QMessageBox.warning(self, "Error", "No model found in bundle.")
                return

            # ③ feature 순서 맞추기
            if feature_names is not None:
                data_to_scale = data_to_scale.reindex(columns=feature_names)
                print("[Auto-align] Columns reordered to match training features.")
            else:
                print("[Warning] Model has no saved feature names — predictions may be unreliable!")

            # ④ 저장된 scaler 적용
            if scaler:
                data_scaled = scaler.transform(data_to_scale)
                data_scaled = pd.DataFrame(data_scaled, columns=data_to_scale.columns, index=data_to_scale.index)
            else:
                data_scaled = data_to_scale

            # ⑤ reducer 적용
            if reducer:
                data_reduced = reducer.transform(data_scaled)
                data_used = pd.DataFrame(
                    data_reduced,
                    columns=[f"Component {i + 1}" for i in range(data_reduced.shape[1])],
                    index=data_scaled.index
                )
            else:
                data_used = data_scaled

            # ⑥ 예측
            predictions = model.predict(data_used)

            # ⑦ label 역매핑 적용 (문자형 복원)
            if label_mapping:
                inverse_map = {v: k for k, v in label_mapping.items()}
                predictions = [inverse_map.get(p, p) for p in predictions]

            # ⑧ 결과 표시
            self.prediction_table.clear()
            self.prediction_table.setColumnCount(2)
            self.prediction_table.setHorizontalHeaderLabels(["Sample", "Prediction"])
            self.prediction_table.setRowCount(len(predictions))

            for i, pred in enumerate(predictions):
                sample_name = (
                    self.unknown_data.iloc[i, 0]
                    if "Sample" in self.unknown_data.columns
                    else f"Sample {i + 1}"
                )
                self.prediction_table.setItem(i, 0, QTableWidgetItem(str(sample_name)))
                self.prediction_table.setItem(i, 1, QTableWidgetItem(str(pred)))

            self.prediction_table.resizeColumnsToContents()
            self.prediction_status.setText(
                f"Predictions completed successfully using model '{type(model).__name__}'."
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load or predict:\n{e}")

    def showUnknownData(self, data):
        self.unknownTable.setRowCount(len(data))
        self.unknownTable.setColumnCount(len(data.columns))
        self.unknownTable.setHorizontalHeaderLabels(data.columns)
        for i, row in data.iterrows():
            for j, cell in enumerate(row):
                self.unknownTable.setItem(i, j, QTableWidgetItem(str(cell)))

    def scaleUnknownData(self):
        if self.unknown_data is None:
            QMessageBox.warning(self, "Data Error", "No unknown data loaded.")
            return

        # Select only numeric columns for scaling
        numeric_columns = self.unknown_data.select_dtypes(include=[np.number]).columns
        data_to_scale = self.unknown_data[numeric_columns]

        scaler_name = self.scalerComboBox.currentText()
        scaler = self.scalers[scaler_name]

        # Perform scaling
        self.scaled_unknown_data = scaler.transform(data_to_scale)
        scaled_df = pd.DataFrame(self.scaled_unknown_data, columns=numeric_columns)

        # Include non-numeric columns in the scaled data frame
        non_numeric_columns = self.unknown_data.select_dtypes(exclude=[np.number]).columns
        for column in non_numeric_columns:
            scaled_df[column] = self.unknown_data[column]

        self.showUnknownData(scaled_df)

    def applyPredictionScaler(self):
        scaler_name = self.scalerComboBox.currentText()
        self.scaler = self.scalers[scaler_name]

        # Select only numeric columns for scaling
        numeric_columns = self.unknown_data.select_dtypes(include=[np.number]).columns
        data_to_scale = self.unknown_data[numeric_columns]


        self.scaled_unknown_data = self.scaler.transform(data_to_scale)

        # Include non-numeric columns in the scaled data frame
        non_numeric_columns = self.unknown_data.select_dtypes(exclude=[np.number]).columns
        scaled_df = pd.DataFrame(self.scaled_unknown_data, columns=numeric_columns)
        for column in non_numeric_columns:
            scaled_df[column] = self.unknown_data[column].values

        self.showScaledUnknownData(scaled_df)

    def showScaledUnknownData(self):
        scaled_df = pd.DataFrame(self.scaled_unknown_data, columns=self.unknown_data.columns)
        self.unknownTable.clear()
        self.unknownTable.setRowCount(len(scaled_df))
        self.unknownTable.setColumnCount(len(scaled_df.columns))
        self.unknownTable.setHorizontalHeaderLabels(scaled_df.columns)

        for i, row in scaled_df.iterrows():
            for j, cell in enumerate(row):
                self.unknownTable.setItem(i, j, QTableWidgetItem(str(cell)))

    # applyModelReducer (feature 이름 기준 align)
    # ============================================================
    def applyModelReducer(self):
        try:
            if not hasattr(self, 'scaled_unknown_data'):
                QMessageBox.warning(self, "Data Error", "Please scale the unknown data first.")
                return

            # 현재 선택된 모델 이름 확인
            selected_models = [name for name, checkbox in self.modelCheckBoxes.items() if checkbox.isChecked()]
            if not selected_models:
                QMessageBox.warning(self, "Model Selection Error", "Please select at least one model.")
                return

            model_name = selected_models[0]
            reducer = self.model_reducers.get(model_name, None)
            if reducer is None:
                QMessageBox.warning(self, "Reducer Error", f"No reducer found for '{model_name}'.")
                return

            # feature 이름 기준 align
            unknown_df = pd.DataFrame(self.scaled_unknown_data, columns=self.unknown_data.columns)
            if hasattr(self, 'feature_names'):
                missing = set(self.feature_names) - set(unknown_df.columns)
                if missing:
                    QMessageBox.warning(self, "Feature Mismatch",
                                        f"The following features are missing in unknown data:\n{', '.join(missing)}")
                    return
                # feature 이름 기준 재정렬
                unknown_df = unknown_df.loc[:, self.feature_names]

            reduced = reducer.transform(unknown_df)
            self.reduced_unknown_data = reduced
            self.showUnknownData(pd.DataFrame(reduced))
            print(f"[Reducer Applied] Using reducer from {model_name}")

        except Exception as e:
            QMessageBox.warning(self, "Reducer Error", f"Reducer could not be applied:\n{e}")
            print(f"[Reducer Error] {e}")

    def showReducedUnknownData(self, reduced_data):
        self.unknownTable.clear()
        self.unknownTable.setRowCount(len(reduced_data))
        self.unknownTable.setColumnCount(reduced_data.shape[1])
        self.unknownTable.setHorizontalHeaderLabels([f"Component {i + 1}" for i in range(reduced_data.shape[1])])

        for i, row in enumerate(reduced_data):
            for j, value in enumerate(row):
                self.unknownTable.setItem(i, j, QTableWidgetItem(f"{value:.4f}"))

    def predictModel(self):
        """
        예측 시 feature 이름 기반으로 align하여 reducer 및 model 예측 수행
        """
        if not hasattr(self, 'unknown_data'):
            QMessageBox.warning(self, "Data Error", "Please load unknown sample CSV first.")
            return

        if not hasattr(self, 'scaled_unknown_data'):
            QMessageBox.warning(self, "Scaling Error", "Please scale the unknown data first.")
            return

        selected_models = [name for name, cb in self.modelCheckBoxes.items() if cb.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "Model Selection Error", "Please select at least one trained model.")
            return

        # feature 이름 기준 align
        unknown_df = pd.DataFrame(self.scaled_unknown_data, columns=self.unknown_data.columns)
        if hasattr(self, 'feature_names'):
            missing = set(self.feature_names) - set(unknown_df.columns)
            if missing:
                QMessageBox.warning(self, "Feature Mismatch",
                                    f"The following features are missing in unknown data:\n{', '.join(missing)}")
                return
            unknown_df = unknown_df.loc[:, self.feature_names]

        data_used = unknown_df.values

        # reducer 자동 적용
        try:
            valid_models = ["KNN Classification", "KNN Regression", "SV Classification", "SV Regression"]
            valid_selected_model = next((m for m in selected_models if m in valid_models), None)

            if valid_selected_model and valid_selected_model in self.model_reducers:
                reducer = self.model_reducers[valid_selected_model]
                reduced = reducer.transform(data_used)
                self.reduced_unknown_data = reduced
                self.showUnknownData(pd.DataFrame(reduced))
                data_used = reduced
                print(f"[Reducer Applied] Using reducer from {valid_selected_model}")
        except Exception as e:
            print(f"[Reducer Auto-Apply Error] {e}")
            QMessageBox.warning(self, "Reducer Error", f"Reducer could not be applied:\n{e}")

        # 예측 실행
        predictions = {}
        for model_name in selected_models:
            model = self.models.get(model_name)
            if not model:
                QMessageBox.warning(self, "Model Error", f"Model '{model_name}' not found.")
                continue

            try:
                predictions[model_name] = model.predict(data_used)
            except Exception as e:
                QMessageBox.warning(self, "Prediction Error", f"Error predicting with '{model_name}':\n{e}")
                continue

        # 결과 표시 (UI 유지)
        sample_names = self.get_sample_names_from_unknown_data()
        if predictions:
            self.showPredictions(predictions, sample_names)
        else:
            QMessageBox.information(self, "Prediction Info", "No predictions were generated.")
    def get_sample_names_from_unknown_data(self):
        if 'Sample' in self.unknown_data.columns:
            sample_names = self.unknown_data['Sample']
        elif 'ID' in self.unknown_data.columns:
            sample_names = self.unknown_data['ID']
        elif 'Name' in self.unknown_data.columns:
            sample_names = self.unknown_data['Name']
        elif 'sample' in self.unknown_data.columns:
            sample_names = self.unknown_data['sample']
        else:
            sample_names = pd.Series(range(1, len(self.unknown_data) + 1))
        return sample_names

    def showPredictions(self, predictions, sample_names):
        dialog = QDialog(self)
        dialog.setWindowTitle("Predictions")
        dialog.setGeometry(100, 100, 600, 400)
        dialog_layout = QVBoxLayout(dialog)

        table = QTableWidget(dialog)
        table.setRowCount(len(sample_names))
        table.setColumnCount(len(predictions) + 1)
        table.setHorizontalHeaderLabels(["Sample Name"] + list(predictions.keys()))

        for i, sample in enumerate(sample_names):
            table.setItem(i, 0, QTableWidgetItem(str(sample)))
            for j, model_name in enumerate(predictions.keys(), start=1):
                table.setItem(i, j, QTableWidgetItem(str(predictions[model_name][i])))

        table.resizeColumnsToContents()
        dialog_layout.addWidget(table)
        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()

    def showMLPResults(self, y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test):
        plt.rcParams['font.size'] = self.fontSizeInput.value()
        plt.rcParams['font.family'] = self.fontTypeComboBox.currentText()

        fig, ax = plt.subplots(figsize=(10, 8))

        scatter_train = ax.scatter(y_train, y_pred_train, c='blue', label='Training Set', marker='o', s=50, alpha=0.3)
        scatter_test = ax.scatter(y_test, y_pred_test, c='red', label='Test Set', marker='x', s=100, alpha=0.7)

        # x축, y축 이름 설정
        x_label = 'Observed'
        y_label = 'Predicted'
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Figure 제목 설정
        title = 'MLP Predicted vs. Observed'
        ax.set_title(title)

        legend = ax.legend()
        legend.set_draggable(True)

        ax.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                'k--', label='45-degree line')

        ax.text(0.05, 0.95,
                f'Training R2: {r2_train:.3f}\nTest R2: {r2_test:.3f}\nMSE: {mse_test:.3f}\nRMSE: {rmse_test:.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # Create a FigureCanvas object
        figure_canvas = FigureCanvas(fig)
        self.figure_canvas = figure_canvas  # Store the figure_canvas as an attribute

        # Create a dialog to display the plot
        dialog = QDialog(self)
        dialog.setWindowTitle("MLP Results")
        dialog.setGeometry(100, 100, 800, 600)
        dialog_layout = QVBoxLayout(dialog)

        # Add the FigureCanvas to the dialog layout
        dialog_layout.addWidget(figure_canvas)

        toolbar = NavigationToolbar(figure_canvas, dialog)


        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(toolbar)
        dialog_layout.addLayout(toolbar_layout)

        dialog_layout.addWidget(figure_canvas)

        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()
        self.current_fig = fig
        self.current_ax = ax
        self.current_dialog = dialog


    def setupKnnTab(self):
        layout = QVBoxLayout()
        guide_frame = QFrame()
        guide_layout = QVBoxLayout(guide_frame)

        guide_label = QLabel(
            "<h3>🧮 KNN (K-Nearest Neighbors) 모델 </h3>"
            "<p>KNN은 새로운 데이터 포인트가 주어졌을 때, "
            "기존 데이터 중 가장 가까운 K개의 이웃을 찾아 "
            "그들의 다수결(분류) 또는 평균(회귀)에 따라 예측하는 방법입니다.<br>"
            "모델이 데이터를 학습하지 않고, 예측 시점에 거리를 계산해 결과를 도출하는 "
            "‘Lazy Learning’ 방식입니다.</p>"
        )
        guide_label.setWordWrap(True)
        guide_layout.addWidget(guide_label)

        dimreduce_label = QLabel(
            "<h4>📉 차원 축소(Dimensionality Reduction) 기법 비교</h4>"
            "<ul>"
            "<li><b>PCA (Principal Component Analysis)</b>: <i>비지도 학습</i> 기반. "
            "데이터의 분산이 가장 큰 방향으로 축을 재정의하여 차원을 축소합니다.<br>"
            "‣ 데이터의 클래스 정보를 사용하지 않으며, 시각화나 노이즈 제거에 유용합니다.</li><br>"
            "<li><b>LDA (Linear Discriminant Analysis)</b>: <i>지도 학습</i> 기반. "
            "클래스 간 분리를 극대화하는 축을 찾아 차원을 축소합니다.<br>"
            "‣ 레이블이 있는 분류 문제에서 클래스 간 경계를 더 명확하게 시각화할 수 있습니다.</li><br>"
            "<li><b>NCA (Neighborhood Components Analysis)</b>: <i>지도 학습</i> 기반. "
            "KNN의 분류 성능을 최대화하도록 feature 공간을 학습합니다.<br>"
            "‣ LDA보다 유연하며, 비선형적 데이터 관계에서도 더 높은 성능을 낼 수 있습니다.</li>"
            "</ul>"
        )
        dimreduce_label.setWordWrap(True)
        dimreduce_label.setStyleSheet("font-size: 12px; color: #333;")
        guide_layout.addWidget(dimreduce_label)
        guide_frame.setFrameShape(QFrame.Box)
        guide_frame.setStyleSheet("background-color: #fafafa; border: 1px solid #ccc; padding: 8px;")
        layout.addWidget(guide_frame)

        groupBox = QFrame()
        groupBox.setFrameShape(QFrame.Box)
        groupBox.setFrameShadow(QFrame.Sunken)
        groupBoxLayout = QVBoxLayout(groupBox)

        label = QLabel("Select dimensionality reduction method:")
        groupBoxLayout.addWidget(label)

        self.pcaCheckBox = QCheckBox("PCA")
        self.ldaCheckBox = QCheckBox("LDA")
        self.ncaCheckBox = QCheckBox("NCA")
        self.noneCheckBox = QCheckBox("None (Binary data only)")
        self.ncaCheckBox.setChecked(True)

        self.dimensionalityGroup = QButtonGroup()
        for checkbox in [self.pcaCheckBox, self.ldaCheckBox, self.ncaCheckBox, self.noneCheckBox]:
            groupBoxLayout.addWidget(checkbox)
            checkbox.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.dimensionalityGroup.addButton(checkbox)
        self.dimensionalityGroup.setExclusive(True)

        groupBoxLayout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        layout.addWidget(groupBox)

        neighborsGroupBox = QFrame()
        neighborsGroupBox.setFrameShape(QFrame.Box)
        neighborsGroupBox.setFrameShadow(QFrame.Sunken)
        neighborsLayout = QVBoxLayout(neighborsGroupBox)

        n_neighbors_label = QLabel("Enter the number of neighbors:")
        neighborsLayout.addWidget(n_neighbors_label)

        n_neighbors_layout = QHBoxLayout()
        self.n_neighbors_input = QSpinBox()
        self.n_neighbors_input.setMinimum(2)
        self.n_neighbors_input.setValue(3)
        self.n_neighbors_input.setFixedWidth(50)
        n_neighbors_layout.addWidget(self.n_neighbors_input)

        tooltip_button = QPushButton("?")
        tooltip_button.setFixedSize(20, 20)
        tooltip_button.clicked.connect(self.showTooltip)
        tooltip_button.setToolTip(
            "Higher values may result in underfitting, while lower values may result in overfitting. Recommended value is 3.")
        n_neighbors_layout.addWidget(tooltip_button)

        neighborsLayout.addLayout(n_neighbors_layout)
        neighborsLayout.setSizeConstraint(QVBoxLayout.SetFixedSize)
        layout.addWidget(neighborsGroupBox)

        groupBox.setFixedHeight(neighborsGroupBox.sizeHint().height())
        neighborsGroupBox.setFixedHeight(neighborsGroupBox.sizeHint().height())

        buttons_layout = QHBoxLayout()

        self.createClassModelButton = QPushButton("Create Classification Model")
        self.createClassModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createClassModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createClassModelButton.clicked.connect(self.createClassificationModel)
        buttons_layout.addWidget(self.createClassModelButton)

        self.createRegModelButton = QPushButton("Create Regression Model")
        self.createRegModelButton.setFont(QFont('Arial', 12, QFont.Bold))
        self.createRegModelButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.createRegModelButton.clicked.connect(self.createRegressionModel)
        buttons_layout.addWidget(self.createRegModelButton)

        layout.addLayout(buttons_layout)

        # Right side layout for font settings and legend settings
        right_side_layout = QVBoxLayout()
        right_side_layout.setAlignment(Qt.AlignTop)
        right_side_layout.setSizeConstraint(QVBoxLayout.SetFixedSize)

        # Font settings
        font_settings_layout = QVBoxLayout()
        font_label = QLabel("Font settings:")
        font_settings_layout.addWidget(font_label)

        font_type_label = QLabel("Font type:")
        font_settings_layout.addWidget(font_type_label)

        self.fontTypeComboBox = QComboBox()
        self.fontTypeComboBox.addItems(["Arial", "Calibri", "Times New Roman", "Verdana"])
        font_settings_layout.addWidget(self.fontTypeComboBox)

        font_size_label = QLabel("Font size:")
        font_settings_layout.addWidget(font_size_label)

        self.fontSizeInput = QSpinBox()
        self.fontSizeInput.setRange(6, 32)
        self.fontSizeInput.setValue(12)
        font_settings_layout.addWidget(self.fontSizeInput)

        right_side_layout.addLayout(font_settings_layout)

        # Legend settings
        legend_settings_layout = QVBoxLayout()
        legend_label = QLabel("Legend settings:")
        legend_settings_layout.addWidget(legend_label)

        self.legendNameInput = QLineEdit()
        self.legendNameInput.setPlaceholderText("Enter legend name")
        legend_settings_layout.addWidget(self.legendNameInput)

        right_side_layout.addLayout(legend_settings_layout)

        # Legend checkbox
        self.legendCheckBox = QCheckBox("Show Legend")
        self.legendCheckBox.setChecked(True)
        right_side_layout.addWidget(self.legendCheckBox)

        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(right_side_layout)

        self.knnTab.setLayout(main_layout)
    def showTooltip(self):
        QMessageBox.information(self, "Number of Neighbors",
                                "Higher values may result in underfitting, while lower values may result in overfitting. Recommended value is 3.")

    def checkDataSplit(self):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        output_dir = os.path.join(base_dir, 'Temp')
        x_train_path = os.path.join(output_dir, 'X_train.csv')
        y_train_path = os.path.join(output_dir, 'y_train.csv')
        x_test_path = os.path.join(output_dir, 'X_test.csv')
        y_test_path = os.path.join(output_dir, 'y_test.csv')

        if not (os.path.exists(x_train_path) and os.path.exists(y_train_path) and os.path.exists(
                x_test_path) and os.path.exists(y_test_path)):
            QMessageBox.warning(self, "Data Split Error", "Please split the data first.")
            return False
        return True

    def createClassificationModel(self):
        if not self.checkDataSplit():
            return
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        n_neighbors = self.n_neighbors_input.value()

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        selected_method = self.getSelectedDimReductionMethod()
        if not selected_method:
            QMessageBox.warning(self, "Selection Error", "Please select a dimensionality reduction method.")
            return

        method_name, reducer = selected_method
        reducer.fit(X_train_numeric, y_train)
        X_train_embedded = reducer.transform(X_train_numeric)
        X_test_embedded = reducer.transform(X_test_numeric)

        knn.fit(X_train_embedded, y_train)
        accuracy = knn.score(X_test_embedded, y_test)

        y_pred_test = knn.predict(X_test_embedded)
        cm = confusion_matrix(y_test, y_pred_test)
        unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
        true = [f'true_{label}' for label in unique_labels]
        pred = [f'pred_{label}' for label in unique_labels]
        precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
            precision = np.nan_to_num(precision)  # Convert NaNs to zero

        cm_df = pd.DataFrame(cm, index=true, columns=pred)
        cm_df['Prediction Accuracy (%)'] = precision

        self.plotResults(method_name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors, accuracy)
        self.showConfusionMatrix(cm_df)
        self.models['KNN Classification'] = knn
        self.model_reducers['KNN Classification'] = reducer

    def createRegressionModel(self):
        if not self.checkDataSplit():
            return
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        n_neighbors = self.n_neighbors_input.value()

        knn = KNeighborsRegressor(n_neighbors=n_neighbors)
        selected_method = self.getSelectedDimReductionMethod()
        if not selected_method:
            QMessageBox.warning(self, "Selection Error", "Please select a dimensionality reduction method.")
            return

        method_name, reducer = selected_method
        reducer.fit(X_train_numeric, y_train)
        X_train_embedded = reducer.transform(X_train_numeric)
        X_test_embedded = reducer.transform(X_test_numeric)

        knn.fit(X_train_embedded, y_train)
        accuracy = knn.score(X_test_embedded, y_test)

        self.plotResults(method_name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors, accuracy)
        self.models['KNN Regression'] = knn
        if reducer:
            self.model_reducers['KNN Regression'] = reducer
    def plotResults(self, name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors, accuracy):

        plt.figure() #KNN 2차원 result scatter plot
        unique_labels_train = np.unique(y_train)
        unique_labels_test = np.unique(y_test)
        label_colors = {
            label: "#{:02x}{:02x}{:02x}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for label in np.unique(np.concatenate((y_train, y_test)))}

        legend_name = self.legendNameInput.text() if self.legendNameInput.text() else "Label"

        for label in np.unique(np.concatenate((y_train, y_test))):
            train_indices = np.where(y_train == label)[0]
            test_indices = np.where(y_test == label)[0]

            train_color = label_colors[label]
            test_color = label_colors[label]

            if len(train_indices) > 0:
                plt.scatter(X_train_embedded[train_indices, 0], X_train_embedded[train_indices, 1], c=train_color, s=30,
                            marker='o', alpha=0.5, label=f"Train {legend_name} {label}")
            if len(test_indices) > 0:
                plt.scatter(X_test_embedded[test_indices, 0], X_test_embedded[test_indices, 1], c=test_color, s=30,
                            marker='x', alpha=0.5, label=f"Test {legend_name} {label}")


        legend = plt.legend()
        legend.set_draggable(True)
        plt.title(f"{name} - KNN (k={n_neighbors})\nTest accuracy = {accuracy:.3f}",
                  fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.xlabel("Component 1", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.ylabel("Component 2", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.show()

        self.plotObservedVsPredicted(y_test, y_test, y_test, y_test, "KNN Observed vs Predicted")

    def plotObservedVsPredicted(self, y_train, y_pred_train, y_test, y_pred_test, title):
        if hasattr(self, 'observed_vs_predicted_dialog'):
            self.observed_vs_predicted_dialog.close()
        # 폰트 설정
        plt.rcParams['font.size'] = self.fontSizeInput.value()
        plt.rcParams['font.family'] = self.fontTypeComboBox.currentText()

        # 새로운 Figure 객체 생성
        fig, ax = plt.subplots(figsize=(10, 8))

        # 훈련 세트와 테스트 세트의 산점도
        scatter_train = ax.scatter(y_train, y_pred_train, c='blue', label='Training Set', marker='o', s=50, alpha=0.3)
        scatter_test = ax.scatter(y_test, y_pred_test, c='red', label='Test Set', marker='x', s=100, alpha=0.7)

        # 축 레이블과 제목 설정
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.set_title(title)

        # 범례 추가 및 드래그 가능하게 설정
        legend = ax.legend()
        legend.set_draggable(True)

        # 45도 선 추가
        ax.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                'k--', label='45-degree line')

        # R^2, MSE, RMSE 정보 추가
        ax.text(0.05, 0.95,
                f'Training R2: {r2_score(y_train, y_pred_train):.3f}\nTest R2: {r2_score(y_test, y_pred_test):.3f}\nMSE: {mean_squared_error(y_test, y_pred_test):.3f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # FigureCanvas 객체 생성 및 저장
        figure_canvas = FigureCanvas(fig)
        self.figure_canvas = figure_canvas  # figure_canvas를 인스턴스 속성으로 저장

        # 다이얼로그 생성하여 플롯 표시
        dialog = QDialog(self)
        dialog.setWindowTitle("Observed vs Predicted")
        dialog.setGeometry(100, 100, 800, 600)

        dialog_layout = QVBoxLayout(dialog)
        # 네비게이션 툴바 추가
        toolbar = NavigationToolbar(figure_canvas, dialog)
        dialog_layout.addWidget(toolbar)
        # FigureCanvas를 다이얼로그 레이아웃에 추가
        dialog_layout.addWidget(figure_canvas)



        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()
        # 다이얼로그를 인스턴스 속성으로 저장 (다음번에 닫기 위해)
        self.observed_vs_predicted_dialog = dialog

        # matplotlib의 현재 플롯을 닫아 중복 표시 방지
        plt.close(fig)

    def getSelectedDimReductionMethod(self):
        if self.pcaCheckBox.isChecked():
            return "PCA", PCA(n_components=2)
        elif self.ldaCheckBox.isChecked():
            return "LDA", LDA(n_components=2)
        elif self.ncaCheckBox.isChecked():
            return "NCA", NCA(n_components=2, max_iter=100, tol=1e-5, random_state=42)
        return None

    def showConfusionMatrix(self, cm_df):
        dialog = QDialog(self)
        dialog.setWindowTitle("Confusion Matrix")
        dialog.setGeometry(100, 100, 400, 300)
        dialog_layout = QVBoxLayout(dialog)

        table = QTableWidget(dialog)
        table.setRowCount(cm_df.shape[0])
        table.setColumnCount(cm_df.shape[1])
        table.setHorizontalHeaderLabels(cm_df.columns)
        table.setVerticalHeaderLabels(cm_df.index)

        for i in range(cm_df.shape[0]):
            for j in range(cm_df.shape[1]):
                if cm_df.columns[j] == 'Prediction Accuracy (%)':
                    item = QTableWidgetItem(f"{cm_df.iloc[i, j]:.3f}")
                else:
                    item = QTableWidgetItem(f"{cm_df.iloc[i, j]:.0f}")
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        dialog_layout.addWidget(table)
        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()

    def toggleRandomSelectOptions(self):
        self.randomSelectOptions.setVisible(
            self.rawDataRadioButton.isChecked() or self.scaledDataRadioButton.isChecked())

    def setupDataSplitTab(self):
        layout = QGridLayout()

        dataTypeGroupBox = QFrame()
        dataTypeGroupBox.setFrameShape(QFrame.Box)
        dataTypeGroupBox.setFrameShadow(QFrame.Sunken)
        dataTypeGroupBoxLayout = QVBoxLayout(dataTypeGroupBox)

        dataTypeLabel = QLabel("Select a data file:")
        dataTypeGroupBoxLayout.addWidget(dataTypeLabel)

        self.scaledDataRadioButton = QRadioButton("Use Scaled Data (most recent)")
        self.rawDataRadioButton = QRadioButton("Use Raw Data")
        self.scaledDataRadioButton.setChecked(True)


        dataTypeGroup = QButtonGroup()
        dataTypeGroup.addButton(self.rawDataRadioButton)
        dataTypeGroup.addButton(self.scaledDataRadioButton)

        dataTypeGroupBoxLayout.addWidget(self.rawDataRadioButton)
        dataTypeGroupBoxLayout.addWidget(self.scaledDataRadioButton)

        layout.addWidget(dataTypeGroupBox, 0, 0, 1, 1)

        self.randomSelectOptions = QWidget()
        self.randomSelectOptionsLayout = QVBoxLayout(self.randomSelectOptions)
        self.randomSelectLabel = QLabel("Random Select Options:")
        self.stratifyCheckBox = QCheckBox("Stratify")
        self.stratifyHelpLabel = QLabel(
            "→ Stratify는 각 클래스 비율이 학습/테스트 세트에 동일하게 유지되도록 데이터를 나눕니다.<br>"
            "   불균형 데이터셋에서 클래스 분포를 보존하려면 사용하세요."
        )
        self.stratifyHelpLabel.setStyleSheet("color: gray; font-size: 11px; margin-left: 20px;")
        self.testSetRatioLabel = QLabel("Enter the test set ratio (0-1):")
        self.testSetRatioInput = QLineEdit()
        self.testSetRatioInput.setValidator(QDoubleValidator(0.01, 0.99, 2))
        self.testSetRatioInput.setText("0.3")
        self.randomStateLabel = QLabel("Enter the random state:")
        self.randomStateInput = QLineEdit()
        self.randomStateInput.setValidator(QDoubleValidator(0, 9999, 0))
        self.randomStateInput.setText("0")
        self.randomStateHelpLabel = QLabel(
            "→ Random State 값이 같으면 매번 동일한 데이터가 테스트 세트로 선택됩니다.<br>"
            "   값을 바꾸면 데이터 분할이 달라집니다."
        )
        self.randomStateHelpLabel.setStyleSheet("color: gray; font-size: 11px; margin-left: 20px;")
        self.randomSelectOptionsLayout.addWidget(self.randomSelectLabel)
        self.randomSelectOptionsLayout.addWidget(self.stratifyCheckBox)
        self.randomSelectOptionsLayout.addWidget(self.stratifyHelpLabel)
        self.randomSelectOptionsLayout.addWidget(self.testSetRatioLabel)
        self.randomSelectOptionsLayout.addWidget(self.testSetRatioInput)
        self.randomSelectOptionsLayout.addWidget(self.randomStateLabel)
        self.randomSelectOptionsLayout.addWidget(self.randomStateInput)
        self.randomSelectOptionsLayout.addWidget(self.randomStateHelpLabel)


        layout.addWidget(self.randomSelectOptions, 1, 0, 1, 1)

        self.rawDataRadioButton.toggled.connect(self.toggleRandomSelectOptions)
        self.scaledDataRadioButton.toggled.connect(self.toggleRandomSelectOptions)

        self.splitDataButton = QPushButton("Split Data")
        self.splitDataButton.setFont(QFont('Arial', 14, QFont.Bold))
        self.splitDataButton.setStyleSheet(
            "QPushButton { padding: 10px; border-radius: 10px; border: 2px solid #000000; }")
        self.splitDataButton.clicked.connect(self.splitData)
        layout.addWidget(self.splitDataButton, 2, 0, 1, 2)

        self.trainingSetLabel = QLabel("Training Set:")
        layout.addWidget(self.trainingSetLabel, 3, 0, 1, 1, alignment=Qt.AlignBottom)

        self.trainSetWidget = QTableWidget()
        layout.addWidget(self.trainSetWidget, 4, 0, 1, 1)

        self.testSetLabel = QLabel("Test Set:")
        layout.addWidget(self.testSetLabel, 3, 1, 1, 1, alignment=Qt.AlignBottom)

        self.testSetWidget = QTableWidget()
        layout.addWidget(self.testSetWidget, 4, 1, 1, 1)

        self.dataSplitTab.setLayout(layout)


    def splitData(self):
        if self.rawDataRadioButton.isChecked():
            X_file = resource_path("Temp/original_X.csv")
        elif self.scaledDataRadioButton.isChecked():
            X_file = resource_path("Temp/scaled_X.csv")
        else:
            QMessageBox.warning(self, "Selection Error", "Please select either raw data or scaled data.")
            return

        if not os.path.exists(X_file):
            QMessageBox.warning(self, "File Error", f"{X_file} does not exist.")
            return

        X = pd.read_csv(X_file)
        y = pd.read_csv(resource_path("Temp/scaled_y.csv"))

        try:
            test_size = float(self.testSetRatioInput.text())
            random_state = int(self.randomStateInput.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error",
                                "Please enter valid numbers for test set ratio and random state.")
            return

        stratify = y if self.stratifyCheckBox.isChecked() else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=random_state, stratify=stratify)
        except ValueError as e:
            QMessageBox.warning(self, "Value Error", str(e))
            return


        output_dir = resource_path('Temp')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        X_train_numeric = X_train.drop(columns=['Sample'])
        X_test_numeric = X_test.drop(columns=['Sample'])

        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        X_test_numeric.to_csv(os.path.join(output_dir, 'X_test_numeric.csv'), index=False)
        X_train_numeric.to_csv(os.path.join(output_dir, 'X_train_numeric.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

        self.showSplitData(X_train, X_test, y_train, y_test)

        QMessageBox.information(self, "Information", "Training and Test sets have been saved.")

    def showSplitData(self, X_train, X_test, y_train, y_test):
        # Training Set
        self.trainSetWidget.clear()
        self.trainSetWidget.setRowCount(len(X_train))
        self.trainSetWidget.setColumnCount(X_train.shape[1] + 1)  # +1 for Label column

        # Column headers for training set
        train_headers = list(X_train.columns) + ['Label']
        self.trainSetWidget.setHorizontalHeaderLabels(train_headers)

        for i in range(len(X_train)):
            for j in range(X_train.shape[1]):
                value = X_train.iloc[i, j]
                # 숫자일 경우 소수점 4자리까지, 정수는 소수점 없이 표시
                if isinstance(value, float):
                    formatted_value = f"{int(value)}" if value.is_integer() else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                self.trainSetWidget.setItem(i, j, QTableWidgetItem(formatted_value))

            # y_train 처리 (값만 추출)
            y_value = y_train.values[i]
            if isinstance(y_value, float):
                formatted_value = f"{int(y_value)}" if y_value.is_integer() else f"{y_value:.4f}"
            else:
                formatted_value = str(y_value)
            self.trainSetWidget.setItem(i, X_train.shape[1], QTableWidgetItem(formatted_value))

        # Test Set
        self.testSetWidget.clear()
        self.testSetWidget.setRowCount(len(X_test))
        self.testSetWidget.setColumnCount(X_test.shape[1] + 1)  # +1 for Label column

        # Column headers for test set
        test_headers = list(X_test.columns) + ['Label']
        self.testSetWidget.setHorizontalHeaderLabels(test_headers)

        for i in range(len(X_test)):
            for j in range(X_test.shape[1]):
                value = X_test.iloc[i, j]
                # 숫자일 경우 소수점 4자리까지, 정수는 소수점 없이 표시
                if isinstance(value, float):
                    formatted_value = f"{int(value)}" if value.is_integer() else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                self.testSetWidget.setItem(i, j, QTableWidgetItem(formatted_value))

            # y_test 처리 (값만 추출)
            y_value = y_test.values[i]
            if isinstance(y_value, float):
                formatted_value = f"{int(y_value)}" if y_value.is_integer() else f"{y_value:.4f}"
            else:
                formatted_value = str(y_value)
            self.testSetWidget.setItem(i, X_test.shape[1], QTableWidgetItem(formatted_value))

    def loadCsv(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filename:
            try:
                self.csvViewer.loadCsv(filename)
                self.guideWidget.hide()
                self.csvViewer.show()
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Failed to load CSV: {e}")

    def exitApp(self):
        reply = QMessageBox.question(self, 'Message', '정말 닫으시겠습니까?', QMessageBox.Yes | QMessageBox.Cancel,
                                     QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            QCoreApplication.instance().quit()

    def show_scaled_data(self, scaled_X_df, y, headers):
        self.scaledDataWidget.clear()
        self.scaledDataWidget.setRowCount(len(scaled_X_df))
        self.scaledDataWidget.setColumnCount(len(headers) + 2)  # 샘플 이름과 타겟 포함
        self.scaledDataWidget.setHorizontalHeaderLabels(["Sample"] + headers + ["Label"])

        for i, row in scaled_X_df.iterrows():
            # 샘플 이름을 정수로 표시
            sample_value = row.iloc[0]
            if isinstance(sample_value, float) and sample_value.is_integer():
                formatted_sample = f"{int(sample_value)}"  # 정수로 변환하여 소수점 제거
            else:
                formatted_sample = str(sample_value)
            self.scaledDataWidget.setItem(i, 0, QTableWidgetItem(formatted_sample))

            # 피처
            for j, cell in enumerate(row[1:], start=1):
                if isinstance(cell, float):
                    formatted_value = f"{int(cell)}" if cell.is_integer() else f"{cell:.4f}"  # 정수인지 실수인지 구분
                elif isinstance(cell, int):
                    formatted_value = f"{cell}"  # 정수일 경우
                else:
                    formatted_value = str(cell)  # 그 외의 경우
                self.scaledDataWidget.setItem(i, j, QTableWidgetItem(formatted_value))

            # 타겟(Label)
            if isinstance(y[i], float):
                formatted_label = f"{int(y[i])}" if y[i].is_integer() else f"{y[i]:.4f}"  # 정수인지 실수인지 구분
            elif isinstance(y[i], int):
                formatted_label = f"{y[i]}"  # 정수일 경우
            else:
                formatted_label = str(y[i])  # 그 외의 경우
            self.scaledDataWidget.setItem(i, len(headers) + 1, QTableWidgetItem(formatted_label))  # 타겟

        self.tabs.setCurrentWidget(self.scaledDataTab)


class CsvViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.tableWidget = QTableWidget()
        layout.addWidget(self.tableWidget)
        self.setLayout(layout)
        self.X = None
        self.y = None
        self.original_data = None

    def loadCsv(self, filename):
        data = pd.read_csv(filename).dropna()
        headers = data.columns.tolist()

        dialog = ColumnRoleDialog(headers)
        if dialog.exec_():
            selections = dialog.getSelections()
            label_column_name = next(key for key, value in selections.items() if value == 'Label')
            sample_column_name = next((key for key, value in selections.items() if value == 'Sample'), None)
            feature_columns = [key for key, value in selections.items() if value == 'Feature']

            if sample_column_name is None:
                data['Sample'] = range(1, len(data) + 1)
                sample_column_name = 'Sample'

            if pd.api.types.is_numeric_dtype(data[label_column_name]):
                y = data[label_column_name].to_numpy()
            else:
                unique_labels = pd.unique(data[label_column_name])
                labelDialog = LabelMappingDialog(unique_labels)
                if labelDialog.exec_():
                    labelMappings = labelDialog.getLabelMappings()
                    if labelMappings:
                        data[label_column_name] = data[label_column_name].map(labelMappings).astype(float)
                        y = data[label_column_name].to_numpy()
                    else:
                        return  # If no mapping provided, exit

            for feature_column in feature_columns:
                data[feature_column] = pd.to_numeric(data[feature_column], errors='coerce')


                data = data.rename(columns={sample_column_name: 'Sample'})  # Change the sample column name to 'Sample'
                data = data.rename(columns={label_column_name: 'Label'})
                self.original_data = data[['Sample'] + feature_columns]  # 샘플 이름과 피처들
                self.X = data[feature_columns].to_numpy()
                self.y = y

                self.showCsvData(data.values.tolist(), data.columns.tolist())


    def getSampleNames(self):
        return self.original_data['Sample'].tolist()

    def showCsvData(self, data, headers):
        self.tableWidget.clear()
        self.tableWidget.setRowCount(len(data))
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)

        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                if isinstance(cell, float):
                    formatted_value = f"{int(cell)}" if cell.is_integer() else f"{cell:.4f}"
                elif isinstance(cell, int):
                    formatted_value = f"{cell}"
                else:
                    formatted_value = str(cell)
                self.tableWidget.setItem(i, j, QTableWidgetItem(formatted_value))


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


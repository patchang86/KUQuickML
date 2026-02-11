print("Running... Depending on your PC environment, this may take up to about 1 minute. Do not close this console window while the program is running.")
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
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
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
# Suppress scikit-learn feature-name mismatch warnings in console output.
# The app already blocks loading if feature names/order do not match, so this warning only adds noise.
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but .* was fitted without feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but NeighborhoodComponentsAnalysis was fitted with feature names",
    category=UserWarning,
)
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
from sklearn.base import clone


def resource_path(relative_path):
    """ Function that returns the file path when packaging with PyInstaller """
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
        self.previousModelPredictionTab = QWidget()  # comment

        self.tabs.addTab(self.mainTab, "Main")
        self.tabs.addTab(self.scaledDataTab, "Scaled Data")

        self.tabs.addTab(self.predictionTab, "Prediction")

        self.setupMainTab()
        self.setupScaledDataTab()
        self.setupPredictionTab()
        self.setupPreviousModelPredictionTab()  # comment

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
        datasplitAction.triggered.connect(self.addDataSplitTab)  # comment
        datasplitMenu.addAction(datasplitAction)
        menubar.addMenu(datasplitMenu)

        algorithmMenu = QMenu('4. Algorithm', self)
        knnAction = QAction('KNN', self)
        knnAction.triggered.connect(self.addKnnTab)
        algorithmMenu.addAction(knnAction)
        mlpAction = QAction('Multi-Layer Perceptron', self)
        mlpAction.triggered.connect(self.addMLPTab)  # comment
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

    def _drop_sample_and_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove the Sample column + force everything to numeric (errors become NaN)"""
        if 'Sample' in df.columns:
            df = df.drop(columns=['Sample'])
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

    def _fit_preprocess_train_test(self, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, y_train):
        """
        Preprocessing for training (fit scaler/reducer on the training data)
        Returns: X_train_used, X_test_used, fitted_scaler, fitted_reducer, feature_names
        """
        # comment
        feature_names = list(X_train_df.columns)

        # comment
        base_scaler = self.scaler
        scaler = clone(base_scaler) if base_scaler else None

        if scaler:
            X_train_scaled = scaler.fit_transform(X_train_df)
            X_test_scaled = scaler.transform(X_test_df)
        else:
            X_train_scaled = X_train_df.values
            X_test_scaled = X_test_df.values

        # comment
        reducer = None
        selected = self.getSelectedDimReductionMethod()
        if selected:
            _, reducer = selected
            reducer.fit(np.asarray(X_train_scaled), y_train)
            X_train_used = reducer.transform(np.asarray(X_train_scaled))
            X_test_used = reducer.transform(np.asarray(X_test_scaled))
        else:
            X_train_used = X_train_scaled
            X_test_used = X_test_scaled

        return X_train_used, X_test_used, scaler, reducer, feature_names

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

            # comment
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
        # comment
        layout = QVBoxLayout()
        someLabel = QLabel("This is the Previous Model Prediction Tab")
        layout.addWidget(someLabel)
        self.previousModelPredictionTab.setLayout(layout)

    def saveModel(self, model_name, filename):
        bundle = self.models.get(model_name)
        if not isinstance(bundle, dict) or "model" not in bundle:
            QMessageBox.warning(self, "Error", "Selected item is not a valid saved model bundle.")
            return
        try:
            joblib.dump(bundle, filename)
            scaler = bundle.get("scaler")
            reducer = bundle.get("reducer")
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
        model_choice, ok = QInputDialog.getItem(
            self, "Select Model to Save",
            "Choose a model to save:",
            list(self.models.keys()), 0, False
        )
        if ok and model_choice:
            options = QFileDialog.Options()
            suggested_name = f"{model_choice}"
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Model",
                suggested_name,
                "Joblib Files (*.joblib);;All Files (*)",
                options=options
            )
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

        # comment
        svm_overview = QLabel(
            "<h3>⚙️ Support Vector Machine (SVM)</h3>"
            "<p>SVM is an algorithm that finds a hyperplane that best separates classes (or fits a regression function).<br>"
            "With kernel functions, it can also separate nonlinear data in a higher-dimensional space.</p>"
        )
        svm_overview.setWordWrap(True)
        layout.addWidget(svm_overview)

        # comment
        groupBox = QFrame()
        groupBox.setFrameShape(QFrame.Box)
        groupBox.setFrameShadow(QFrame.Sunken)
        groupBoxLayout = QVBoxLayout(groupBox)
        paramGrid = QGridLayout()
        paramGrid.setHorizontalSpacing(10)
        paramGrid.setVerticalSpacing(10)
        groupBoxLayout.addLayout(paramGrid)


        # SVM Type
        frame_type = QFrame()
        frame_type.setFrameShape(QFrame.Box)
        frame_type.setFrameShadow(QFrame.Sunken)
        frame_type_layout = QVBoxLayout(frame_type)
        label = QLabel("Select SVM Type:")
        self.svm_type = QComboBox()
        self.svm_type.addItems(["One-vs-Rest SVM", "One-vs-One SVM"])
        desc = QLabel("Select the SVM classification strategy.<br>"
                      "<b>One-vs-Rest</b>: Compare one class vs. all others (faster)<br>"
                      "<b>One-vs-One</b>: Train all pairwise class combinations (often higher accuracy)")
        desc.setStyleSheet(desc_style)
        frame_type_layout.addWidget(label)
        frame_type_layout.addWidget(self.svm_type)
        frame_type_layout.addWidget(desc)
        paramGrid.addWidget(frame_type, 0, 0)

        # Kernel Type
        frame_kernel = QFrame()
        frame_kernel.setFrameShape(QFrame.Box)
        frame_kernel.setFrameShadow(QFrame.Sunken)
        frame_kernel_layout = QVBoxLayout(frame_kernel)
        label = QLabel("Select Kernel Type:")
        self.kernel_type = QComboBox()
        self.kernel_type.addItems(["linear", "poly", "rbf", "sigmoid"])
        desc = QLabel("A kernel maps input data into a higher-dimensional space.<br>"
                      "<b>linear</b>: Linear boundary, fast<br>"
                      "<b>poly</b>: Polynomial kernel<br>"
                      "<b>rbf</b>: Gaussian-based, strong for nonlinear patterns<br>"
                      "<b>sigmoid</b>: Neural-network-like behavior")
        desc.setStyleSheet(desc_style)
        frame_kernel_layout.addWidget(label)
        frame_kernel_layout.addWidget(self.kernel_type)
        frame_kernel_layout.addWidget(desc)
        paramGrid.addWidget(frame_kernel, 0, 1)

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
        desc = QLabel("C controls the regularization strength (how much error is tolerated).<br>"
                      "Smaller C → better generalization (more stable), larger C → higher training accuracy (risk of overfitting).")
        desc.setStyleSheet(desc_style)
        frame_c_layout.addWidget(label)
        frame_c_layout.addWidget(self.c_value)
        frame_c_layout.addWidget(desc)
        paramGrid.addWidget(frame_c, 1, 0)

        # Random State
        frame_random = QFrame()
        frame_random.setFrameShape(QFrame.Box)
        frame_random.setFrameShadow(QFrame.Sunken)
        frame_random_layout = QVBoxLayout(frame_random)
        label = QLabel("Random State:")
        self.random_state_input = QSpinBox()
        self.random_state_input.setRange(0, 999999)
        self.random_state_input.setValue(42)
        desc = QLabel("Seed value to control randomness.<br>Use the same value to reproduce identical results.")
        desc.setStyleSheet(desc_style)
        frame_random_layout.addWidget(label)
        frame_random_layout.addWidget(self.random_state_input)
        frame_random_layout.addWidget(desc)
        paramGrid.addWidget(frame_random, 1, 1)

        # --- SVR (Regression) Parameters ---
        frame_svr = QFrame()
        frame_svr.setFrameShape(QFrame.Box)
        frame_svr.setFrameShadow(QFrame.Sunken)
        frame_svr_layout = QVBoxLayout(frame_svr)
        title = QLabel("<b>SVR (Regression) Parameters</b>")
        frame_svr_layout.addWidget(title)

        label = QLabel("SVR Epsilon:")
        self.svrEpsilonInput = QDoubleSpinBox()
        self.svrEpsilonInput.setRange(0.0, 10.0)
        self.svrEpsilonInput.setValue(0.1)
        self.svrEpsilonInput.setSingleStep(0.01)
        desc = QLabel("Width of the epsilon-insensitive tube. Smaller values can make the model more sensitive to data.")
        desc.setStyleSheet(desc_style)
        frame_svr_layout.addWidget(label)
        frame_svr_layout.addWidget(self.svrEpsilonInput)
        frame_svr_layout.addWidget(desc)

        paramGrid.addWidget(frame_svr, 2, 0, 1, 2)

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

        self.pcaCheckBox.setChecked(True)  # comment
        self.dimensionalityGroup = QButtonGroup()
        for checkbox in [self.pcaCheckBox, self.ldaCheckBox, self.ncaCheckBox, self.noneCheckBox]:
            frame_reducer_layout.addWidget(checkbox)
            self.dimensionalityGroup.addButton(checkbox)
        self.dimensionalityGroup.setExclusive(True)
        desc = QLabel("Dimensionality reduction projects data into a lower-dimensional space to improve efficiency and visualization.<br>"
                      "<b>PCA</b>: Principal Component Analysis (common)<br>"
                      "<b>LDA</b>: Optimizes class separation<br>"
                      "<b>NCA</b>: Suitable for distance-based classification<br>"
                      "<b>None</b>: No dimensionality reduction")
        desc.setStyleSheet(desc_style)
        frame_reducer_layout.addWidget(desc)
        groupBoxLayout.addWidget(frame_reducer)

        groupBoxLayout.setContentsMargins(5, 5, 5, 5)
        groupBoxLayout.setSpacing(15)
        layout.addWidget(groupBox)

        # comment
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

        # 5-fold CV buttons
        self.svmCvClassButton = QPushButton("5-Fold CV (SVM Classification)")
        self.svmCvClassButton.clicked.connect(self.runSVMClassificationCV)
        buttons_layout.addWidget(self.svmCvClassButton)

        self.svmCvRegButton = QPushButton("5-Fold CV (SVM Regression)")
        self.svmCvRegButton.clicked.connect(self.runSVMRegressionCV)
        buttons_layout.addWidget(self.svmCvRegButton)
        layout.addLayout(buttons_layout)

        # comment
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

        # comment
        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(right_side_layout)

        self.SVMTab.setLayout(main_layout)
        self.tabs.setCurrentWidget(self.SVMTab)

    def createSVMModel(self):
        loading = QProgressDialog(
            "Creating the model...\n\nDepending on your computer, this may take a few minutes.",
            None, 0, 0, self
        )
        loading.setWindowTitle("Creating SVM model")
        loading.setWindowModality(Qt.ApplicationModal)
        loading.setMinimumWidth(420)
        loading.setAutoClose(False)
        loading.setAutoReset(False)
        loading.show()
        QApplication.processEvents()

        if not self.checkDataSplit():
            loading.close()
            return

        try:
            X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
            X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
            y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
            y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

            X_train_numeric = self._drop_sample_and_numeric(X_train)
            X_test_numeric = self._drop_sample_and_numeric(X_test)

            feature_names = list(X_train_numeric.columns)

            # comment
            selected = self.getSelectedDimReductionMethod()
            reducer = None
            if selected:
                _, reducer = selected
                reducer.fit(X_train_numeric.values, y_train)
                X_train_used = reducer.transform(X_train_numeric.values)
                X_test_used = reducer.transform(X_test_numeric.values)
            else:
                X_train_used = X_train_numeric.values
                X_test_used = X_test_numeric.values

            kernel = self.kernel_type.currentText()
            c_value = self.c_value.value()

            if self.svm_type.currentText() == "One-vs-One SVM":
                svc_model = OneVsOneClassifier(SVC(kernel=kernel, C=c_value))
            else:
                svc_model = OneVsRestClassifier(SVC(kernel=kernel, C=c_value))

            svc_model.fit(X_train_used, y_train)
            y_pred_train = svc_model.predict(X_train_used)
            y_pred_test = svc_model.predict(X_test_used)

            cm = confusion_matrix(y_test, y_pred_test)
            with np.errstate(divide='ignore', invalid='ignore'):
                precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
                precision = np.nan_to_num(precision)

            labels = [f"Class {i}" for i in range(cm.shape[0])]
            cm_df = pd.DataFrame(
                cm,
                index=[f"Actual {label}" for label in labels],
                columns=[f"Predicted {label}" for label in labels]
            )
            cm_df["Prediction Accuracy (%)"] = precision
            self.showConfusionMatrix(cm_df)

            overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
            self.plotScatterWithDecisionBoundary(
                X_train_used, y_train, X_test_used, y_pred_test, svc_model,
                f"SVM Scatter Plot with Decision Boundary (kernel={kernel})\nTest accuracy = {overall_accuracy:.3f}"
            )

            # comment
            if kernel == "linear" and reducer is None and hasattr(svc_model, "coef_"):
                self.showImportantCoefficients(X_train_numeric, svc_model)
            else:
                # comment
                self.showSVMImportanceUnavailable(
                    kernel=kernel,
                    reducer=reducer,
                    X_test=X_test_numeric,
                    y_test=y_test,
                    model=svc_model,
                    feature_names=feature_names,
                    title_prefix="SVM (Classification)",
                    task="classification"
                )

            self.plotObservedVsPredicted(
                y_train, y_pred_train, y_test, y_pred_test,
                f"SVC Observed vs Predicted (kernel={kernel})"
            )

            # comment
            self.models["SV Classification"] = {
                "model": svc_model,
                "scaler": self._get_bundle_scaler(),
                "reducer": reducer,
                "feature_names": feature_names,
                "label_mapping": self._get_label_mapping()
            }
            if reducer:
                self.model_reducers["SV Classification"] = reducer

        except Exception as e:
            QMessageBox.warning(self, "SVM Error", f"Failed to create SVM model:\n{e}")
        finally:
            QTimer.singleShot(200, loading.close)

    def createSVMRegressionModel(self):
        if not self.checkDataSplit():
            return

        try:
            X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
            X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
            y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
            y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

            X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
            X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)

            feature_names = list(X_train_numeric.columns)

            # comment
            reducer = None
            if self.pcaCheckBox.isChecked():
                reducer = PCA(n_components=2)
                reducer.fit(X_train_numeric.values)
                X_train_used = reducer.transform(X_train_numeric.values)
                X_test_used = reducer.transform(X_test_numeric.values)
            else:
                # comment
                X_train_used = X_train_numeric.values
                X_test_used = X_test_numeric.values

            kernel = self.kernel_type.currentText()
            c_value = float(self.c_value.value())
            epsilon = float(self.svrEpsilonInput.value())

            svr_model = SVR(kernel=kernel, C=c_value, epsilon=epsilon)
            svr_model.fit(X_train_used, y_train)

            y_pred_train = svr_model.predict(X_train_used)
            y_pred_test = svr_model.predict(X_test_used)

            r2_test = r2_score(y_test, y_pred_test)
            mse_test = mean_squared_error(y_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)

            if kernel == "linear" and reducer is None and hasattr(svr_model, "coef_"):
                self.showImportantCoefficients(X_train_numeric, svr_model)
            else:
                # comment
                self.showSVMImportanceUnavailable(
                    kernel=kernel,
                    reducer=reducer,
                    X_test=X_test_numeric,
                    y_test=y_test,
                    model=svr_model,
                    feature_names=feature_names,
                    title_prefix="SVR (Regression)",
                    task="regression"
                )


            # comment
            if reducer is not None and hasattr(X_train_used, "shape") and X_train_used.shape[1] == 2:
                self.plotScatterWithRegressionSurface(
                    X_train_used, y_train,
                    X_test_used, y_test,
                    svr_model,
                    f"SVR Regression Surface (kernel={kernel})"
                )

            self.plotObservedVsPredicted(
                y_train, y_pred_train, y_test, y_pred_test,
                f"SVR Observed vs Predicted (kernel={kernel})\nTest R2={r2_test:.3f}, RMSE={rmse_test:.3f}"
            )

            self.models["SV Regression"] = {
                "model": svr_model,
                "scaler": self._get_bundle_scaler(),
                "reducer": reducer,
                "feature_names": feature_names,
                "label_mapping": None
            }
            if reducer:
                self.model_reducers["SV Regression"] = reducer

        except Exception as e:
            QMessageBox.warning(self, "SVR Error", f"Failed to create SVR model:\n{e}")

    # ------------------------------
    # 5-fold CV runners (SVM)
    # ------------------------------
    def runSVMClassificationCV(self):
        try:
            # comment
            svm_type_text = self.svm_type.currentText()
            kernel = self.kernel_type.currentText()
            c_value = float(self.c_value.value())

            base_svc = SVC(C=c_value, kernel=kernel, probability=True, decision_function_shape="ovr")

            if "One-vs-Rest" in svm_type_text:
                estimator = OneVsRestClassifier(base_svc)
            elif "One-vs-One" in svm_type_text:
                # Use native SVC multiclass (internally OvO) but keep probability outputs for ROC-AUC
                estimator = SVC(C=c_value, kernel=kernel, probability=True, decision_function_shape="ovo")
            else:
                estimator = base_svc

            self.run_5fold_cv(estimator, task="classification")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))

    def runSVMRegressionCV(self):
        try:
            kernel = self.kernel_type.currentText()
            c_value = self.c_value.value()
            epsilon = self.svrEpsilonInput.value()
            estimator = SVR(C=c_value, kernel=kernel, epsilon=epsilon)
            self.run_5fold_cv(estimator, task="regression")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))



    def plotScatterWithDecisionBoundary(self, X_train_reduced, y_train, X_test_reduced, y_pred_test, model, title):
        """Visualize the classification decision boundary in a 2D reduced space.
        - binary: use decision_function (or proba) for a smoother boundary
        - multiclass: visualize by predicted class on a grid
        """
        plt.figure(figsize=(10, 8))

        x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
        y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1

        # comment
        grid_res = 400
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        # comment
        unique_labels = np.unique(np.concatenate((y_train, y_pred_test)))
        is_multiclass = len(unique_labels) > 2

        if (not is_multiclass) and hasattr(model, "decision_function"):
            # comment
            score = model.decision_function(grid).reshape(xx.shape)
            plt.contourf(xx, yy, score, levels=60, alpha=0.30, cmap="viridis")
            # comment
            try:
                plt.contour(xx, yy, score, levels=[0], colors="k", linewidths=1.0)
            except Exception:
                pass
        elif (not is_multiclass) and hasattr(model, "predict_proba"):
            proba = model.predict_proba(grid)
            # comment
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                p1 = proba[:, 1].reshape(xx.shape)
                plt.contourf(xx, yy, p1, levels=60, alpha=0.30, cmap="viridis")
                try:
                    plt.contour(xx, yy, p1, levels=[0.5], colors="k", linewidths=1.0)
                except Exception:
                    pass
            else:
                Z = model.predict(grid).reshape(xx.shape)
                plt.contourf(xx, yy, Z, alpha=0.30, cmap="viridis")
        else:
            # multiclass (or fallback): class label map
            Z = model.predict(grid).reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.30, cmap="viridis")

        # comment
        label_colors = {
            label: "#{:02x}{:02x}{:02x}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for label in unique_labels
        }

        legend_widget = getattr(self, "legendNameInput", None)
        legend_name = legend_widget.text() if (legend_widget is not None and legend_widget.text()) else "Label"

        # Train/Test scatter
        for label in unique_labels:
            train_indices = np.where(y_train == label)[0]
            test_indices = np.where(y_pred_test == label)[0]  # comment

            color = label_colors[label]

            if len(train_indices) > 0:
                plt.scatter(
                    X_train_reduced[train_indices, 0], X_train_reduced[train_indices, 1],
                    c=color, s=30, marker="o", alpha=0.55,
                    label=f"Train {legend_name} {label}"
                )
            if len(test_indices) > 0:
                plt.scatter(
                    X_test_reduced[test_indices, 0], X_test_reduced[test_indices, 1],
                    c=color, s=30, marker="x", alpha=0.85,
                    label=f"Test {legend_name} {label}"
                )

        legend = plt.legend()
        legend.set_draggable(True)
        plt.title(title, fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.xlabel("Component 1", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.ylabel("Component 2", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.show()



    def plotScatterWithRegressionSurface(self, X_train_reduced, y_train, X_test_reduced, y_test, model, title):
        """Display regression predictions in a 2D reduced space in a form similar to a classification boundary plot.
        - background: fill with contourf of grid predictions
        - points: Train/Test samples
        """
        plt.figure(figsize=(10, 8))

        x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
        y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1

        grid_res = 400
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        cf = plt.contourf(xx, yy, Z, levels=60, alpha=0.30, cmap="viridis")
        cb = plt.colorbar(cf)
        cb.set_label("Predicted value (background)")

        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)
        y_all = np.concatenate([y_train_arr, y_test_arr])
        unique_vals = np.unique(y_all)
        max_unique_direct = 15

        def _is_integer_like(arr):
            if arr.size == 0:
                return False
            return np.all(np.isfinite(arr)) and np.all(np.abs(arr - np.round(arr)) < 1e-9)

        use_direct = (unique_vals.size <= max_unique_direct) or _is_integer_like(unique_vals)

        if use_direct:
            categories = [str(v) for v in unique_vals]
            val_to_cat = {v: str(v) for v in unique_vals}
            y_train_cat = np.array([val_to_cat[v] for v in y_train_arr], dtype=object)
            y_test_cat = np.array([val_to_cat[v] for v in y_test_arr], dtype=object)
        else:
            n_bins = min(10, max(3, int(np.sqrt(len(y_all)))))
            try:
                bins = pd.qcut(y_all, q=n_bins, duplicates="drop")
            except Exception:
                bins = pd.cut(y_all, bins=n_bins, duplicates="drop")

            bin_str = bins.astype(str)
            y_train_cat = bin_str[:len(y_train_arr)]
            y_test_cat = bin_str[len(y_train_arr):]
            categories = list(pd.unique(bin_str))

        label_colors = {
            cat: "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for cat in categories
        }

        for cat in categories:
            train_idx = np.where(y_train_cat == cat)[0]
            test_idx = np.where(y_test_cat == cat)[0]
            color = label_colors[cat]

            if len(train_idx) > 0:
                plt.scatter(
                    X_train_reduced[train_idx, 0], X_train_reduced[train_idx, 1],
                    c=color, s=30, marker='o', alpha=0.55,
                    label=f"Train label {cat}"
                )
            if len(test_idx) > 0:
                plt.scatter(
                    X_test_reduced[test_idx, 0], X_test_reduced[test_idx, 1],
                    c=color, s=70, marker='x', alpha=0.90,
                    label=f"Test label {cat}"
                )

        legend = plt.legend()
        legend.set_draggable(True)

        plt.title(title, fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.xlabel("Component 1", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.ylabel("Component 2", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
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

        info = QLabel(
            "Feature importance is shown using the coefficients (coef_) of a linear SVM/SVR.\n"
            "A larger absolute value means a stronger influence, and the sign (+/−) indicates the direction (increase/decrease) of the prediction."
        )
        info.setWordWrap(True)
        dialog_layout.addWidget(info)

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


    def showSVMImportanceUnavailable(self, kernel, reducer, X_test, y_test, model, feature_names, title_prefix="SVM", task="classification"):
        """If feature importance cannot be displayed via coef_, show the reason and
        compute permutation importance when possible.

        - X_test is expected to be a DataFrame in the *original feature space* (before any reduction/transform).
        - If a reducer exists, permutation importance still permutes the *original features*,
          but prediction is passed through reducer.transform before reaching the trained model.
        """

        class _ReducerWrappedEstimator:
            """Wrapper for permutation_importance: original X -> (reducer) -> model.predict"""
            def __init__(self, fitted_model, fitted_reducer=None):
                self._model = fitted_model
                self._reducer = fitted_reducer

            def _to_array(self, X):
                try:
                    return X.values  # DataFrame
                except Exception:
                    return X

            def predict(self, X):
                Xv = self._to_array(X)
                if self._reducer is not None:
                    Xv = self._reducer.transform(Xv)
                return self._model.predict(Xv)

            def decision_function(self, X):
                if hasattr(self._model, "decision_function"):
                    Xv = self._to_array(X)
                    if self._reducer is not None:
                        Xv = self._reducer.transform(Xv)
                    return self._model.decision_function(Xv)
                raise AttributeError("Wrapped model has no decision_function")

            def predict_proba(self, X):
                if hasattr(self._model, "predict_proba"):
                    Xv = self._to_array(X)
                    if self._reducer is not None:
                        Xv = self._reducer.transform(Xv)
                    return self._model.predict_proba(Xv)
                raise AttributeError("Wrapped model has no predict_proba")


            @property
            def classes_(self):
                return getattr(self._model, "classes_", None)

            def fit(self, X, y=None):
                """Dummy fit for sklearn scorer compatibility (estimator is already fitted)."""
                return self

            def get_params(self, deep=True):
                # For sklearn's inspection/scoring utilities
                return {"fitted_model": self._model, "fitted_reducer": self._reducer}

            def set_params(self, **params):
                # Minimal setter for sklearn compatibility
                if "fitted_model" in params:
                    self._model = params["fitted_model"]
                if "fitted_reducer" in params:
                    self._reducer = params["fitted_reducer"]
                return self

        reasons = []
        if kernel != "linear":
            reasons.append("The selected kernel is nonlinear, so coefficients (coef_) for input features are not defined.")
        if reducer is not None:
            reasons.append("Dimensionality reduction transformed the original features, so coef_-based importance cannot be computed on the original feature basis.")

        reason_text = "\n".join(f"- {r}" for r in reasons) if reasons else "- Cannot compute coef_-based importance."

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Feature Importance")
        msg_box.setText(
            f"{title_prefix} cannot display feature importance using coef_.\n\n"
            f"{reason_text}\n\n"
            "Instead, permutation importance (how much the score drops when a feature is shuffled) can be computed and displayed.\n"
            "(Permutation importance can be computationally expensive.)"
        )

        perm_btn = msg_box.addButton("Compute permutation importance", QMessageBox.ActionRole)
        close_btn = msg_box.addButton("Close", QMessageBox.RejectRole)
        msg_box.setDefaultButton(close_btn)
        msg_box.exec_()

        if msg_box.clickedButton() != perm_btn:
            return

        if not feature_names:
            try:
                feature_names = list(X_test.columns)
            except Exception:
                feature_names = [f"Feature {i+1}" for i in range(X_test.shape[1])]

        try:
            wrapped = _ReducerWrappedEstimator(model, reducer)
            scoring = "accuracy" if task == "classification" else "r2"

            perm = permutation_importance(
                wrapped,
                X_test,  # comment
                y_test,
                n_repeats=10,
                random_state=42,
                scoring=scoring
            )
            sorted_idx = np.argsort(perm.importances_mean)[::-1]
            feature_importances = [(feature_names[i], float(perm.importances_mean[i])) for i in sorted_idx]
            self.showMLPFeatureImportances(feature_importances)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Permutation Importance Error",
                f"Failed to compute permutation importance:\n{e}"
            )


    def setupMainTab(self):
        self.mainLayout = QVBoxLayout()

        # comment
        self.guideWidget = QWidget()
        guideLayout = QVBoxLayout(self.guideWidget)

        guideTitle = QLabel("Welcome to KUQuickML")
        guideTitle.setFont(QFont('Arial', 18, QFont.Bold))
        guideTitle.setAlignment(Qt.AlignCenter)
        guideLayout.addWidget(guideTitle)

        guideText = QLabel(
            "This program is a GUI tool for beginners in machine learning.\n\n"
            "① Load a CSV file. (Each column contains feature values (x), sample names, and label values (y).)\n"
            "  If label values are not numeric, they will be mapped to arbitrary numbers (e.g., 0, 1, 2).\n"
            "② Perform data scaling.\n"
            "③ Split the data into Train/Test sets.\n"
            "④ Choose an algorithm (KNN, MLP, RF, SVM) and train a model.\n"
            "⑤ Save the model or predict unknown samples.\n"
            "See the example at the bottom for the required CSV format.\n"
        )
        guideText.setAlignment(Qt.AlignLeft)
        guideText.setWordWrap(True)
        guideLayout.addWidget(guideText)

        # comment
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

        # comment
        self.csvViewer = CsvViewer()
        self.csvViewer.hide()  # comment
        self.mainLayout.addWidget(self.csvViewer)

        self.mainTab.setLayout(self.mainLayout)

    def setupScaledDataTab(self):
        layout = QVBoxLayout()
        guide_frame = QFrame()
        guide_layout = QVBoxLayout(guide_frame)
        guide_label = QLabel(
            "<h3>📊 Data Scaling Guide</h3>"
            "<p>Scaling adjusts the value ranges of features to improve model training performance.<br>"
            "Choose an appropriate scaler depending on your data characteristics and model type.</p>"
            "<ul>"
            "<li><b>StandardScaler</b>: Normalize to mean 0 and standard deviation 1. A common default for many ML models.<br>"
            "‣ Pros: Effective for normally distributed data.<br>"
            "‣ Cons: Sensitive to outliers.</li><br>"
            "<li><b>MinMaxScaler</b>: Scale to the [0, 1] range.<br>"
            "‣ Pros: Helps fast convergence for neural networks, etc.<br>"
            "‣ Cons: Very sensitive to outliers.</li><br>"
            "<li><b>RobustScaler</b>: Transform using the median and IQR (interquartile range).<br>"
            "‣ Pros: Stable when there are many outliers.<br>"
            "‣ Cons: If the distribution is close to normal, precision may decrease.</li><br>"
            "<li><b>MaxAbsScaler</b>: Scale each feature by its maximum absolute value (so max abs becomes 1).<br>"
            "‣ Pros: Preserves sparse matrices (sparse data).<br>"
            "‣ Cons: Not suitable when positive/negative proportions vary widely.</li><br>"
            "<li><b>Normalizer</b>: Scale each sample vector to length 1.<br>"
            "‣ Pros: Good for text vectors or distance-based models (KNN).<br>"
            "‣ Cons: Does not correct the distribution across features.</li>"
            "</ul>"
        )
        guide_label.setWordWrap(True)
        guide_layout.addWidget(guide_label)
        guide_frame.setFrameShape(QFrame.Box)
        guide_frame.setStyleSheet("background-color: #fafafa; padding: 8px; border: 1px solid #ccc;")

        layout.addWidget(guide_frame)

        # comment
        self.scalerStatusLabel = QLabel("Current Scaling Method: None")
        self.scalerStatusLabel.setStyleSheet("font-weight: bold; color: darkgreen;")
        layout.addWidget(self.scalerStatusLabel)

        self.scaledDataWidget = QTableWidget()
        layout.addWidget(self.scaledDataWidget)
        self.scaledDataTab.setLayout(layout)

    def setupMLPTab(self):
        layout = QVBoxLayout()
        desc_style = "color: #555; font-size: 10pt; margin-bottom: 4px;"

        # comment
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

        # comment
        left_col = QVBoxLayout()

        # comment
        hidden_layer_label = QLabel("Hidden Layer Size (comma separated):")
        self.hidden_layer_input = QLineEdit()
        self.hidden_layer_input.setPlaceholderText("50,50")
        self.hidden_layer_input.setFixedWidth(300)
        desc = QLabel(
            "Specify the hidden-layer architecture. <br>(e.g., 100,50,30 → three hidden layers with 100, 50, and 30 neurons). <br>More neurons/layers can learn complex patterns but increase overfitting risk.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([hidden_layer_label, self.hidden_layer_input, desc]))

        # comment
        alpha_label = QLabel("Alpha (Regularization strength):")
        self.alpha_input = QLineEdit()
        self.alpha_input.setPlaceholderText("0.0001")
        self.alpha_input.setFixedWidth(300)
        desc = QLabel("Limits the magnitude of weights to prevent overfitting. <br>Larger values make the model simpler; smaller values make it more complex.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([alpha_label, self.alpha_input, desc]))

        # Max Iteration
        max_iter_label = QLabel("Max Iterations:")
        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 999999)
        self.max_iter_input.setValue(1000)
        self.max_iter_input.setFixedWidth(300)
        desc = QLabel("Maximum number of training iterations. Increase this if the model does not converge.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([max_iter_label, self.max_iter_input, desc]))

        # Random State
        random_state_label = QLabel("Random State:")
        self.random_state_input = QSpinBox()
        self.random_state_input.setRange(0, 999999)
        self.random_state_input.setValue(42)
        self.random_state_input.setFixedWidth(300)
        desc = QLabel("Fix the random initialization seed. Keep the same value to reproduce results.")
        desc.setStyleSheet(desc_style)
        left_col.addWidget(wrap_in_box([random_state_label, self.random_state_input, desc]))

        # comment
        right_col = QVBoxLayout()

        # Solver
        solver_label = QLabel("Solver:")
        self.solver_input = QComboBox()
        self.solver_input.addItems(['adam', 'sgd', 'lbfgs'])
        self.solver_input.setCurrentText('adam')
        self.solver_input.setFixedWidth(300)
        desc = QLabel(
            "Weight-optimization algorithm. <br>'adam': stable <br>'lbfgs': suitable for small datasets <br> 'sgd': suitable for large datasets; allows tuning the optimization process")
        desc.setStyleSheet(desc_style)
        right_col.addWidget(wrap_in_box([solver_label, self.solver_input, desc]))

        # Activation
        activation_label = QLabel("Activation Function:")
        self.activation_input = QComboBox()
        self.activation_input.addItems(['identity', 'logistic', 'tanh', 'relu'])
        self.activation_input.setCurrentText('relu')
        self.activation_input.setFixedWidth(300)
        desc = QLabel(
            "The activation function determines the output of neurons. <br>'relu': most common and stable <br> 'tanh': stable learning <br> 'logistic': used for binary output layers or small networks <br")
        desc.setStyleSheet(desc_style)
        right_col.addWidget(wrap_in_box([activation_label, self.activation_input, desc]))

        # Learning Rate
        learning_rate_label = QLabel("Learning Rate (learning_rate_init):")
        self.learning_rate_input = QLineEdit()
        self.learning_rate_input.setPlaceholderText("0.001")
        self.learning_rate_input.setFixedWidth(300)
        desc = QLabel("Controls the learning rate. Too large is unstable; too small slows learning.")
        desc.setStyleSheet(desc_style)
        right_col.addWidget(wrap_in_box([learning_rate_label, self.learning_rate_input, desc]))
        # comment
        font_box = QFrame()
        font_box.setFrameShape(QFrame.Box)
        font_box.setFrameShadow(QFrame.Sunken)
        font_box.setFixedHeight(80)  # comment
        font_layout = QVBoxLayout(font_box)
        font_layout.setContentsMargins(4, 2, 4, 2)  # comment
        font_layout.setSpacing(1)  # comment

        font_label = QLabel("Font settings:")
        font_label.setStyleSheet("font-size: 8pt; margin-bottom: 0px;")  # comment
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

        # comment
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

        self.mlpCvClassButton = QPushButton("5-Fold CV (MLP Classification)")
        self.mlpCvClassButton.clicked.connect(self.runMLPClassificationCV)
        buttons_layout.addWidget(self.mlpCvClassButton)

        self.mlpCvRegButton = QPushButton("5-Fold CV (MLP Regression)")
        self.mlpCvRegButton.clicked.connect(self.runMLPRegressionCV)
        buttons_layout.addWidget(self.mlpCvRegButton)

        # comment
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

        X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
        X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)

        feature_names = list(X_train_numeric.columns)

        # comment
        X_train_used = X_train_numeric.values
        X_test_used = X_test_numeric.values

        hidden_layer_input_text = self.hidden_layer_input.text().strip()
        hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(",")))

        alpha_input_text = self.alpha_input.text().strip()
        alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)

        lr_input_text = self.learning_rate_input.text().strip()
        learning_rate = 0.001 if not lr_input_text else float(lr_input_text)

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
            mlp.fit(X_train_used, y_train)

        if mlp.n_iter_ == mlp.max_iter:
            QMessageBox.warning(self, "Iteration Warning", "Maximum iterations reached. Consider increasing max_iter.")

        y_pred_train = mlp.predict(X_train_used)
        y_pred_test = mlp.predict(X_test_used)

        cm = confusion_matrix(y_test, y_pred_test)
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
            precision = np.nan_to_num(precision)

        labels = [f"Class {i}" for i in range(cm.shape[0])]
        cm_df = pd.DataFrame(
            cm,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels]
        )
        cm_df["Prediction Accuracy (%)"] = precision
        self.showConfusionMatrix(cm_df)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)

        perm_importance = permutation_importance(mlp, X_test_used, y_test, n_repeats=10, random_state=42)
        sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
        feature_importances = [(feature_names[idx], perm_importance.importances_mean[idx]) for idx in sorted_idx]
        self.showMLPFeatureImportances(feature_importances)

        # comment
        self.models["MLP Classification"] = {
            "model": mlp,
            "scaler": self._get_bundle_scaler(),
            "reducer": None,
            "feature_names": feature_names,
            "label_mapping": self._get_label_mapping()
        }

    def createMLPRegressionModel(self):
        if not self.checkDataSplit():
            return

        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
        X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)

        feature_names = list(X_train_numeric.columns)

        X_train_used = X_train_numeric.values
        X_test_used = X_test_numeric.values

        hidden_layer_input_text = self.hidden_layer_input.text().strip()
        hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(",")))

        alpha_input_text = self.alpha_input.text().strip()
        alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)

        lr_input_text = self.learning_rate_input.text().strip()
        learning_rate = 0.001 if not lr_input_text else float(lr_input_text)

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
            mlp.fit(X_train_used, y_train)

        if mlp.n_iter_ == mlp.max_iter:
            QMessageBox.warning(self, "Iteration Warning", "Maximum iterations reached. Consider increasing max_iter.")

        y_pred_train = mlp.predict(X_train_used)
        y_pred_test = mlp.predict(X_test_used)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)

        perm_importance = permutation_importance(mlp, X_test_used, y_test, n_repeats=10, random_state=42)
        sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
        feature_importances = [(feature_names[idx], perm_importance.importances_mean[idx]) for idx in sorted_idx]
        self.showMLPFeatureImportances(feature_importances)

        # comment
        self.models["MLP Regression"] = {
            "model": mlp,
            "scaler": self._get_bundle_scaler(),
            "reducer": None,
            "feature_names": feature_names,
            "label_mapping": None
        }

    def showMLPFeatureImportances(self, feature_importances):
        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Importances")
        dialog.setGeometry(100, 100, 600, 400)  # comment

        dialog_layout = QVBoxLayout(dialog)

        info_label = QLabel(
            "<b>How feature importance is computed</b><br>"
            "MLP importance on this screen is computed using scikit-learn's <code>permutation_importance</code>.<br>"
            "Importance is defined as how much model performance decreases when the values of a feature are randomly shuffled (performance drop).<br>"
            "Larger values mean more important features; values near zero or negative may indicate little influence or noise."
        )
        info_label.setWordWrap(True)
        dialog_layout.addWidget(info_label)

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

    # ------------------------------
    # 5-fold CV runners (MLP)
    # ------------------------------
    def runMLPClassificationCV(self):
        try:
            hidden_layer_input_text = self.hidden_layer_input.text().strip()
            hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(",")))

            alpha_input_text = self.alpha_input.text().strip()
            alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)

            lr_input_text = self.learning_rate_input.text().strip()
            learning_rate = 0.001 if not lr_input_text else float(lr_input_text)

            estimator = MLPClassifier(
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
                self.run_5fold_cv(estimator, task="classification")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))

    def runMLPRegressionCV(self):
        try:
            hidden_layer_input_text = self.hidden_layer_input.text().strip()
            hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(",")))

            alpha_input_text = self.alpha_input.text().strip()
            alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)

            lr_input_text = self.learning_rate_input.text().strip()
            learning_rate = 0.001 if not lr_input_text else float(lr_input_text)

            estimator = MLPRegressor(
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
                self.run_5fold_cv(estimator, task="regression")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))

    def setupRFTab(self):
        layout = QVBoxLayout()

        desc_style = "color: #555; font-size: 10pt; margin-bottom: 4px;"

        # comment
        rf_overview = QLabel(
            "<h3>🌲 Random Forest (RF)</h3>"
            "<p>Random Forest is an ensemble method that trains many decision trees and aggregates their predictions.<br>"
            "It tends to overfit less and performs well for both classification and regression.</p>"
        )
        rf_overview.setWordWrap(True)
        layout.addWidget(rf_overview)

        # comment
        groupBox = QFrame()
        groupBox.setFrameShape(QFrame.Box)
        groupBox.setFrameShadow(QFrame.Sunken)
        groupBoxLayout = QVBoxLayout(groupBox)

        params = [
            ("Max Depth:", 20, 1, 99999, "Limit the maximum depth of each tree.<br>Larger values make the model more complex; smaller values make it simpler."),
            ("N Estimators:", 20, 1, 99999, "Number of trees to build.<br>More trees often give more stable results but increase training time."),
            ("Min Samples Leaf:", 1, 1, 99999, "Minimum number of samples required at a leaf node.<br>Larger values simplify the model and reduce overfitting."),
            ("Min Samples Split:", 2, 2, 99999, "Minimum number of samples required to split an internal node.<br>Larger values reduce tree depth and simplify the model."),
            ("Random State:", 0, 0, 99999, "Seed value to control randomness.<br>Using the same value reproduces the same results.")
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

        # comment
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

        self.rfCvClassButton = QPushButton("5-Fold CV (RF Classification)")
        self.rfCvClassButton.clicked.connect(self.runRFClassificationCV)
        buttons_layout.addWidget(self.rfCvClassButton)

        self.rfCvRegButton = QPushButton("5-Fold CV (RF Regression)")
        self.rfCvRegButton.clicked.connect(self.runRFRegressionCV)
        buttons_layout.addWidget(self.rfCvRegButton)
        layout.addLayout(buttons_layout)

        # comment
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

        # comment
        main_layout = QHBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(right_side_layout)

        self.RFTab.setLayout(main_layout)

    # ------------------------------
    # 5-fold CV runners (RF)
    # ------------------------------
    def runRFClassificationCV(self):
        try:
            # comment
            max_depth = int(self.param_inputs["Max Depth:"].value())
            n_estimators = int(self.param_inputs["N Estimators:"].value())
            min_samples_leaf = int(self.param_inputs["Min Samples Leaf:"].value())
            min_samples_split = int(self.param_inputs["Min Samples Split:"].value())
            random_state = int(self.param_inputs["Random State:"].value())

            estimator = RandomForestClassifier(
                max_depth=max_depth,
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1
            )
            self.run_5fold_cv(estimator, task="classification")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))

    def runRFRegressionCV(self):
        try:
            max_depth = int(self.param_inputs["Max Depth:"].value())
            n_estimators = int(self.param_inputs["N Estimators:"].value())
            min_samples_leaf = int(self.param_inputs["Min Samples Leaf:"].value())
            min_samples_split = int(self.param_inputs["Min Samples Split:"].value())
            random_state = int(self.param_inputs["Random State:"].value())

            estimator = RandomForestRegressor(
                max_depth=max_depth,
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=random_state,
                n_jobs=-1
            )
            self.run_5fold_cv(estimator, task="regression")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))

    def createRFClassificationModel(self):
        if not self.checkDataSplit():
            return

        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
        X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)

        feature_names = list(X_train_numeric.columns)

        rf_clf = RandomForestClassifier(
            n_estimators=self.param_inputs["N Estimators:"].value(),
            max_depth=self.param_inputs["Max Depth:"].value(),
            min_samples_leaf=self.param_inputs["Min Samples Leaf:"].value(),
            min_samples_split=self.param_inputs["Min Samples Split:"].value(),
            random_state=0,
            n_jobs=-1
        )

        rf_clf.fit(X_train_numeric.values, y_train)
        y_pred_train = rf_clf.predict(X_train_numeric.values)
        y_pred_test = rf_clf.predict(X_test_numeric.values)
        accuracy = accuracy_score(y_test, y_pred_test)

        unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
        true = [f"true_{label}" for label in unique_labels]
        pred = [f"pred_{label}" for label in unique_labels]
        cm = confusion_matrix(y_test, y_pred_test)
        cm_df = pd.DataFrame(cm, index=true, columns=pred)

        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
            precision = np.nan_to_num(precision)
        cm_df["Prediction Accuracy (%)"] = precision

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        characteristics = X_train_numeric.columns
        importances = rf_clf.feature_importances_
        variable_importances = sorted(zip(characteristics, importances), key=lambda x: x[1], reverse=True)

        self.showRFClfResults(accuracy, cm_df, r2_train, r2_test, variable_importances)
        self.plotObservedVsPredicted(y_train, y_pred_train, y_test, y_pred_test,
                                     "RF Classification: Observed vs Predicted")

        # comment
        self.models["RF Classification"] = {
            "model": rf_clf,
            "scaler": self._get_bundle_scaler(),
            "reducer": None,
            "feature_names": feature_names,
            "label_mapping": self._get_label_mapping()
        }

    def createRFRegressionModel(self):
        if not self.checkDataSplit():
            return

        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()

        X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
        X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)

        feature_names = list(X_train_numeric.columns)

        rf_regr = RandomForestRegressor(
            n_estimators=self.param_inputs["N Estimators:"].value(),
            max_depth=self.param_inputs["Max Depth:"].value(),
            min_samples_leaf=self.param_inputs["Min Samples Leaf:"].value(),
            min_samples_split=self.param_inputs["Min Samples Split:"].value(),
            random_state=0,
            n_jobs=-1
        )

        rf_regr.fit(X_train_numeric.values, y_train)
        y_pred_train = rf_regr.predict(X_train_numeric.values)
        y_pred_test = rf_regr.predict(X_test_numeric.values)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)

        characteristics = X_train_numeric.columns
        importances = rf_regr.feature_importances_
        variable_importances = sorted(zip(characteristics, importances), key=lambda x: x[1], reverse=True)

        self.showRFRegResults(r2_train, r2_test, mse_test, rmse_test, variable_importances)
        self.plotObservedVsPredicted(y_train, y_pred_train, y_test, y_pred_test, "RF Regression: Observed vs Predicted")

        # comment
        self.models["RF Regression"] = {
            "model": rf_regr,
            "scaler": self._get_bundle_scaler(),
            "reducer": None,
            "feature_names": feature_names,
            "label_mapping": None
        }

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

        # Variable importance note
        var_imp_info = QLabel('Variable importance is displayed directly from scikit-learn RandomForest\'s `feature_importances_`. It is computed based on how much each feature reduces node impurity on average across the forest (MDI, often called “Gini importance”). Larger values mean the model split more often/more strongly on that feature and it contributed more to predictions. ')
        var_imp_info.setWordWrap(True)
        dialog_layout.addWidget(var_imp_info)

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

        # Variable importance note
        var_imp_info = QLabel('Variable importance is displayed directly from scikit-learn RandomForest\'s `feature_importances_`. It is computed based on how much each feature reduces node impurity on average across the forest (MDI, often called “Gini importance”). Larger values mean the model split more often/more strongly on that feature and it contributed more to predictions. ')
        var_imp_info.setWordWrap(True)
        dialog_layout.addWidget(var_imp_info)

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

        # comment
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

        # comment
        load_model_button = QPushButton("Load Saved Model")
        load_model_button.setFont(QFont('Arial', 12, QFont.Bold))
        load_model_button.setStyleSheet("QPushButton { padding: 8px; border-radius: 8px; border: 2px solid #000000; }")
        load_model_button.clicked.connect(self.loadPreviousModel)
        layout.addWidget(load_model_button)

        # comment
        load_unknown_button = QPushButton("Load Unknown Data (CSV)")
        load_unknown_button.setFont(QFont('Arial', 12, QFont.Bold))
        load_unknown_button.setStyleSheet(
            "QPushButton { padding: 8px; border-radius: 8px; border: 2px solid #000000; }")
        load_unknown_button.clicked.connect(self.loadUnknownSample)
        layout.addWidget(load_unknown_button)

        # comment
        self.prediction_table = QTableWidget()
        layout.addWidget(self.prediction_table)

        # comment
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
            # comment
            self.unknown_data = pd.read_csv(filename)
            # comment
            unknown_df = self.unknown_data.copy()
            if 'Sample' in unknown_df.columns:
                sample_series = unknown_df['Sample']
                unknown_df = unknown_df.drop(columns=['Sample'])
            else:
                sample_series = pd.Series([f"Sample {i + 1}" for i in range(len(unknown_df))])

            # comment
            unknown_df = unknown_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            data_to_scale = unknown_df

            # comment
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

            # comment
            if feature_names is not None:
                data_to_scale = data_to_scale.reindex(columns=feature_names)
                if data_to_scale.isnull().any().any():
                    data_to_scale = data_to_scale.fillna(0)
            else:
                print("[Warning] Model has no saved feature names — predictions may be unreliable!")

            # comment
            if scaler:
                data_scaled = scaler.transform(data_to_scale)
                data_scaled = pd.DataFrame(data_scaled, columns=data_to_scale.columns, index=data_to_scale.index)
            else:
                data_scaled = data_to_scale

            # comment
            if reducer:
                data_reduced = reducer.transform(data_scaled)
                data_used = pd.DataFrame(
                    data_reduced,
                    columns=[f"Component {i + 1}" for i in range(data_reduced.shape[1])],
                    index=data_scaled.index
                )
            else:
                data_used = data_scaled

            print("unknown raw mean/std:", data_to_scale.values.mean(), data_to_scale.values.std())
            if scaler:
                tmp = scaler.transform(data_to_scale)
                print("after scaler mean/std:", tmp.mean(), tmp.std())
            incoming = data_used.values if hasattr(data_used, "values") else np.asarray(data_used)


            print("incoming shape:", incoming.shape)
            print("incoming nan:", np.isnan(incoming).any(), "inf:", np.isinf(incoming).any())
            print("incoming mean/std:", float(np.mean(incoming)), float(np.std(incoming)))
            incoming = data_used.values if hasattr(data_used, "values") else np.asarray(data_used)

            print("model n_features_in_:", getattr(model, "n_features_in_", None))
            print("incoming shape:", incoming.shape)
            print("saved feature_names len:", len(feature_names) if feature_names is not None else None)
            print("scaler:", type(scaler).__name__ if scaler else None, "reducer:",
                  type(reducer).__name__ if reducer else None)

            # comment
            incoming = data_used.values if hasattr(data_used, "values") else np.asarray(data_used)

            # comment
            X_train_df = pd.read_csv(resource_path("Temp/X_train.csv"))

            # comment
            X_train_df = X_train_df.drop(columns=["Sample"], errors="ignore")
            X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce").fillna(0)

            # comment
            if feature_names is not None:
                X_train_df = X_train_df.reindex(columns=feature_names).fillna(0)

            # comment
            if scaler:
                X_train_proc = scaler.transform(X_train_df)
            else:
                X_train_proc = X_train_df.values

            # comment
            if reducer:
                X_train_proc = reducer.transform(X_train_proc)

            X_train_proc = np.asarray(X_train_proc)

            print("incoming shape:", incoming.shape)
            print("X_train_proc shape:", X_train_proc.shape)

            # comment
            d = np.linalg.norm(X_train_proc[None, :, :] - incoming[:, None, :], axis=2)
            min_d = d.min(axis=1)
            argmin = d.argmin(axis=1)

            print("min distance per unknown row:", min_d)
            print("closest train row index:", argmin)

            # comment
            tol = 1e-9
            same_mask = min_d <= tol
            print("same_mask (min_d<=1e-9):", same_mask)
            print("how many exactly same?:", same_mask.sum(), "/", len(same_mask))

            # comment
            predictions = model.predict(data_used)

            # comment
            if label_mapping:
                inverse_map = {v: k for k, v in label_mapping.items()}
                predictions = [inverse_map.get(p, p) for p in predictions]

            # comment
            self.prediction_table.clear()
            self.prediction_table.setColumnCount(2)
            self.prediction_table.setHorizontalHeaderLabels(["Sample", "Prediction"])
            self.prediction_table.setRowCount(len(predictions))

            for i, pred in enumerate(predictions):
                self.prediction_table.setItem(i, 0, QTableWidgetItem(str(sample_series.iloc[i])))
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

    # comment
    # ============================================================
    def applyModelReducer(self):
        try:
            if not hasattr(self, 'scaled_unknown_data'):
                QMessageBox.warning(self, "Data Error", "Please scale the unknown data first.")
                return

            # comment
            selected_models = [name for name, checkbox in self.modelCheckBoxes.items() if checkbox.isChecked()]
            if not selected_models:
                QMessageBox.warning(self, "Model Selection Error", "Please select at least one model.")
                return

            model_name = selected_models[0]
            reducer = self.model_reducers.get(model_name, None)
            if reducer is None:
                QMessageBox.warning(self, "Reducer Error", f"No reducer found for '{model_name}'.")
                return

            # comment
            unknown_df = pd.DataFrame(self.scaled_unknown_data, columns=self.unknown_data.columns)
            if hasattr(self, 'feature_names'):
                missing = set(self.feature_names) - set(unknown_df.columns)
                if missing:
                    QMessageBox.warning(self, "Feature Mismatch",
                                        f"The following features are missing in unknown data:\n{', '.join(missing)}")
                    return
                # comment
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
        During prediction, align by feature names and then run reducer/model prediction
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

        # comment
        unknown_df = pd.DataFrame(self.scaled_unknown_data, columns=self.unknown_data.columns)
        if hasattr(self, 'feature_names'):
            missing = set(self.feature_names) - set(unknown_df.columns)
            if missing:
                QMessageBox.warning(self, "Feature Mismatch",
                                    f"The following features are missing in unknown data:\n{', '.join(missing)}")
                return
            unknown_df = unknown_df.loc[:, self.feature_names]

        data_used = unknown_df.values

        # comment
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

        # comment
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

        # comment
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

        # comment
        x_label = 'Observed'
        y_label = 'Predicted'
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # comment
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
            "<h3>🧮 KNN (K-Nearest Neighbors) Model </h3>"
            "<p>When a new data point is given, KNN "
            "finds the K nearest neighbors in the existing data and "
            "predicts by majority vote (classification) or average (regression).<br>"
            "It does not explicitly learn parameters; instead it computes distances at prediction time, "
            "so it is often called a 'lazy learning' method.</p>"
        )
        guide_label.setWordWrap(True)
        guide_layout.addWidget(guide_label)

        dimreduce_label = QLabel(
            "<h4>📉 Comparison of dimensionality reduction methods</h4>"
            "<ul>"
            "<li><b>PCA (Principal Component Analysis)</b>: Based on unsupervised learning. "
            "It reduces dimensions by redefining axes along directions with the largest variance.<br>"
            "‣ It does not use class labels and is useful for visualization or noise reduction.</li><br>"
            "<li><b>LDA (Linear Discriminant Analysis)</b>: Based on supervised learning. "
            "It reduces dimensions by finding axes that maximize separation between classes.<br>"
            "‣ In labeled classification problems, it can visualize class boundaries more clearly.</li><br>"
            "<li><b>NCA (Neighborhood Components Analysis)</b>: Based on supervised learning. "
            "It learns the feature space to maximize KNN classification performance.<br>"
            "‣ It is more flexible than LDA and can perform better even with nonlinear relationships.</li>"
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

        # 5-fold CV buttons
        self.knnCvClassButton = QPushButton("5-Fold CV (Classification)")
        self.knnCvClassButton.clicked.connect(self.runKnnClassificationCV)
        buttons_layout.addWidget(self.knnCvClassButton)

        self.knnCvRegButton = QPushButton("5-Fold CV (Regression)")
        self.knnCvRegButton.clicked.connect(self.runKnnRegressionCV)
        buttons_layout.addWidget(self.knnCvRegButton)

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

    # ------------------------------
    # 5-fold CV runners (KNN)
    # ------------------------------
    def showKNNPermutationImportance(self, reducer, X_eval, y_eval, model, feature_names, title_prefix="KNN", task="classification"):
        """KNN does not provide feature importance via internal parameters,
        so importance is computed and displayed using permutation importance (score drop when a feature is shuffled).

        - X_eval is expected to be in the original feature space.
        """
        try:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Feature Importance")
            msg.setText(
                f"{title_prefix} does not compute feature importance directly from internal model values.\n\n"
                "Feature importance can be computed and displayed using permutation importance.\n"
                "(Importance is computed as how much the performance score drops when the feature is shuffled)"
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if msg.exec_() != QMessageBox.Ok:
                return

            scoring = "accuracy" if task == "classification" else "r2"

            class _ReducerWrappedEstimator:
                def __init__(self, reducer, model):
                    self.reducer = reducer
                    self.model = model

                def fit(self, X, y=None):
                    return self

                def predict(self, X):
                    X_in = X
                    if isinstance(X_in, pd.DataFrame):
                        X_in = X_in.values
                    if self.reducer is not None:
                        X_in = self.reducer.transform(X_in)
                    return self.model.predict(X_in)

                def score(self, X, y):
                    y_pred = self.predict(X)
                    if task == "classification":
                        return accuracy_score(y, y_pred)
                    return r2_score(y, y_pred)

            est = _ReducerWrappedEstimator(reducer, model)

            X_use = X_eval.copy()
            y_use = np.array(y_eval)
            max_n = 2000
            if len(X_use) > max_n:
                rng = np.random.RandomState(42)
                idx = rng.choice(len(X_use), size=max_n, replace=False)
                if isinstance(X_use, pd.DataFrame):
                    X_use = X_use.iloc[idx]
                else:
                    X_use = X_use[idx]
                y_use = y_use[idx]

            result = permutation_importance(
                est, X_use, y_use,
                n_repeats=5,
                random_state=42,
                scoring=scoring
            )

            importances_mean = result.importances_mean
            names = list(feature_names)
            if len(names) != len(importances_mean):
                names = [f"X{i}" for i in range(len(importances_mean))]

            order = np.argsort(importances_mean)[::-1]
            sorted_names = [names[i] for i in order]
            sorted_vals = [importances_mean[i] for i in order]

            dialog = QDialog(self)
            dialog.setWindowTitle("Feature Importances")
            layout = QVBoxLayout(dialog)

            info = QLabel(
                f"{title_prefix} feature importance was computed using permutation importance.\n"
                "Importance is computed based on how much the performance score decreases when each feature is shuffled."
            )
            info.setWordWrap(True)
            layout.addWidget(info)

            table = QTableWidget(dialog)
            table.setRowCount(len(sorted_names))
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Feature", "Importance"])
            for i, (fname, val) in enumerate(zip(sorted_names, sorted_vals)):
                table.setItem(i, 0, QTableWidgetItem(str(fname)))
                table.setItem(i, 1, QTableWidgetItem(f"{val:.6f}"))
            table.resizeColumnsToContents()
            layout.addWidget(table)

            dialog.setLayout(layout)
            dialog.setWindowModality(Qt.NonModal)
            dialog.show()

        except Exception as e:
            QMessageBox.warning(self, "Permutation Importance Error", f"Failed to compute permutation importance:\n{e}")

    def runKnnClassificationCV(self):
        try:
            n_neighbors = self.n_neighbors_input.value()
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.run_5fold_cv(knn, task="classification")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))

    def runKnnRegressionCV(self):
        try:
            n_neighbors = self.n_neighbors_input.value()
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            self.run_5fold_cv(knn, task="regression")
        except Exception as e:
            QMessageBox.warning(self, "CV Error", str(e))

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
        reducer.fit(X_train_numeric.values, y_train)
        X_train_embedded = reducer.transform(X_train_numeric.values)
        X_test_embedded = reducer.transform(X_test_numeric.values)

        knn.fit(X_train_embedded, y_train)
        accuracy = knn.score(X_test_embedded, y_test)
        self.plotResults(method_name,
                         X_train_embedded, y_train,
                         X_test_embedded, y_test,
                         n_neighbors,
                         score_value=accuracy, score_label="Test accuracy")

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

        # (Removed) Duplicate scatter plot call
        self.showConfusionMatrix(cm_df)
        # comment
        feature_names = list(X_train_numeric.columns)
        self.showKNNPermutationImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix="KNN Classification", task="classification")


        self.models["KNN Classification"] = {
            "model": knn,
            "scaler": self._get_bundle_scaler(),  # comment
            "reducer": reducer,  # PCA/LDA/NCA
            "feature_names": feature_names,  # comment
            "label_mapping": self._get_label_mapping()  # comment
        }

        if reducer:
            self.model_reducers["KNN Classification"] = reducer

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
        reducer.fit(X_train_numeric.values, y_train)
        X_train_embedded = reducer.transform(X_train_numeric.values)
        X_test_embedded = reducer.transform(X_test_numeric.values)

        knn.fit(X_train_embedded, y_train)


        y_pred_train = knn.predict(X_train_embedded)
        y_pred_test = knn.predict(X_test_embedded)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        self.plotResults(method_name,
                         X_train_embedded, y_train,
                         X_test_embedded, y_test,
                         n_neighbors,
                         score_value=r2_test, score_label="Test R2")

        self.plotObservedVsPredicted(
            y_train, y_pred_train,
            y_test, y_pred_test,
            f"KNN Regression Observed vs Predicted\nTrain R2={r2_train:.3f}, Test R2={r2_test:.3f}"
        )

        # comment
        feature_names = list(X_train_numeric.columns)
        self.showKNNPermutationImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix="KNN Regression", task="regression")


        self.models["KNN Regression"] = {
            "model": knn,
            "scaler": self._get_bundle_scaler(),  # comment
            "reducer": reducer,  # comment
            "feature_names": feature_names,
            "label_mapping": None  # comment
        }

        if reducer:
            self.model_reducers["KNN Regression"] = reducer

    def plotResults(self, name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors,
                    score_value=None, score_label=None):

        plt.figure() #comment
        unique_labels_train = np.unique(y_train)
        unique_labels_test = np.unique(y_test)
        label_colors = {
            label: "#{:02x}{:02x}{:02x}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for label in np.unique(np.concatenate((y_train, y_test)))}

        legend_widget = getattr(self, "legendNameInput", None)
        legend_name = legend_widget.text() if (legend_widget is not None and legend_widget.text()) else "Label"

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
        title = f"{name} - KNN (k={n_neighbors})"
        if score_value is not None and score_label is not None:
            title += f"\n{score_label} = {score_value:.3f}"

        plt.title(title,
                  fontsize=self.fontSizeInput.value(),
                  fontname=self.fontTypeComboBox.currentText())

        plt.xlabel("Component 1", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.ylabel("Component 2", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.show()


    def plotObservedVsPredicted(self, y_train, y_pred_train, y_test, y_pred_test, title):
        if hasattr(self, 'observed_vs_predicted_dialog'):
            self.observed_vs_predicted_dialog.close()
        # comment
        plt.rcParams['font.size'] = self.fontSizeInput.value()
        plt.rcParams['font.family'] = self.fontTypeComboBox.currentText()

        # comment
        fig, ax = plt.subplots(figsize=(10, 8))

        # comment
        scatter_train = ax.scatter(y_train, y_pred_train, c='blue', label='Training Set', marker='o', s=50, alpha=0.3)
        scatter_test = ax.scatter(y_test, y_pred_test, c='red', label='Test Set', marker='x', s=100, alpha=0.7)

        # comment
        ax.set_xlabel('Observed')
        ax.set_ylabel('Predicted')
        ax.set_title(title)

        # comment
        legend = ax.legend()
        legend.set_draggable(True)

        # comment
        ax.plot([min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                [min(y_train.min(), y_test.min()), max(y_train.max(), y_test.max())],
                'k--', label='45-degree line')

        # comment
        ax.text(0.05, 0.95,
                f'Training R2: {r2_score(y_train, y_pred_train):.3f}\nTest R2: {r2_score(y_test, y_pred_test):.3f}\nMSE: {mean_squared_error(y_test, y_pred_test):.3f}\nRMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f}',
                transform=ax.transAxes, fontsize=12, verticalalignment='top')

        # comment
        figure_canvas = FigureCanvas(fig)
        self.figure_canvas = figure_canvas  # comment

        # comment
        dialog = QDialog(self)
        dialog.setWindowTitle("Observed vs Predicted")
        dialog.setGeometry(100, 100, 800, 600)

        dialog_layout = QVBoxLayout(dialog)
        # comment
        toolbar = NavigationToolbar(figure_canvas, dialog)
        dialog_layout.addWidget(toolbar)
        # comment
        dialog_layout.addWidget(figure_canvas)

        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()
        # comment
        self.observed_vs_predicted_dialog = dialog

        # comment
        plt.close(fig)

    def getSelectedDimReductionMethod(self):
        if hasattr(self, "pcaCheckBox") and self.pcaCheckBox.isChecked():
            return "PCA", PCA(n_components=2)
        elif hasattr(self, "ldaCheckBox") and self.ldaCheckBox.isChecked():
            return "LDA", LDA(n_components=2)
        elif hasattr(self, "ncaCheckBox") and self.ncaCheckBox.isChecked():
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

    # ------------------------------
    # 5-fold Cross-Validation helpers (k is fixed to 5)
    # ------------------------------

    def _require_data_split_ready(self):
        """Return True if CSV is loaded and Data Split tab has been initialized."""
        if not hasattr(self, "rawDataRadioButton") or not hasattr(self, "scaledDataRadioButton"):
            QMessageBox.information(
                self,
                "CV Notice",
                "Please load a CSV data file first and then perform Data Split."
            )
            return False
        return True

    def _refresh_cv_group_columns(self):
        """Populate GroupKFold group-column dropdown from the currently loaded dataset."""
        if not hasattr(self, "cvGroupColumnCombo"):
            return

        self.cvGroupColumnCombo.clear()
        df = getattr(self.csvViewer, "original_data", None)
        if df is None:
            return
        # show all columns (including Sample/Label if present). user can pick an ID-like column
        self.cvGroupColumnCombo.addItems(list(df.columns))

    def _update_cv_strategy_ui(self):
        if not hasattr(self, "cvSplitStrategyCombo"):
            return

        strategy = self.cvSplitStrategyCombo.currentText()
        desc_map = {
            "StratifiedKFold (classification)": (
                "This is the most commonly used 5-fold strategy for classification.\n"
                "Each fold is split so that class proportions stay as similar as possible (useful for imbalanced data)."
            ),
            "KFold (general)": (
                "This is the most basic 5-fold strategy.\n"
                "It splits evenly without considering label proportions (or with shuffling, depending on settings)."
            ),
            "GroupKFold (grouped samples)": (
                "If samples from the same entity (patient/user/sample ID) appear in both train and validation, leakage can occur.\n"
                "GroupKFold keeps the same group within a single fold only. (e.g., patient_id)"
            ),
            "TimeSeriesSplit (time order)": (
                "This 5-fold strategy is for time-series / time-ordered data.\n"
                "It trains on the past and validates on the future, and does not use shuffling."
            ),
        }

        self.cvSplitStrategyDesc.setText(desc_map.get(strategy, ""))

        is_group = strategy.startswith("GroupKFold")
        self.cvGroupColumnLabel.setVisible(is_group)
        self.cvGroupColumnCombo.setVisible(is_group)

    def _get_cv_splitter(self, y, n_splits=5):
        """Return (splitter, groups_or_None)."""
        strategy = getattr(self, "cvSplitStrategyCombo", None)
        strategy = strategy.currentText() if strategy else "StratifiedKFold (classification)"

        if strategy.startswith("StratifiedKFold"):
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            return splitter, None
        if strategy.startswith("KFold"):
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            return splitter, None
        if strategy.startswith("TimeSeriesSplit"):
            splitter = TimeSeriesSplit(n_splits=n_splits)
            return splitter, None
        if strategy.startswith("GroupKFold"):
            splitter = GroupKFold(n_splits=n_splits)
            group_col = self.cvGroupColumnCombo.currentText() if hasattr(self, "cvGroupColumnCombo") else None
            df = getattr(self.csvViewer, "original_data", None)
            if df is None or not group_col or group_col not in df.columns:
                raise ValueError("GroupKFold requires a valid group column. Please load data and select a group column.")
            groups = df[group_col].values
            return splitter, groups

        # fallback
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return splitter, None

    def _get_cv_X_y(self):
        """Load X/y used for CV from Temp (raw or scaled X, and scaled_y)."""
        if not self._require_data_split_ready():
            return None, None
        if self.rawDataRadioButton.isChecked():
            X_file = resource_path("Temp/original_X.csv")
        else:
            X_file = resource_path("Temp/scaled_X.csv")

        if not os.path.exists(X_file):
            raise FileNotFoundError(f"{X_file} does not exist. Please load data (and scale if needed) first.")

        X = pd.read_csv(X_file)
        y_df = pd.read_csv(resource_path("Temp/scaled_y.csv"))
        y = y_df.values.ravel()

        X_numeric = self._drop_sample_and_numeric(X).fillna(0)
        return X_numeric, y

    def _format_mean_std(self, arr, digits=3):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return "N/A"
        return f"{arr.mean():.{digits}f} ± {arr.std(ddof=1):.{digits}f}"

    def run_5fold_cv(self, estimator, task="classification"):
        """Run 5-fold CV with the selected split strategy and show a results dialog."""
        X, y = self._get_cv_X_y()
        if X is None or y is None:
            return
        splitter, groups = self._get_cv_splitter(y, n_splits=5)
        if splitter is None:
            return

        fold_scores = {"accuracy": [], "f1": [], "roc_auc": [], "r2": [], "rmse": []}

        # Determine multi-class
        unique_y = np.unique(y)
        is_multiclass = len(unique_y) > 2

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups) if groups is not None else splitter.split(X, y), start=1):
            X_train_df = X.iloc[train_idx]
            X_test_df = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            # Reducer is fit inside each fold (if selected)
            selected = self.getSelectedDimReductionMethod()
            reducer = None
            if selected:
                _, reducer = selected

            if reducer is not None:
                reducer.fit(X_train_df.values, y_train)
                X_train_used = reducer.transform(X_train_df.values)
                X_test_used = reducer.transform(X_test_df.values)
            else:
                X_train_used = X_train_df.values
                X_test_used = X_test_df.values

            model = clone(estimator)
            model.fit(X_train_used, y_train)

            if task == "regression":
                y_pred = model.predict(X_test_used)
                r2 = r2_score(y_test, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                fold_scores["r2"].append(r2)
                fold_scores["rmse"].append(rmse)
                continue

            # classification
            y_pred = model.predict(X_test_used)
            fold_scores["accuracy"].append(accuracy_score(y_test, y_pred))

            # F1
            avg = "macro" if is_multiclass else "binary"
            try:
                fold_scores["f1"].append(f1_score(y_test, y_pred, average=avg))
            except Exception:
                fold_scores["f1"].append(np.nan)            # ROC-AUC (optional)
            auc_val = np.nan
            try:
                # Get score/probability outputs
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test_used)
                elif hasattr(model, "decision_function"):
                    y_score = model.decision_function(X_test_used)
                else:
                    y_score = None

                if y_score is not None:
                    all_classes = np.unique(y)  # classes from full dataset
                    if is_multiclass:
                        # Robust multiclass AUC: compute OvR AUC per class when possible and average.
                        from sklearn.preprocessing import label_binarize
                        y_bin = label_binarize(y_test, classes=all_classes)

                        # Ensure y_score is 2D with class-wise columns
                        if isinstance(y_score, np.ndarray) and y_score.ndim == 1:
                            # If a 1D score is returned in multiclass, AUC is not defined.
                            auc_val = np.nan
                        else:
                            y_score_mat = np.asarray(y_score)

                            # If columns mismatch, try to align when model exposes classes_
                            if hasattr(model, "classes_") and y_score_mat.ndim == 2:
                                model_classes = np.asarray(model.classes_)
                                # Map model output columns to all_classes
                                col_map = {c: i for i, c in enumerate(model_classes)}
                                aligned = np.full((y_score_mat.shape[0], len(all_classes)), np.nan, dtype=float)
                                for j, c in enumerate(all_classes):
                                    if c in col_map and col_map[c] < y_score_mat.shape[1]:
                                        aligned[:, j] = y_score_mat[:, col_map[c]]
                                y_score_mat = aligned

                            aucs = []
                            for j in range(len(all_classes)):
                                # Need both 0 and 1 present for binary AUC for that class
                                if y_bin[:, j].max() == 1 and y_bin[:, j].min() == 0 and not np.all(np.isnan(y_score_mat[:, j])):
                                    try:
                                        aucs.append(roc_auc_score(y_bin[:, j], y_score_mat[:, j]))
                                    except Exception:
                                        pass
                            auc_val = float(np.mean(aucs)) if len(aucs) > 0 else np.nan
                    else:
                        # binary: require both classes present in this fold
                        if len(np.unique(y_test)) < 2:
                            auc_val = np.nan
                        else:
                            if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                                auc_val = roc_auc_score(y_test, y_score[:, 1])
                            else:
                                auc_val = roc_auc_score(y_test, y_score)
            except Exception:
                auc_val = np.nan
            fold_scores["roc_auc"].append(auc_val)

        self._show_cv_results_dialog(fold_scores, task=task)

    def _show_cv_results_dialog(self, fold_scores: dict, task="classification"):
        dialog = QDialog(self)
        dialog.setWindowTitle("5-Fold Cross-Validation Results")
        dialog.setGeometry(120, 120, 560, 320)
        layout = QVBoxLayout(dialog)

        strategy = self.cvSplitStrategyCombo.currentText() if hasattr(self, "cvSplitStrategyCombo") else "(unknown)"
        header = QLabel(f"Split strategy: <b>{strategy}</b>  |  k=5")
        header.setWordWrap(True)
        layout.addWidget(header)

        table = QTableWidget(dialog)

        if task == "regression":
            metrics = [
                ("R2 (CV)", fold_scores["r2"]),
                ("RMSE (CV)", fold_scores["rmse"]),
            ]
        else:
            metrics = [
                ("Accuracy (CV)", fold_scores["accuracy"]),
                ("F1-score (CV)", fold_scores["f1"]),
                ("ROC-AUC (CV)", fold_scores["roc_auc"]),
            ]

        table.setRowCount(len(metrics))
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Metric", "Mean ± SD", "Fold scores"])

        for i, (name, arr) in enumerate(metrics):
            arr_np = np.asarray(arr, dtype=float)
            # remove nan for mean/std display if present
            arr_clean = arr_np[~np.isnan(arr_np)]
            mean_std = self._format_mean_std(arr_clean) if arr_clean.size else "N/A"
            fold_txt = ", ".join([f"{v:.3f}" if not np.isnan(v) else "NA" for v in arr_np])
            table.setItem(i, 0, QTableWidgetItem(name))
            table.setItem(i, 1, QTableWidgetItem(mean_std))
            table.setItem(i, 2, QTableWidgetItem(fold_txt))

        table.resizeColumnsToContents()
        layout.addWidget(table)

        dialog.setLayout(layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()

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

        # ------------------------------
        # 5-fold Cross-Validation options (k is fixed to 5)
        # ------------------------------
        cvGroupBox = QFrame()
        cvGroupBox.setFrameShape(QFrame.Box)
        cvGroupBox.setFrameShadow(QFrame.Sunken)
        cvLayout = QVBoxLayout(cvGroupBox)

        cvTitle = QLabel("5-Fold Cross-Validation (k=5)")
        cvTitle.setStyleSheet("font-weight: bold;")
        cvLayout.addWidget(cvTitle)

        cvSelectLabel = QLabel("Select CV split strategy:")
        self.cvSplitStrategyCombo = QComboBox()
        self.cvSplitStrategyCombo.addItems([
            "StratifiedKFold (classification)",
            "KFold (general)",
            #"GroupKFold (grouped samples)",
            #"TimeSeriesSplit (time order)"
        ])
        cvLayout.addWidget(cvSelectLabel)
        cvLayout.addWidget(self.cvSplitStrategyCombo)

        self.cvSplitStrategyDesc = QLabel("")
        self.cvSplitStrategyDesc.setWordWrap(True)
        self.cvSplitStrategyDesc.setStyleSheet("color: gray; font-size: 11px;")
        cvLayout.addWidget(self.cvSplitStrategyDesc)

        # GroupKFold needs a group column (e.g., patient ID / subject ID)
        self.cvGroupColumnLabel = QLabel("Group column (for GroupKFold):")
        self.cvGroupColumnCombo = QComboBox()
        self.cvGroupColumnLabel.setVisible(False)
        self.cvGroupColumnCombo.setVisible(False)
        cvLayout.addWidget(self.cvGroupColumnLabel)
        cvLayout.addWidget(self.cvGroupColumnCombo)

        layout.addWidget(cvGroupBox, 0, 1, 2, 1)

        self.randomSelectOptions = QWidget()
        self.randomSelectOptionsLayout = QVBoxLayout(self.randomSelectOptions)
        self.randomSelectLabel = QLabel("Random Select Options:")
        self.stratifyCheckBox = QCheckBox("Stratify")
        self.stratifyHelpLabel = QLabel(
            "→ Stratify splits data so that class proportions are preserved in both train/test sets.<br>"
            "   Use this to preserve class distribution in imbalanced datasets."
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
            "→ With the same Random State value, the same samples are chosen for the test set each time.<br>"
            "   Changing the value changes the split."
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

        # CV UI init
        self.cvSplitStrategyCombo.currentIndexChanged.connect(self._update_cv_strategy_ui)
        self._refresh_cv_group_columns()
        self._update_cv_strategy_ui()

        self.dataSplitTab.setLayout(layout)

    def _get_bundle_scaler(self):
        # comment
        return self.scaler if getattr(self, "last_split_used_scaled", False) else None

    def _get_label_mapping(self):
        return getattr(self.csvViewer, "label_mapping", None)

    def splitData(self):
        if self.rawDataRadioButton.isChecked():
            X_file = resource_path("Temp/original_X.csv")
        elif self.scaledDataRadioButton.isChecked():
            X_file = resource_path("Temp/scaled_X.csv")
        else:
            QMessageBox.warning(self, "Selection Error", "Please select either raw data or scaled data.")
            return
        self.last_split_used_scaled = self.scaledDataRadioButton.isChecked()

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
                # comment
                if isinstance(value, float):
                    formatted_value = f"{int(value)}" if value.is_integer() else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                self.trainSetWidget.setItem(i, j, QTableWidgetItem(formatted_value))

            # comment
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
                # comment
                if isinstance(value, float):
                    formatted_value = f"{int(value)}" if value.is_integer() else f"{value:.4f}"
                else:
                    formatted_value = str(value)
                self.testSetWidget.setItem(i, j, QTableWidgetItem(formatted_value))

            # comment
            y_value = y_test.values[i]
            if isinstance(y_value, float):
                formatted_value = f"{int(y_value)}" if y_value.is_integer() else f"{y_value:.4f}"
            else:
                formatted_value = str(y_value)
            self.testSetWidget.setItem(i, X_test.shape[1], QTableWidgetItem(formatted_value))

    def loadCsv(self, checked=False):
        # comment
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "",
            "CSV Files (*.csv);;All Files (*)", options=options
        )
        if not filename:
            return

        try:
            # comment
            self.csvViewer.loadCsv(filename)

            # comment
            if hasattr(self, "guideWidget"):
                self.guideWidget.hide()
            self.csvViewer.show()

            # comment
            output_dir = resource_path('Temp')
            os.makedirs(output_dir, exist_ok=True)

            # comment
            if getattr(self.csvViewer, "original_data", None) is not None:
                self.csvViewer.original_data.to_csv(os.path.join(output_dir, "original_X.csv"), index=False)

            # comment
            if getattr(self.csvViewer, "y", None) is not None:
                pd.DataFrame(self.csvViewer.y, columns=["Label"]).to_csv(os.path.join(output_dir, "scaled_y.csv"),
                                                                         index=False)

            # refresh CV group-column list (if Data Split tab is already created)
            self._refresh_cv_group_columns()

            QMessageBox.information(self, "Load Complete", f"Loaded CSV:\n{os.path.basename(filename)}")

        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load CSV: {e}")

    def exitApp(self):
        reply = QMessageBox.question(self, 'Message', 'Are you sure you want to close?', QMessageBox.Yes | QMessageBox.Cancel,
                                     QMessageBox.Cancel)
        if reply == QMessageBox.Yes:
            QCoreApplication.instance().quit()

    def show_scaled_data(self, scaled_X_df, y, headers):
        self.scaledDataWidget.clear()
        self.scaledDataWidget.setRowCount(len(scaled_X_df))
        self.scaledDataWidget.setColumnCount(len(headers) + 2)  # comment
        self.scaledDataWidget.setHorizontalHeaderLabels(["Sample"] + headers + ["Label"])

        for i, row in scaled_X_df.iterrows():
            # comment
            sample_value = row.iloc[0]
            if isinstance(sample_value, float) and sample_value.is_integer():
                formatted_sample = f"{int(sample_value)}"  # comment
            else:
                formatted_sample = str(sample_value)
            self.scaledDataWidget.setItem(i, 0, QTableWidgetItem(formatted_sample))

            # comment
            for j, cell in enumerate(row[1:], start=1):
                if isinstance(cell, float):
                    formatted_value = f"{int(cell)}" if cell.is_integer() else f"{cell:.4f}"  # comment
                elif isinstance(cell, int):
                    formatted_value = f"{cell}"  # comment
                else:
                    formatted_value = str(cell)  # comment
                self.scaledDataWidget.setItem(i, j, QTableWidgetItem(formatted_value))

            # comment
            if isinstance(y[i], float):
                formatted_label = f"{int(y[i])}" if y[i].is_integer() else f"{y[i]:.4f}"  # comment
            elif isinstance(y[i], int):
                formatted_label = f"{y[i]}"  # comment
            else:
                formatted_label = str(y[i])  # comment
            self.scaledDataWidget.setItem(i, len(headers) + 1, QTableWidgetItem(formatted_label))  # comment

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
        if not dialog.exec_():
            return

        selections = dialog.getSelections()
        label_column_name = next(key for key, value in selections.items() if value == 'Label')
        sample_column_name = next((key for key, value in selections.items() if value == 'Sample'), None)
        feature_columns = [key for key, value in selections.items() if value == 'Feature']

        if sample_column_name is None:
            data['Sample'] = range(1, len(data) + 1)
            sample_column_name = 'Sample'

        # comment
        if pd.api.types.is_numeric_dtype(data[label_column_name]):
            y = data[label_column_name].to_numpy()
            self.label_mapping = None
        else:
            unique_labels = pd.unique(data[label_column_name])
            labelDialog = LabelMappingDialog(unique_labels)
            if not labelDialog.exec_():
                return
            labelMappings = labelDialog.getLabelMappings()
            if not labelMappings:
                return
            self.label_mapping = labelMappings
            data[label_column_name] = data[label_column_name].map(labelMappings).astype(float)
            y = data[label_column_name].to_numpy()

        # comment
        for col in feature_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # comment
        data = data.rename(columns={sample_column_name: 'Sample', label_column_name: 'Label'})

        # comment
        self.original_data = data[['Sample'] + feature_columns]
        self.X = data[feature_columns].to_numpy()
        self.y = y

        # comment
        self.showCsvData(data[['Sample'] + feature_columns + ['Label']].values.tolist(),
                         ['Sample'] + feature_columns + ['Label'])

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




# ------------------------
# Safety patch: ensure menu actions have handlers
# (Some earlier edits accidentally nested these functions outside MyApp.)
# ------------------------
def _kuquickml_myapp_loadCsv(self, checked=False):
    """Open a CSV file and load it through CsvViewer, then persist Temp files for split/CV."""
    options = QFileDialog.Options()
    filename, _ = QFileDialog.getOpenFileName(
        self, "Open CSV File", "",
        "CSV Files (*.csv);;All Files (*)", options=options
    )
    if not filename:
        return
    try:
        # Use CsvViewer loader (includes ColumnRoleDialog / label mapping)
        self.csvViewer.loadCsv(filename)

        # Optional: hide guide widget if present
        if hasattr(self, "guideWidget"):
            self.guideWidget.hide()
        self.csvViewer.show()

        # Persist files used by data split / CV routines
        output_dir = resource_path('Temp')
        os.makedirs(output_dir, exist_ok=True)

        if getattr(self.csvViewer, "original_data", None) is not None:
            self.csvViewer.original_data.to_csv(os.path.join(output_dir, "original_X.csv"), index=False)

        if getattr(self.csvViewer, "y", None) is not None:
            pd.DataFrame(self.csvViewer.y, columns=["Label"]).to_csv(
                os.path.join(output_dir, "scaled_y.csv"), index=False
            )

        # refresh CV group-column list (if Data Split tab is already created)
        if hasattr(self, "_refresh_cv_group_columns"):
            self._refresh_cv_group_columns()

        QMessageBox.information(self, "Load Complete", f"Loaded CSV:\n{os.path.basename(filename)}")
    except Exception as e:
        QMessageBox.warning(self, "Load Error", f"Failed to load CSV: {e}")


def _kuquickml_myapp_exitApp(self):
    reply = QMessageBox.question(
        self, 'Message', 'Are you sure you want to close?',
        QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel
    )
    if reply == QMessageBox.Yes:
        QCoreApplication.instance().quit()


# Attach missing handlers if they are not present on MyApp
try:
    if not hasattr(MyApp, "loadCsv"):
        MyApp.loadCsv = _kuquickml_myapp_loadCsv
    if not hasattr(MyApp, "exitApp"):
        MyApp.exitApp = _kuquickml_myapp_exitApp
except Exception:
    # If MyApp is not defined for some reason, ignore.
    pass

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


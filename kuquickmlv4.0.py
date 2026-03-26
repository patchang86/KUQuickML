print("실행 중입니다... PC 환경에 따라 최대 1분 가량 소요될 수 있습니다. 프로그램이 실행 중인 동안 본 콘솔 창을 닫지 마십시오.")
import sys
import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PyQt5.QtWidgets import (QColorDialog, QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget, QPushButton, QDialog, QLabel, QComboBox, QHBoxLayout, QFileDialog, QAction, QMenu,
                             QMessageBox, QScrollArea, QSizePolicy, QTabWidget, QCheckBox, QSpinBox, QFrame,
                             QButtonGroup, QRadioButton, QListWidgetItem, QGridLayout, QDialogButtonBox, QListWidget,
                             QInputDialog, QLineEdit, QDoubleSpinBox,QToolTip,QSplitter)
from PyQt5.QtWidgets import QProgressDialog, QMessageBox, QApplication
from PyQt5.QtWidgets import QHeaderView
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
import datetime as _dt
import traceback
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
from sklearn.metrics import mutual_info_score
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
import seaborn as sns
import joblib
from sklearn.base import clone
try:
    import optuna
    OPTUNA_IMPORT_ERROR = None
except Exception as e:
    optuna = None
    OPTUNA_IMPORT_ERROR = repr(e)

class _ReducerWrappedEstimator:
    """Global wrapper for permutation_importance with optional reducer."""
    def __init__(self, fitted_model, fitted_reducer=None):
        self._model = fitted_model
        self._reducer = fitted_reducer

    def _to_array(self, X):
        try:
            return X.values
        except Exception:
            return X

    def predict(self, X):
        Xv = self._to_array(X)
        if self._reducer is not None:
            Xv = self._reducer.transform(Xv)
        return self._model.predict(Xv)

    def decision_function(self, X):
        Xv = self._to_array(X)
        if self._reducer is not None:
            Xv = self._reducer.transform(Xv)
        if hasattr(self._model, "decision_function"):
            return self._model.decision_function(Xv)
        raise AttributeError("Wrapped model has no decision_function")

    def predict_proba(self, X):
        Xv = self._to_array(X)
        if self._reducer is not None:
            Xv = self._reducer.transform(Xv)
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(Xv)
        raise AttributeError("Wrapped model has no predict_proba")

    def score(self, X, y):
        Xv = self._to_array(X)
        if self._reducer is not None:
            Xv = self._reducer.transform(Xv)
        return self._model.score(Xv, y)



def resource_path(relative_path):
    """ PyInstaller로 패키징할 때 파일 경로를 반환하는 함수 """
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def _kuquickml_strip_non_feature_columns(columns):
    """Return only real feature columns for prediction-time compatibility checks."""
    ignored = {
        'sample', 'name', 'label', 'prediction', 'class', 'target', 'y'
    }
    cleaned = []
    for col in columns:
        c = str(col).strip()
        if c.lower() not in ignored:
            cleaned.append(c)
    return cleaned


def _kuquickml_get_saved_feature_names(bundle, fallback_columns=None):
    """Best-effort retrieval of saved training feature names from a model bundle."""
    feature_names = bundle.get("feature_names")
    if feature_names is None:
        model = bundle.get("model")
        feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = fallback_columns
    if feature_names is None:
        return None
    return _kuquickml_strip_non_feature_columns(list(feature_names))


def _kuquickml_find_missing_required_features(expected_features, loaded_columns):
    """Compare feature names strictly except for leading/trailing whitespace only."""
    loaded_trimmed = [str(c).strip() for c in loaded_columns]
    loaded_set = set(loaded_trimmed)
    missing = [f for f in expected_features if str(f).strip() not in loaded_set]
    return missing, loaded_trimmed

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

        predictionMenu = menubar.addMenu('★5. Prediction')
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
        """Sample 컬럼 제거 + 전부 numeric 강제(에러는 NaN)"""
        if 'Sample' in df.columns:
            df = df.drop(columns=['Sample'])
        df = df.apply(pd.to_numeric, errors='coerce')
        return df

    def _fit_preprocess_train_test(self, X_train_df: pd.DataFrame, X_test_df: pd.DataFrame, y_train):
        """
        학습용 전처리(학습 데이터 기준으로 scaler/reducer fit)
        반환: X_train_used, X_test_used, fitted_scaler, fitted_reducer, feature_names
        """
        # 0) feature 이름 저장 (순서가 가장 중요)
        feature_names = list(X_train_df.columns)

        # 1) scaler (모델마다 따로 fit된 scaler를 갖게 함)
        base_scaler = self.scaler
        scaler = clone(base_scaler) if base_scaler else None

        if scaler:
            X_train_scaled = scaler.fit_transform(X_train_df)
            X_test_scaled = scaler.transform(X_test_df)
        else:
            X_train_scaled = X_train_df.values
            X_test_scaled = X_test_df.values

        # 2) reducer (반드시 scaler 이후에 fit/transform)
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




    def _format_hidden_layers_text(self, value):
        if isinstance(value, str):
            mapping = {
                '50': (50,),
                '100': (100,),
                '150': (150,),
                '50,50': (50, 50),
                '100,50': (100, 50),
                '100,100': (100, 100),
            }
            value = mapping.get(value, value)
        if isinstance(value, (tuple, list)):
            return ",".join(str(int(v)) for v in value)
        return str(value)

    def _get_fixed_optuna_metric(self, task):
        return "accuracy" if str(task).lower() == "classification" else "r2"

    def _quantize_optuna_float(self, value, decimals=4, lower=None, upper=None):
        q = round(float(value), int(decimals))
        if lower is not None:
            q = max(float(lower), q)
        if upper is not None:
            q = min(float(upper), q)
        return float(q)

    def _display_optuna_value(self, v):
        try:
            if isinstance(v, str):
                hidden_layer_map = {
                    '50': '(50,)',
                    '100': '(100,)',
                    '150': '(150,)',
                    '50,50': '(50, 50)',
                    '100,50': '(100, 50)',
                    '100,100': '(100, 100)',
                }
                return hidden_layer_map.get(v, v)
            if isinstance(v, (float, np.floating)):
                fv = float(v)
                if fv == 0:
                    return "0.0000"
                if abs(fv) >= 0.0001:
                    return f"{fv:.4f}"
                return f"{fv:.4g}"
            return str(v)
        except Exception:
            return str(v)

    def _get_optuna_search_range_text(self, model_name, task):
        model_name = str(model_name).upper()
        task = str(task).lower()
        lines = [
            "Automatic tuning settings",
            "- Trials: 30",
            "- Cross-validation: 5-fold",
            f"- Optimization metric: {'accuracy' if task == 'classification' else 'R²'}",
            "",
            "Search range for selected model",
        ]
        if model_name == "KNN":
            lines += [
                "- Reducer: None, PCA, LDA, NCA",
                "- n_neighbors: 1 to 16",
            ]
        elif model_name == "MLP":
            lines += [
                "- hidden_layer_sizes: (50,), (100,), (150,), (50,50), (100,50), (100,100)",
                "- alpha: 1e-6 to 1e-1 (log scale)",
                "- solver: adam, sgd, lbfgs",
                "- activation: relu, tanh, logistic",
                "- learning_rate_init: 1e-4 to 1e-1 (log scale)",
            ]
        elif model_name == "RF":
            lines += [
                "- max_depth: 2 to 30",
                "- n_estimators: 20 to 400 (step 20)",
                "- min_samples_leaf: 1 to 10",
                "- min_samples_split: 2 to 20",
            ]
        elif model_name == "SVM" and task == "classification":
            lines += [
                "- Reducer: None, PCA, LDA",
                "- multiclass strategy: One-vs-Rest, One-vs-One",
                "- C: 1e-3 to 1e3 (log scale)",
                "- kernel: linear, rbf, poly, sigmoid",
            ]
        elif model_name == "SVM" and task == "regression":
            lines += [
                "- Reducer: None, PCA, LDA, NCA",
                "- C: 1e-3 to 1e3 (log scale)",
                "- kernel: linear, rbf, poly, sigmoid",
                "- epsilon: 1e-3 to 1.0 (log scale)",
            ]
        lines += ["", "Only parameters available in KUickML are tuned automatically."]
        return "\n".join(lines)

    def _build_optuna_panel(self, model_name, task, button_text, callback, attr_name):
        frame = QFrame()
        frame.setFrameShape(QFrame.Box)
        frame.setFrameShadow(QFrame.Sunken)
        panel_layout = QVBoxLayout(frame)

        checkbox = QCheckBox("Automatic hyperparameter tuning (Optuna)")
        checkbox.setChecked(False)
        panel_layout.addWidget(checkbox)

        summary = QLabel("Fixed settings: 30 trials · 5-fold CV · accuracy / R²")
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #444; padding: 2px 0 4px 0;")
        summary.setVisible(False)
        panel_layout.addWidget(summary)

        button_row = QHBoxLayout()
        range_btn = QPushButton("View search range")
        range_btn.setVisible(False)
        range_btn.clicked.connect(lambda: self._show_optuna_search_range_dialog(model_name, task))
        button_row.addWidget(range_btn)

        btn = QPushButton(button_text)
        btn.setEnabled(False)
        btn.clicked.connect(callback)
        button_row.addWidget(btn)
        panel_layout.addLayout(button_row)

        checkbox.toggled.connect(summary.setVisible)
        checkbox.toggled.connect(range_btn.setVisible)
        checkbox.toggled.connect(btn.setEnabled)

        setattr(self, f"{attr_name}CheckBox", checkbox)
        setattr(self, f"{attr_name}SummaryLabel", summary)
        setattr(self, f"{attr_name}RangeButton", range_btn)
        setattr(self, f"{attr_name}Button", btn)
        return frame

    def _show_optuna_search_range_dialog(self, model_name, task):
        QMessageBox.information(
            self,
            f"Search Range - {model_name} {task}",
            self._get_optuna_search_range_text(model_name, task)
        )

    def _update_optuna_info_label(self, model_name, task, attr_name):
        summary = getattr(self, f"{attr_name}SummaryLabel", None)
        if summary is not None:
            summary.setText("Fixed settings: 30 trials · 5-fold CV · accuracy / R²")

    def _make_regression_supervision_labels(self, y, n_bins=3):
        y_arr = np.asarray(y).ravel()
        if len(np.unique(y_arr)) < 2:
            return np.zeros_like(y_arr, dtype=int)
        bins = min(n_bins, max(2, len(np.unique(y_arr))))
        try:
            cat = pd.qcut(y_arr, q=bins, duplicates='drop', labels=False)
            arr = np.asarray(cat, dtype=int)
        except Exception:
            try:
                cat = pd.cut(y_arr, bins=bins, duplicates='drop', labels=False)
                arr = np.asarray(cat, dtype=int)
            except Exception:
                arr = np.zeros_like(y_arr, dtype=int)
        arr = np.nan_to_num(arr, nan=0).astype(int)
        if len(np.unique(arr)) < 2:
            median = np.nanmedian(y_arr)
            arr = (y_arr >= median).astype(int)
        return arr

    def _fit_transform_with_reducer_name(self, reducer_name, X_train, X_test, y_train, task):
        reducer_name = str(reducer_name or 'None')
        X_train_arr = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
        X_test_arr = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
        if reducer_name == 'None':
            return X_train_arr, X_test_arr, None

        n_features = int(X_train_arr.shape[1]) if X_train_arr.ndim >= 2 else 1

        if reducer_name == 'PCA':
            n_comp = max(1, min(2, n_features))
            reducer = PCA(n_components=n_comp)
            reducer.fit(X_train_arr)
        elif reducer_name == 'LDA':
            y_supervision = y_train if str(task).lower() == 'classification' else self._make_regression_supervision_labels(y_train)
            y_supervision = np.asarray(y_supervision).ravel()
            n_classes = len(np.unique(y_supervision))
            # LDA can use at most min(n_features, n_classes - 1) components.
            # In binary classification and in regression folds that collapse to 2 pseudo-classes,
            # force LDA to 1D automatically. If only one supervision class remains,
            # LDA cannot be fitted, so fall back to 1D PCA.
            if n_classes < 2:
                n_comp = max(1, min(1, n_features))
                reducer = PCA(n_components=n_comp)
                reducer.fit(X_train_arr)
            else:
                if n_classes <= 2:
                    n_comp = 1
                else:
                    max_comp = min(n_features, n_classes - 1)
                    n_comp = max(1, max_comp)
                reducer = LDA(n_components=n_comp)
                reducer.fit(X_train_arr, y_supervision)
        elif reducer_name == 'NCA':
            y_supervision = y_train if str(task).lower() == 'classification' else self._make_regression_supervision_labels(y_train)
            n_comp = max(1, min(2, n_features))
            reducer = NCA(n_components=n_comp, max_iter=100, tol=1e-5, random_state=42)
            reducer.fit(X_train_arr, y_supervision)
        else:
            raise ValueError(f'Unknown reducer: {reducer_name}')

        return reducer.transform(X_train_arr), reducer.transform(X_test_arr), reducer

    def _run_optuna_cv(self, model_name, task):
        if optuna is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Optuna Unavailable")
            msg.setText("Optuna could not be imported.")
            msg.setInformativeText("Please check the detailed error below.")
            msg.setDetailedText(OPTUNA_IMPORT_ERROR or "Unknown import error")
            msg.exec_()
            return None

        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)

        metric = self._get_fixed_optuna_metric(task)
        direction = 'maximize'
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if str(task).lower() == 'classification' else KFold(n_splits=5, shuffle=True, random_state=42)

        model_name_u = str(model_name).upper()

        def suggest_params(trial):
            if model_name_u == 'KNN':
                return {
                    'reducer': trial.suggest_categorical('reducer', ['None', 'PCA', 'LDA', 'NCA']),
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 16),
                }
            if model_name_u == 'MLP':
                return {
                    'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', ['50', '100', '150', '50,50', '100,50', '100,100']),
                    'alpha': self._quantize_optuna_float(trial.suggest_float('alpha', 1e-6, 1e-1, log=True), lower=1e-6, upper=1e-1),
                    'solver': trial.suggest_categorical('solver', ['adam', 'sgd', 'lbfgs']),
                    'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
                    'learning_rate_init': self._quantize_optuna_float(trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True), lower=1e-4, upper=1e-1),
                }
            if model_name_u == 'RF':
                return {
                    'max_depth': trial.suggest_int('max_depth', 2, 30),
                    'n_estimators': trial.suggest_int('n_estimators', 20, 400, step=20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                }
            if model_name_u == 'SVM' and str(task).lower() == 'classification':
                return {
                    'reducer': trial.suggest_categorical('reducer', ['None', 'PCA', 'LDA']),
                    'multiclass_strategy': trial.suggest_categorical('multiclass_strategy', ['One-vs-Rest SVM', 'One-vs-One SVM']),
                    'C': self._quantize_optuna_float(trial.suggest_float('C', 1e-3, 1e3, log=True), lower=1e-3, upper=1e3),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                }
            if model_name_u == 'SVM' and str(task).lower() == 'regression':
                return {
                    'reducer': trial.suggest_categorical('reducer', ['None', 'PCA', 'LDA', 'NCA']),
                    'C': self._quantize_optuna_float(trial.suggest_float('C', 1e-3, 1e3, log=True), lower=1e-3, upper=1e3),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                    'epsilon': self._quantize_optuna_float(trial.suggest_float('epsilon', 1e-3, 1.0, log=True), lower=1e-3, upper=1.0),
                }
            raise ValueError(f'Unsupported model_name: {model_name_u}')

        def build_estimator(params):
            if model_name_u == 'KNN':
                if str(task).lower() == 'classification':
                    return KNeighborsClassifier(n_neighbors=params['n_neighbors'])
                return KNeighborsRegressor(n_neighbors=params['n_neighbors'])
            if model_name_u == 'MLP':
                hidden_layer_map = {
                    '50': (50,),
                    '100': (100,),
                    '150': (150,),
                    '50,50': (50, 50),
                    '100,50': (100, 50),
                    '100,100': (100, 100),
                }
                common = dict(
                    hidden_layer_sizes=hidden_layer_map.get(params['hidden_layer_sizes'], params['hidden_layer_sizes']),
                    alpha=float(params['alpha']),
                    solver=params['solver'],
                    activation=params['activation'],
                    learning_rate_init=float(params['learning_rate_init']),
                    max_iter=int(self.max_iter_input.value()) if hasattr(self, 'max_iter_input') else 1000,
                    random_state=int(self.random_state_input.value()) if hasattr(self, 'random_state_input') else 42,
                )
                return MLPClassifier(**common) if str(task).lower() == 'classification' else MLPRegressor(**common)
            if model_name_u == 'RF':
                common = dict(
                    max_depth=int(params['max_depth']),
                    n_estimators=int(params['n_estimators']),
                    min_samples_leaf=int(params['min_samples_leaf']),
                    min_samples_split=int(params['min_samples_split']),
                    random_state=42,
                    n_jobs=-1,
                )
                return RandomForestClassifier(**common) if str(task).lower() == 'classification' else RandomForestRegressor(**common)
            if model_name_u == 'SVM' and str(task).lower() == 'classification':
                base = SVC(kernel=params['kernel'], C=float(params['C']))
                if params['multiclass_strategy'] == 'One-vs-One SVM':
                    return OneVsOneClassifier(base)
                return OneVsRestClassifier(base)
            if model_name_u == 'SVM' and str(task).lower() == 'regression':
                return SVR(kernel=params['kernel'], C=float(params['C']), epsilon=float(params['epsilon']))
            raise ValueError(f'Unsupported model_name: {model_name_u}')

        def objective(trial):
            params = suggest_params(trial)
            scores = []
            failed_folds = 0
            split_iter = splitter.split(X_train_numeric, y_train) if str(task).lower() == 'classification' else splitter.split(X_train_numeric)
            for train_idx, valid_idx in split_iter:
                X_tr = X_train_numeric.iloc[train_idx]
                X_va = X_train_numeric.iloc[valid_idx]
                y_tr = y_train[train_idx]
                y_va = y_train[valid_idx]
                reducer_name = params.get('reducer', 'None')
                try:
                    X_tr_used, X_va_used, _ = self._fit_transform_with_reducer_name(reducer_name, X_tr, X_va, y_tr, task)
                    model = build_estimator(params)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
                        model.fit(X_tr_used, y_tr)
                    preds = model.predict(X_va_used)
                    score = accuracy_score(y_va, preds) if metric == 'accuracy' else r2_score(y_va, preds)
                    if np.isnan(score) or np.isinf(score):
                        failed_folds += 1
                        continue
                    scores.append(float(score))
                except Exception:
                    failed_folds += 1
                    continue
            if not scores:
                return -1e9
            mean_score = float(np.mean(scores))
            if failed_folds:
                mean_score -= failed_folds * 1e3
            return mean_score

        progress = QProgressDialog(f"Automatic tuning in progress...\nModel: {model_name_u}\nMetric: {metric}", None, 0, 30, self)
        progress.setWindowTitle('Optuna Tuning')
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        QApplication.processEvents()

        best_state = {'score': None, 'params': None}

        def callback(study, trial):
            progress.setValue(len(study.trials))
            best = study.best_trial if len(study.trials) > 0 else None
            if best is not None:
                best_state['score'] = best.value
                best_state['params'] = dict(best.params)
                progress.setLabelText(
                    f"Automatic tuning in progress...\nModel: {model_name_u}\nMetric: {metric}\nTrial {len(study.trials)} / 30\nCurrent best score: {best.value:.4f}"
                )
            QApplication.processEvents()

        try:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction=direction, sampler=sampler)
            study.optimize(objective, n_trials=30, callbacks=[callback], show_progress_bar=False)
            best_params = dict(study.best_trial.params)
            best_score = float(study.best_trial.value)
            trials_df = study.trials_dataframe()
            return {'best_params': best_params, 'best_score': best_score, 'metric': metric, 'trials_df': trials_df}
        except Exception as e:
            QMessageBox.warning(self, 'Optuna Error', f'Automatic tuning failed\n{e}')
            return None
        finally:
            progress.close()

    def _show_optuna_results_dialog(self, model_name, task, tuning_result):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Optuna Results - {model_name} {task}")
        dialog.setGeometry(100, 100, 800, 500)
        layout = QVBoxLayout(dialog)

        def _fmt_optuna_value(v):
            return self._display_optuna_value(v)

        best_score = tuning_result.get('best_score')
        metric = tuning_result.get('metric')
        best_params = tuning_result.get('best_params', {})
        summary_html = (
            f"<b>Best CV {metric}: {best_score:.4f}</b><br><br><b>Best parameters:</b><br>"
            + "<br>".join(f"- {k}: {_fmt_optuna_value(v)}" for k, v in best_params.items())
        )
        summary = QLabel()
        summary.setTextFormat(Qt.RichText)
        summary.setText(summary_html)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QTableWidget(dialog)
        df = tuning_result.get('trials_df')
        if df is not None and not df.empty:
            show_cols = [c for c in df.columns if c in ('number', 'value') or c.startswith('params_')]
            df_show = df[show_cols].copy() if show_cols else df.copy()
            table.setRowCount(df_show.shape[0])
            table.setColumnCount(df_show.shape[1])
            table.setHorizontalHeaderLabels([str(c) for c in df_show.columns])
            for i in range(df_show.shape[0]):
                for j in range(df_show.shape[1]):
                    table.setItem(i, j, QTableWidgetItem(_fmt_optuna_value(df_show.iloc[i, j])))
            table.resizeColumnsToContents()
        layout.addWidget(table)

        dialog.setLayout(layout)
        dialog.exec_()

    def _apply_optuna_best_params_to_ui(self, model_name, task, params):
        model_name = str(model_name).upper()
        task = str(task).lower()

        def set_reducer(name):
            mapping = {
                'PCA': getattr(self, 'pcaCheckBox', None),
                'LDA': getattr(self, 'ldaCheckBox', None),
                'NCA': getattr(self, 'ncaCheckBox', None),
                'None': getattr(self, 'noneCheckBox', None),
            }
            widget = mapping.get(str(name), mapping.get('None'))
            if widget is not None:
                widget.setChecked(True)

        if model_name == 'KNN':
            if 'n_neighbors' in params and hasattr(self, 'n_neighbors_input'):
                self.n_neighbors_input.setValue(int(params['n_neighbors']))
            set_reducer(params.get('reducer', 'None'))
            return

        if model_name == 'MLP':
            if 'hidden_layer_sizes' in params and hasattr(self, 'hidden_layer_input'):
                self.hidden_layer_input.setText(self._format_hidden_layers_text(params['hidden_layer_sizes']))
            if 'alpha' in params and hasattr(self, 'alpha_input'):
                self.alpha_input.setText(f"{float(params['alpha']):g}")
            if 'solver' in params and hasattr(self, 'solver_input'):
                idx = self.solver_input.findText(str(params['solver']))
                if idx >= 0:
                    self.solver_input.setCurrentIndex(idx)
            if 'activation' in params and hasattr(self, 'activation_input'):
                idx = self.activation_input.findText(str(params['activation']))
                if idx >= 0:
                    self.activation_input.setCurrentIndex(idx)
            if 'learning_rate_init' in params and hasattr(self, 'learning_rate_input'):
                self.learning_rate_input.setText(f"{float(params['learning_rate_init']):g}")
            return

        if model_name == 'RF':
            mapping = {
                'max_depth': 'Max Depth:',
                'n_estimators': 'N Estimators:',
                'min_samples_leaf': 'Min Samples Leaf:',
                'min_samples_split': 'Min Samples Split:',
            }
            for key, widget_key in mapping.items():
                if key in params and hasattr(self, 'param_inputs') and widget_key in self.param_inputs:
                    self.param_inputs[widget_key].setValue(int(params[key]))
            return

        if model_name == 'SVM':
            if task == 'classification' and 'multiclass_strategy' in params and hasattr(self, 'svm_type'):
                idx = self.svm_type.findText(str(params['multiclass_strategy']))
                if idx >= 0:
                    self.svm_type.setCurrentIndex(idx)
            if 'kernel' in params and hasattr(self, 'kernel_type'):
                idx = self.kernel_type.findText(str(params['kernel']))
                if idx >= 0:
                    self.kernel_type.setCurrentIndex(idx)
            if 'C' in params and hasattr(self, 'c_value'):
                self.c_value.setValue(float(params['C']))
            if 'epsilon' in params and hasattr(self, 'svrEpsilonInput'):
                self.svrEpsilonInput.setValue(float(params['epsilon']))
            set_reducer(params.get('reducer', 'None'))

    def runOptunaKNNClassification(self):
        result = self._run_optuna_cv('KNN', 'classification')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('KNN', 'classification', result['best_params'])
        self._show_optuna_results_dialog('KNN', 'classification', result)
        self.createClassificationModel()

    def runOptunaKNNRegression(self):
        result = self._run_optuna_cv('KNN', 'regression')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('KNN', 'regression', result['best_params'])
        self._show_optuna_results_dialog('KNN', 'regression', result)
        self.createRegressionModel()

    def runOptunaMLPClassification(self):
        result = self._run_optuna_cv('MLP', 'classification')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('MLP', 'classification', result['best_params'])
        self._show_optuna_results_dialog('MLP', 'classification', result)
        self.createMLPClassificationModel()

    def runOptunaMLPRegression(self):
        result = self._run_optuna_cv('MLP', 'regression')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('MLP', 'regression', result['best_params'])
        self._show_optuna_results_dialog('MLP', 'regression', result)
        self.createMLPRegressionModel()

    def runOptunaRFClassification(self):
        result = self._run_optuna_cv('RF', 'classification')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('RF', 'classification', result['best_params'])
        self._show_optuna_results_dialog('RF', 'classification', result)
        self.createRFClassificationModel()

    def runOptunaRFRegression(self):
        result = self._run_optuna_cv('RF', 'regression')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('RF', 'regression', result['best_params'])
        self._show_optuna_results_dialog('RF', 'regression', result)
        self.createRFRegressionModel()

    def runOptunaSVMClassification(self):
        result = self._run_optuna_cv('SVM', 'classification')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('SVM', 'classification', result['best_params'])
        self._show_optuna_results_dialog('SVM', 'classification', result)
        self.createSVMModel()

    def runOptunaSVMRegression(self):
        result = self._run_optuna_cv('SVM', 'regression')
        if result is None:
            return
        self._apply_optuna_best_params_to_ui('SVM', 'regression', result['best_params'])
        self._show_optuna_results_dialog('SVM', 'regression', result)
        self.createSVMRegressionModel()

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
        desc = QLabel("SVM의 분류 전략을 선택합니다.<br>"
                      "<b>One-vs-Rest</b>: 한 클래스를 나머지 전부와 비교 (빠름)<br>"
                      "<b>One-vs-One</b>: 클래스 간 모든 조합을 학습 (정확도↑)")
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
        desc = QLabel("커널은 입력 데이터를 고차원 공간으로 변환하는 함수입니다.<br>"
                      "<b>linear</b>: 선형 경계, 빠름<br>"
                      "<b>poly</b>: 다항식 커널<br>"
                      "<b>rbf</b>: 가우시안 기반, 비선형에 강함<br>"
                      "<b>sigmoid</b>: 신경망 유사 특성")
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
        self.c_value.setRange(0.01, 2000.0)
        self.c_value.setValue(1.0)
        self.c_value.setSingleStep(0.01)
        desc = QLabel("C 값은 오류 허용 정도를 조절하는 규제(regularization) 강도입니다.<br>"
                      "작으면 일반화 ↑ (느리지만 안정), 크면 훈련 정확도 ↑ (과적합 위험).")
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
        desc = QLabel("무작위 초기화를 제어하는 시드 값입니다.<br>같은 결과를 재현하려면 동일한 값을 유지하세요.")
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
        desc = QLabel("Epsilon-insensitive 영역 폭입니다. 작을수록 데이터에 민감해질 수 있습니다.")
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

        self.pcaCheckBox.setChecked(True)  # PCA를 기본값으로 설정
        self.dimensionalityGroup = QButtonGroup()
        for checkbox in [self.pcaCheckBox, self.ldaCheckBox, self.ncaCheckBox, self.noneCheckBox]:
            frame_reducer_layout.addWidget(checkbox)
            self.dimensionalityGroup.addButton(checkbox)
        self.dimensionalityGroup.setExclusive(True)
        desc = QLabel("차원 축소는 데이터를 저차원 공간으로 투영하여 계산 효율과 시각화를 돕습니다.<br>"
                      "<b>PCA</b>: 주성분 분석 (일반적)<br>"
                      "<b>LDA</b>: 클래스 간 분리 최적화 (SVR 회귀에서는 y를 구간화해 사용)<br>"
                      "<b>NCA</b>: 거리 기반 분류에 적합 (SVR 회귀에서는 y를 구간화해 사용)<br>"
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

        # 5-fold CV buttons
        self.svmCvClassButton = QPushButton("5-Fold CV (SVM Classification)")
        self.svmCvClassButton.clicked.connect(self.runSVMClassificationCV)
        buttons_layout.addWidget(self.svmCvClassButton)

        self.svmCvRegButton = QPushButton("5-Fold CV (SVM Regression)")
        self.svmCvRegButton.clicked.connect(self.runSVMRegressionCV)
        buttons_layout.addWidget(self.svmCvRegButton)
        layout.addLayout(buttons_layout)
        layout.addWidget(self._build_optuna_panel("SVM", "classification", "Tune and Create SVM Classification Model", self.runOptunaSVMClassification, "svmOptunaClass"))
        layout.addWidget(self._build_optuna_panel("SVM", "regression", "Tune and Create SVM Regression Model", self.runOptunaSVMRegression, "svmOptunaReg"))

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
        loading = QProgressDialog(
            "모델 생성 중입니다...\n\n컴퓨터 사양에 따라 몇 분 정도 소요될 수 있습니다.",
            None, 0, 0, self
        )
        loading.setWindowTitle("SVM 모델 생성 중")
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

            # reducer (split 단계에서 이미 scaled/raw가 결정되므로 여기서 추가 스케일링 금지)
            selected = self.getSelectedDimReductionMethod()
            reducer = None
            method_name = "None"
            if selected:
                method_name, reducer = selected
            if reducer is not None:
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
            if hasattr(X_train_used, "shape") and len(X_train_used.shape) == 2 and X_train_used.shape[1] == 2:
                self.plotScatterWithDecisionBoundary(
                    X_train_used, y_train, X_test_used, y_pred_test, svc_model,
                    f"SVM Scatter Plot with Decision Boundary (kernel={kernel})\nTest accuracy = {overall_accuracy:.3f}"
                )

            # 중요계수: linear + reducer 없음일 때만 (coef_ 기반)
            if kernel == "linear" and reducer is None and hasattr(svc_model, "coef_"):
                self.showImportantCoefficients(X_train_numeric, svc_model)
            else:
                # 비선형 커널이거나, 차원 축소가 적용된 경우: 이유 설명 + permutation 옵션
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

            # ✅ bundle 저장 (예측 탭에서 그대로 사용)
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

            selected_reducer = "None"
            if self.pcaCheckBox.isChecked():
                selected_reducer = "PCA"
            elif self.ldaCheckBox.isChecked():
                selected_reducer = "LDA"
            elif self.ncaCheckBox.isChecked():
                selected_reducer = "NCA"

            X_train_used, X_test_used, reducer = self._fit_transform_with_reducer_name(
                selected_reducer, X_train_numeric, X_test_numeric, y_train, task="regression"
            )

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
                # 비선형 커널이거나, 차원 축소가 적용된 경우: 이유 설명 + permutation 옵션
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


            # 2D 차원축소(PCA) 적용 시, SVR 회귀 예측 표면(Regression boundary plot) 표시
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
            # SVM 탭 UI 위젯명(kuquickml3): self.svm_type, self.kernel_type, self.c_value
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
        """2D reduced space에서 분류 결정경계를 시각화합니다.
        - 배경은 흰색으로 유지합니다.
        - binary SVM의 경우 검은 실선은 decision boundary, 회색 점선은 margin으로 표시합니다.
        - multiclass의 경우 산점도 위주로 표시합니다.
        """
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_facecolor("white")

        x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
        y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1

        grid_res = 400
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]

        unique_labels = np.unique(np.concatenate((y_train, y_pred_test)))
        is_multiclass = len(unique_labels) > 2

        extra_handles = []
        extra_labels = []

        if (not is_multiclass) and hasattr(model, "decision_function"):
            score = model.decision_function(grid).reshape(xx.shape)
            try:
                plt.contour(xx, yy, score, levels=[-1, 1], colors="gray", linestyles="--", linewidths=1.0)
                extra_handles.append(Line2D([0], [0], color="gray", lw=1.0, linestyle="--"))
                extra_labels.append("Margin")
            except Exception:
                pass
            try:
                plt.contour(xx, yy, score, levels=[0], colors="black", linewidths=1.5)
                extra_handles.append(Line2D([0], [0], color="black", lw=1.5))
                extra_labels.append("Decision boundary")
            except Exception:
                pass
        elif (not is_multiclass) and hasattr(model, "predict_proba"):
            proba = model.predict_proba(grid)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                p1 = proba[:, 1].reshape(xx.shape)
                try:
                    plt.contour(xx, yy, p1, levels=[0.5], colors="black", linewidths=1.5)
                    extra_handles.append(Line2D([0], [0], color="black", lw=1.5))
                    extra_labels.append("Decision boundary")
                except Exception:
                    pass
        else:
            try:
                Z = model.predict(grid).reshape(xx.shape)
                boundary_levels = np.unique(Z)
                if boundary_levels.size > 1:
                    mid_levels = (boundary_levels[:-1] + boundary_levels[1:]) / 2.0
                    plt.contour(xx, yy, Z, levels=mid_levels, colors="gray", linewidths=0.8, alpha=0.8)
                    extra_handles.append(Line2D([0], [0], color="gray", lw=0.8))
                    extra_labels.append("Class boundary")
            except Exception:
                pass

        label_colors = {
            label: "#{:02x}{:02x}{:02x}".format(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for label in unique_labels
        }

        legend_widget = getattr(self, "legendNameInput", None)
        legend_name = legend_widget.text() if (legend_widget is not None and legend_widget.text()) else "Label"

        for label in unique_labels:
            train_indices = np.where(y_train == label)[0]
            test_indices = np.where(y_pred_test == label)[0]
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

        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles + extra_handles, labels + extra_labels)
        legend.set_draggable(True)
        plt.title(title, fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.xlabel("Component 1", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.ylabel("Component 2", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
        plt.show()



    def plotScatterWithRegressionSurface(self, X_train_reduced, y_train, X_test_reduced, y_test, model, title):
        """2D reduced space에서 회귀 예측값의 contour를 흰 배경 위에 표시합니다.
        - 배경은 흰색으로 유지합니다.
        - contour 선은 같은 predicted value를 의미합니다.
        """
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.set_facecolor("white")

        x_min, x_max = X_train_reduced[:, 0].min() - 1, X_train_reduced[:, 0].max() + 1
        y_min, y_max = X_train_reduced[:, 1].min() - 1, X_train_reduced[:, 1].max() + 1

        grid_res = 400
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res)
        )

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ctr = plt.contour(xx, yy, Z, levels=20, colors="gray", linewidths=0.7, alpha=0.7)
        try:
            plt.clabel(ctr, inline=True, fontsize=8, fmt="%.2f")
        except Exception:
            pass

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
        # Sort by signed importance descending so larger positive values appear first.
        # Negative values are shown lower in the table instead of sorting by absolute magnitude.
        importance = importance.sort_values(by="Importance", ascending=False)

        dialog = QDialog(self)
        dialog.setWindowTitle("Feature Importances")
        dialog.setGeometry(100, 100, 600, 400)

        dialog_layout = QVBoxLayout(dialog)

        info = QLabel(
            "Feature importance는 선형 SVM/SVR의 계수(coef_)를 사용해 표시합니다.\n"
            "절댓값이 클수록 영향이 크며, 부호(+/−)는 예측 방향(증가/감소)을 의미합니다."
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
        _kuquickml_enable_copyable_table(table)
        dialog_layout.addWidget(table)

        dialog.setLayout(dialog_layout)
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()


    def showSVMImportanceUnavailable(self, kernel, reducer, X_test, y_test, model, feature_names, title_prefix="SVM", task="classification"):
        """feature importance를 coef_로 표시할 수 없는 경우, 이유를 안내하고
        가능하면 permutation importance를 계산해 보여줍니다.

        - X_test는 '원본 feature 공간'(차원축소/변환 전) DataFrame을 기대합니다.
        - reducer가 있으면 permutation 중요도 계산 시에도 '원본 feature'를 섞되,
          예측은 reducer.transform을 거쳐 학습된 모델에 전달합니다.
        """

        class _ReducerWrappedEstimator:
            """permutation_importance용 래퍼: 원본 X -> (reducer) -> model.predict"""
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
            reasons.append("선택한 커널이 비선형 커널이어서 입력 feature에 대한 계수(coef_)가 정의되지 않습니다.")
        if reducer is not None:
            reasons.append("차원 축소가 적용되어 원본 feature가 변환되었기 때문에 원본 feature 기준 coef_ 중요도를 만들 수 없습니다.")

        reason_text = "\n".join(f"- {r}" for r in reasons) if reasons else "- coef_ 기반 중요도를 계산할 수 없습니다."

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Feature Importance")
        msg_box.setText(
            f"{title_prefix}에서 Feature importance를 coef_로 표시할 수 없습니다.\n\n"
            f"{reason_text}\n\n"
            "대신, permutation importance(특성을 섞었을 때 점수가 얼마나 떨어지는지)를 계산해 표시할 수 있습니다.\n"
            "(Permutation 중요도는 계산량이 많을 수 있습니다.)"
        )

        perm_btn = msg_box.addButton("Permutation 중요도 계산", QMessageBox.ActionRole)
        close_btn = msg_box.addButton("닫기", QMessageBox.RejectRole)
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
                X_test,  # 원본 feature 공간에서 섞기
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

        self.mlpCvClassButton = QPushButton("5-Fold CV (MLP Classification)")
        self.mlpCvClassButton.clicked.connect(self.runMLPClassificationCV)
        buttons_layout.addWidget(self.mlpCvClassButton)

        self.mlpCvRegButton = QPushButton("5-Fold CV (MLP Regression)")
        self.mlpCvRegButton.clicked.connect(self.runMLPRegressionCV)
        buttons_layout.addWidget(self.mlpCvRegButton)

        # 좌우 배치
        main_columns = QHBoxLayout()
        main_columns.addLayout(left_col)
        main_columns.addSpacing(20)
        main_columns.addLayout(right_col)

        layout.addLayout(main_columns)
        layout.addSpacing(15)
        layout.addLayout(buttons_layout)
        layout.addWidget(self._build_optuna_panel("MLP", "classification", "Tune and Create MLP Classification Model", self.runOptunaMLPClassification, "mlpOptunaClass"))
        layout.addWidget(self._build_optuna_panel("MLP", "regression", "Tune and Create MLP Regression Model", self.runOptunaMLPRegression, "mlpOptunaReg"))

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

        # split 단계에서 raw/scaled 결정났으니 여기서 추가 스케일링 금지
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

        # ✅ bundle 저장
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

        # ✅ bundle 저장
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
        dialog.setGeometry(100, 100, 600, 400)  # 창 크기를 조금 더 크게 설정

        dialog_layout = QVBoxLayout(dialog)

        info_label = QLabel(
            "<b>Feature importance 산출 방식</b><br>"
            "이 화면의 MLP 중요도는 scikit-learn의 <code>permutation_importance</code>를 사용해 계산됩니다.<br>"
            "특정 feature의 값 순서를 무작위로 섞었을 때 모델 성능이 얼마나 감소하는지(=성능 감소량)로 중요도를 정의합니다.<br>"
            "값이 클수록 더 중요하며, 0에 가깝거나 음수인 경우 영향이 작거나 잡음일 수 있습니다."
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
        _kuquickml_enable_copyable_table(table)
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

        self.rfCvClassButton = QPushButton("5-Fold CV (RF Classification)")
        self.rfCvClassButton.clicked.connect(self.runRFClassificationCV)
        buttons_layout.addWidget(self.rfCvClassButton)

        self.rfCvRegButton = QPushButton("5-Fold CV (RF Regression)")
        self.rfCvRegButton.clicked.connect(self.runRFRegressionCV)
        buttons_layout.addWidget(self.rfCvRegButton)
        layout.addLayout(buttons_layout)
        layout.addWidget(self._build_optuna_panel("RF", "classification", "Tune and Create RF Classification Model", self.runOptunaRFClassification, "rfOptunaClass"))
        layout.addWidget(self._build_optuna_panel("RF", "regression", "Tune and Create RF Regression Model", self.runOptunaRFRegression, "rfOptunaReg"))

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

    # ------------------------------
    # 5-fold CV runners (RF)
    # ------------------------------
    def runRFClassificationCV(self):
        try:
            # RF 탭에서 생성한 파라미터 스핀박스들은 self.param_inputs에 저장됩니다.
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

        # ✅ bundle 저장
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

        # ✅ bundle 저장
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
        var_imp_info = QLabel('변수 중요도는 scikit-learn RandomForest의 `feature_importances_` 값을 그대로 표시합니다. 이는 트리 분할에서 평균적으로 불순도(impurity)를 얼마나 줄였는지(=MDI, 흔히 “Gini importance”)를 기반으로 계산됩니다. 값이 클수록 모델이 해당 feature로 더 자주/더 크게 분할해 예측에 기여했음을 의미합니다. ')
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
        _kuquickml_enable_copyable_table(var_importance_table)
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
        var_imp_info = QLabel('변수 중요도는 scikit-learn RandomForest의 `feature_importances_` 값을 그대로 표시합니다. 이는 트리 분할에서 평균적으로 불순도(impurity)를 얼마나 줄였는지(=MDI, 흔히 “Gini importance”)를 기반으로 계산됩니다. 값이 클수록 모델이 해당 feature로 더 자주/더 크게 분할해 예측에 기여했음을 의미합니다. ')
        var_imp_info.setWordWrap(True)
        dialog_layout.addWidget(var_imp_info)

        # Confusion matrix table
        cm_display_df = cm_df.copy()
        try:
            total_row = {col: '' for col in cm_display_df.columns}
            if 'Prediction Accuracy (%)' in cm_display_df.columns:
                total_row['Prediction Accuracy (%)'] = round(float(accuracy) * 100.0, 3)
            elif cm_display_df.shape[1] > 0:
                total_row[cm_display_df.columns[-1]] = round(float(accuracy) * 100.0, 3)
            cm_display_df.loc['Total Prediction Accuracy (%)'] = total_row
        except Exception:
            cm_display_df = cm_df.copy()

        cm_table = QTableWidget(dialog)
        cm_table.setRowCount(cm_display_df.shape[0])
        cm_table.setColumnCount(cm_display_df.shape[1])
        cm_table.setHorizontalHeaderLabels([str(col) for col in cm_display_df.columns])
        cm_table.setVerticalHeaderLabels([str(idx) for idx in cm_display_df.index])

        for i in range(cm_display_df.shape[0]):
            for j in range(cm_display_df.shape[1]):
                value = cm_display_df.iloc[i, j]
                if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
                    text = f"{float(value):.3f}"
                else:
                    text = "" if pd.isna(value) else str(value)
                item = QTableWidgetItem(text)
                cm_table.setItem(i, j, item)

        cm_table.resizeColumnsToContents()
        _kuquickml_enable_copyable_table(cm_table)
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
        _kuquickml_enable_copyable_table(var_importance_table)
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
            unknown_df = self.unknown_data.copy()

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

            # 모델이 실제로 요구하는 feature 이름만 정답으로 사용한다.
            # 비교에서는 앞뒤 공백만 무시하고, 나머지 차이(.1 등)는 다른 feature로 간주한다.
            if feature_names is not None:
                saved_feature_names = [str(c).strip() for c in list(feature_names)]
            elif hasattr(model, "feature_names_in_"):
                saved_feature_names = [str(c).strip() for c in list(model.feature_names_in_)]
            else:
                saved_feature_names = None

            # Sample 컬럼은 모델이 실제로 요구하지 않을 때만 identifier로 취급한다.
            if saved_feature_names is not None and 'Sample' in unknown_df.columns and 'Sample' not in saved_feature_names:
                sample_series = unknown_df['Sample']
                unknown_df = unknown_df.drop(columns=['Sample'])
            else:
                sample_series = pd.Series([f"Sample {i + 1}" for i in range(len(unknown_df))])

            # warning 비교와 실제 reindex가 같은 기준을 쓰도록 열 이름 공백만 정리
            unknown_df = unknown_df.copy()
            unknown_df.columns = [str(c).strip() for c in unknown_df.columns]

            # numeric 강제
            unknown_df = unknown_df.apply(pd.to_numeric, errors='coerce')
            data_to_scale = unknown_df

            # ③ 학습 모델이 기대하는 feature 중 loaded CSV에 없는 항목만 점검
            if saved_feature_names is not None:
                unknown_feature_names = [str(c).strip() for c in list(data_to_scale.columns)]

                missing_features = [f for f in saved_feature_names if f not in unknown_feature_names]
                if missing_features:
                    preview = ", ".join(map(str, missing_features[:10]))
                    if len(missing_features) > 10:
                        preview += ", ..."

                    warning_lines = [
                        "Some required feature columns used to train the model were not found in the loaded CSV.",
                        "",
                        "Leading/trailing spaces are ignored, but all other name differences are treated as different features.",
                        "For example, 'Butyl angelate' and 'Butyl angelate .1' are treated as different columns.",
                        "",
                        f"Required feature count: {len(saved_feature_names)}",
                        f"Loaded column count: {len(unknown_feature_names)}",
                        "",
                        f"Missing required features ({len(missing_features)}): {preview}",
                        "",
                        "Extra columns are allowed and column order will be aligned automatically."
                    ]
                    QMessageBox.warning(self, "Feature Mismatch", "\n".join(warning_lines))


                data_to_scale = data_to_scale.reindex(columns=saved_feature_names)
                if data_to_scale.isnull().any().any():
                    data_to_scale = data_to_scale.fillna(0)
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

            # 1) unknown 최종 입력(스케일/리듀서 적용 후)
            incoming = data_used.values if hasattr(data_used, "values") else np.asarray(data_used)

            # 2) train도 "저장된 scaler/reducer"로 똑같이 전처리해서 비교
            X_train_df = pd.read_csv(resource_path("Temp/X_train.csv"))

            # Sample 제거 + numeric 강제 + NaN 0
            X_train_df = X_train_df.drop(columns=["Sample"], errors="ignore")
            X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce").fillna(0)

            # feature 순서 맞추기(없으면 0채움됨)
            if feature_names is not None:
                X_train_df = X_train_df.reindex(columns=feature_names).fillna(0)

            # scaler 적용
            if scaler:
                X_train_proc = scaler.transform(X_train_df)
            else:
                X_train_proc = X_train_df.values

            # reducer 적용
            if reducer:
                X_train_proc = reducer.transform(X_train_proc)

            X_train_proc = np.asarray(X_train_proc)

            print("incoming shape:", incoming.shape)
            print("X_train_proc shape:", X_train_proc.shape)

            # 3) 거리 계산: unknown 각 행이 train 중 어떤 행과 가장 가까운지
            d = np.linalg.norm(X_train_proc[None, :, :] - incoming[:, None, :], axis=2)
            min_d = d.min(axis=1)
            argmin = d.argmin(axis=1)

            print("min distance per unknown row:", min_d)
            print("closest train row index:", argmin)

            # 4) 판정: 거의 0이면 동일로 간주
            tol = 1e-9
            same_mask = min_d <= tol
            print("same_mask (min_d<=1e-9):", same_mask)
            print("how many exactly same?:", same_mask.sum(), "/", len(same_mask))

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
                compare_expected = _kuquickml_strip_non_feature_columns(list(self.feature_names))
                compare_loaded = _kuquickml_strip_non_feature_columns(list(unknown_df.columns))
                missing = set(compare_expected) - set(compare_loaded)
                if missing:
                    QMessageBox.warning(self, "Feature Mismatch",
                                        f"The following features are missing in unknown data:\n{', '.join(missing)}")
                    return
                # feature 이름 기준 재정렬
                unknown_df = unknown_df.loc[:, compare_expected]

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
            compare_expected = _kuquickml_strip_non_feature_columns(list(self.feature_names))
            compare_loaded = _kuquickml_strip_non_feature_columns(list(unknown_df.columns))
            missing = set(compare_expected) - set(compare_loaded)
            if missing:
                QMessageBox.warning(self, "Feature Mismatch",
                                    f"The following features are missing in unknown data:\n{', '.join(missing)}")
                return
            unknown_df = unknown_df.loc[:, compare_expected]

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

        # 5-fold CV buttons
        self.knnCvClassButton = QPushButton("5-Fold CV (Classification)")
        self.knnCvClassButton.clicked.connect(self.runKnnClassificationCV)
        buttons_layout.addWidget(self.knnCvClassButton)

        self.knnCvRegButton = QPushButton("5-Fold CV (Regression)")
        self.knnCvRegButton.clicked.connect(self.runKnnRegressionCV)
        buttons_layout.addWidget(self.knnCvRegButton)

        layout.addLayout(buttons_layout)

        layout.addWidget(self._build_optuna_panel("KNN", "classification", "Tune and Create Classification Model", self.runOptunaKNNClassification, "knnOptunaClass"))
        layout.addWidget(self._build_optuna_panel("KNN", "regression", "Tune and Create Regression Model", self.runOptunaKNNRegression, "knnOptunaReg"))

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
        """KNN은 모델 내부 파라미터로 feature importance를 제공하지 않으므로,
        permutation importance(특성을 섞었을 때 점수 하락량)를 사용해 중요도를 계산해 표시합니다.

        - X_eval은 원본 feature 공간(DataFrame)을 기대합니다.
        - reducer가 있으면 내부에서 reducer.transform 후 모델에 전달합니다.
        """
        try:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Feature Importance")
            msg.setText(
                f"{title_prefix}는 feature importance를 모델 내부 값으로 직접 계산하지 않습니다.\n\n"
                "Permutation importance를 사용해 중요도를 계산해 표시할 수 있습니다.\n"
                "(특성을 섞었을 때 성능 점수가 얼마나 떨어지는지로 중요도를 계산)"
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
                f"{title_prefix} Feature importance는 permutation importance로 계산했습니다.\n"
                "각 feature를 섞었을 때 성능 점수가 얼마나 감소하는지를 기준으로 중요도를 산출합니다."
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
            _kuquickml_enable_copyable_table(table)
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
        if reducer is not None:
            reducer.fit(X_train_numeric.values, y_train)
            X_train_embedded = reducer.transform(X_train_numeric.values)
            X_test_embedded = reducer.transform(X_test_numeric.values)
        else:
            X_train_embedded = X_train_numeric.values
            X_test_embedded = X_test_numeric.values

        knn.fit(X_train_embedded, y_train)
        accuracy = knn.score(X_test_embedded, y_test)
        if hasattr(X_train_embedded, "shape") and len(X_train_embedded.shape) == 2 and X_train_embedded.shape[1] == 2:
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
        # ✅ bundle 저장 (Save Model / Load Previous Model / Unknown prediction 통일)
        feature_names = list(X_train_numeric.columns)
        self.showKNNPermutationImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix="KNN Classification", task="classification")


        self.models["KNN Classification"] = {
            "model": knn,
            "scaler": self._get_bundle_scaler(),  # split이 scaled였을 때만 scaler 저장
            "reducer": reducer,  # PCA/LDA/NCA
            "feature_names": feature_names,  # feature 순서 고정
            "label_mapping": self._get_label_mapping()  # 문자열 라벨 복원용(있을 때만)
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
        if reducer is not None:
            reducer.fit(X_train_numeric.values, y_train)
            X_train_embedded = reducer.transform(X_train_numeric.values)
            X_test_embedded = reducer.transform(X_test_numeric.values)
        else:
            X_train_embedded = X_train_numeric.values
            X_test_embedded = X_test_numeric.values

        knn.fit(X_train_embedded, y_train)


        y_pred_train = knn.predict(X_train_embedded)
        y_pred_test = knn.predict(X_test_embedded)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        if hasattr(X_train_embedded, "shape") and len(X_train_embedded.shape) == 2 and X_train_embedded.shape[1] == 2:
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

        # ✅ bundle 저장 (Save Model / Load Previous Model / Unknown prediction 통일)
        feature_names = list(X_train_numeric.columns)
        self.showKNNPermutationImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix="KNN Regression", task="regression")


        self.models["KNN Regression"] = {
            "model": knn,
            "scaler": self._get_bundle_scaler(),  # split이 scaled였을 때만 scaler 저장
            "reducer": reducer,  # PCA/LDA/NCA (None이면 그대로)
            "feature_names": feature_names,
            "label_mapping": None  # 회귀는 라벨 매핑 불필요
        }

        if reducer:
            self.model_reducers["KNN Regression"] = reducer

    def plotResults(self, name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors,
                    score_value=None, score_label=None):

        plt.figure() #KNN 2차원 result scatter plot
        ax = plt.gca()
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


        handles, labels = ax.get_legend_handles_labels()
        contour_handle = Line2D([0], [0], color="gray", lw=0.7, label="Predicted value contour")
        legend = ax.legend(handles + [contour_handle], labels + ["Predicted value contour"])
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
        if hasattr(self, "pcaCheckBox") and self.pcaCheckBox.isChecked():
            return "PCA", PCA(n_components=2)
        elif hasattr(self, "ldaCheckBox") and self.ldaCheckBox.isChecked():
            return "LDA", LDA(n_components=2)
        elif hasattr(self, "ncaCheckBox") and self.ncaCheckBox.isChecked():
            return "NCA", NCA(n_components=2, max_iter=100, tol=1e-5, random_state=42)
        elif hasattr(self, "noneCheckBox") and self.noneCheckBox.isChecked():
            return "None", None
        return None

    def showConfusionMatrix(self, cm_df):
        dialog = QDialog(self)
        dialog.setWindowTitle("Confusion Matrix")
        dialog.setGeometry(100, 100, 520, 340)
        dialog_layout = QVBoxLayout(dialog)

        cm_display_df = cm_df.copy()
        try:
            acc_col = 'Prediction Accuracy (%)'
            if acc_col in cm_display_df.columns:
                base_cols = [c for c in cm_display_df.columns if str(c) != acc_col]
                cm_counts = cm_display_df[base_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=float)
                denom = float(cm_counts.sum())
                total_acc = float(np.trace(cm_counts) / denom * 100.0) if denom > 0 else np.nan
                total_row = {col: '' for col in cm_display_df.columns}
                if not pd.isna(total_acc):
                    total_row[acc_col] = round(total_acc, 3)
                cm_display_df.loc['Total Prediction Accuracy (%)'] = total_row
        except Exception:
            cm_display_df = cm_df.copy()

        table = QTableWidget(dialog)
        table.setRowCount(cm_display_df.shape[0])
        table.setColumnCount(cm_display_df.shape[1])
        table.setHorizontalHeaderLabels([str(col) for col in cm_display_df.columns])
        table.setVerticalHeaderLabels([str(idx) for idx in cm_display_df.index])

        for i in range(cm_display_df.shape[0]):
            for j in range(cm_display_df.shape[1]):
                value = cm_display_df.iloc[i, j]
                col_name = str(cm_display_df.columns[j])
                if pd.isna(value):
                    text = ''
                elif col_name == 'Prediction Accuracy (%)' and isinstance(value, (int, float, np.integer, np.floating)):
                    text = f"{float(value):.3f}"
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    # confusion matrix count columns should display as integers unless they are not integral
                    fval = float(value)
                    text = f"{int(round(fval))}" if abs(fval - round(fval)) < 1e-9 else f"{fval:.3f}"
                else:
                    text = str(value)
                table.setItem(i, j, QTableWidgetItem(text))

        table.resizeColumnsToContents()
        _kuquickml_enable_copyable_table(table)
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
                "CV 안내",
                "CSV 데이터 파일을 먼저 Load한 뒤 Data Split을 진행해주세요."
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
                "분류(Classification)에서 가장 많이 쓰는 5-fold 방식입니다.\n"
                "각 fold마다 클래스 비율이 최대한 비슷하게 유지되도록 나눕니다(불균형 데이터에 유리)."
            ),
            "KFold (general)": (
                "가장 기본적인 5-fold 방식입니다.\n"
                "라벨 비율을 고려하지 않고 데이터 순서대로(또는 shuffle 설정에 따라) 균등 분할합니다."
            ),
            "GroupKFold (grouped samples)": (
                "같은 개체(환자/사용자/샘플ID)에서 나온 데이터가 학습과 검증에 섞이면 누수가 생길 수 있습니다.\n"
                "GroupKFold는 같은 그룹이 한 fold 안에만 들어가도록 분할합니다. (예: patient_id)"
            ),
            "TimeSeriesSplit (time order)": (
                "시계열/시간 순서가 중요한 데이터용 5-fold입니다.\n"
                "과거로 학습하고 미래로 검증하며, shuffle을 사용하지 않습니다."
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
            reducer_name = selected[0] if selected else 'None'
            X_train_used, X_test_used, reducer = self._fit_transform_with_reducer_name(
                reducer_name, X_train_df, X_test_df, y_train, task=task
            )

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

        # CV UI init
        self.cvSplitStrategyCombo.currentIndexChanged.connect(self._update_cv_strategy_ui)
        self._refresh_cv_group_columns()
        self._update_cv_strategy_ui()

        self.dataSplitTab.setLayout(layout)

    def _get_bundle_scaler(self):
        # split에서 scaled 데이터를 썼을 때만 scaler를 함께 저장
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

    def loadCsv(self, checked=False):
        # QAction.triggered가 bool(checked)을 넘기므로 checked 인자를 받아야 함
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "",
            "CSV Files (*.csv);;All Files (*)", options=options
        )
        if not filename:
            return

        try:
            # CsvViewer의 로딩 루틴 사용 (ColumnRoleDialog 포함)
            self.csvViewer.loadCsv(filename)

            # UI: 가이드 숨기고 뷰어 보여주기 (원하면 제거 가능)
            if hasattr(self, "guideWidget"):
                self.guideWidget.hide()
            self.csvViewer.show()

            # Temp 폴더에 splitData가 필요로 하는 파일 저장
            output_dir = resource_path('Temp')
            os.makedirs(output_dir, exist_ok=True)

            # original_X.csv : Sample + Feature들
            if getattr(self.csvViewer, "original_data", None) is not None:
                self.csvViewer.original_data.to_csv(os.path.join(output_dir, "original_X.csv"), index=False)

            # scaled_y.csv : Label만 (이름은 splitData에서 scaled_y.csv로 읽고 있어서 유지)
            if getattr(self.csvViewer, "y", None) is not None:
                pd.DataFrame(self.csvViewer.y, columns=["Label"]).to_csv(os.path.join(output_dir, "scaled_y.csv"),
                                                                         index=False)

            # refresh CV group-column list (if Data Split tab is already created)
            self._refresh_cv_group_columns()

            QMessageBox.information(self, "Load Complete", f"Loaded CSV:\n{os.path.basename(filename)}")

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
        data = pd.read_csv(filename)
        if data.shape[0] == 0:
            QMessageBox.warning(self, "Error", "Loaded CSV has 0 rows.")
            return
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

        # Label 처리 (숫자면 그대로, 문자면 매핑)
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

        # Feature numeric 변환
        for col in feature_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # 컬럼명 통일
        data = data.rename(columns={sample_column_name: 'Sample', label_column_name: 'Label'})

        # 저장
        self.original_data = data[['Sample'] + feature_columns]
        self.X = data[feature_columns].to_numpy()
        self.y = y

        # 표시
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
        self, 'Message', '정말 닫으시겠습니까?',
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



# ==============================
# Save bundle metadata patch
# Stores training/CV metrics together with the model bundle when saving.
# ==============================
try:
    import platform
    import copy
    import datetime as _dt
    import sklearn as _sklearn_pkg
    from sklearn.base import clone as _kuquickml_clone, is_classifier as _kuquickml_is_classifier, is_regressor as _kuquickml_is_regressor
    from sklearn.metrics import precision_score as _kuquickml_precision_score, recall_score as _kuquickml_recall_score, mean_absolute_error as _kuquickml_mae
except Exception:
    platform = None
    copy = None
    _dt = None
    _sklearn_pkg = None


def _kuquickml_safe_auc(y_true, estimator, X_used, task="classification"):
    if task != "classification":
        return None
    try:
        unique_y = np.unique(y_true)
        if len(unique_y) < 2:
            return None
        if hasattr(estimator, "predict_proba"):
            y_score = estimator.predict_proba(X_used)
        elif hasattr(estimator, "decision_function"):
            y_score = estimator.decision_function(X_used)
        else:
            return None

        if len(unique_y) == 2:
            if isinstance(y_score, np.ndarray) and y_score.ndim == 2:
                return float(roc_auc_score(y_true, y_score[:, 1]))
            return float(roc_auc_score(y_true, y_score))

        # multiclass ROC-AUC: One-vs-Rest with macro averaging
        if isinstance(y_score, np.ndarray) and y_score.ndim == 1:
            return None
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _kuquickml_training_metrics(task, estimator, X_used, y_true):
    y_pred = estimator.predict(X_used)
    if task == "regression":
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(_kuquickml_mae(y_true, y_pred)),
        }

    avg = "binary" if len(np.unique(y_true)) == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(_kuquickml_precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(_kuquickml_recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    auc = _kuquickml_safe_auc(y_true, estimator, X_used, task="classification")
    if auc is not None:
        metrics["roc_auc"] = auc
    return metrics


def _kuquickml_test_metrics(task, estimator, X_used, y_true):
    return _kuquickml_training_metrics(task, estimator, X_used, y_true)


def _kuquickml_metric_mean_std(values):
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return {"mean": float(arr[0]), "std": 0.0}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}


def _kuquickml_cv_metrics(self, estimator, reducer, task="classification"):
    X, y = self._get_cv_X_y()
    splitter, groups = self._get_cv_splitter(y, n_splits=5)

    if task == "classification":
        fold_scores = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    else:
        fold_scores = {"r2": [], "mse": [], "rmse": [], "mae": []}

    split_iter = splitter.split(X, y, groups=groups) if groups is not None else splitter.split(X, y)
    for train_idx, test_idx in split_iter:
        X_train_df = X.iloc[train_idx]
        X_test_df = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        fold_reducer = None
        if reducer is not None:
            try:
                fold_reducer = _kuquickml_clone(reducer)
            except Exception:
                fold_reducer = copy.deepcopy(reducer) if copy is not None else None

        if fold_reducer is not None:
            if task == "classification":
                fold_reducer.fit(X_train_df.values, y_train)
            else:
                try:
                    fold_reducer.fit(X_train_df.values, y_train)
                except TypeError:
                    fold_reducer.fit(X_train_df.values)
            X_train_used = fold_reducer.transform(X_train_df.values)
            X_test_used = fold_reducer.transform(X_test_df.values)
        else:
            X_train_used = X_train_df.values
            X_test_used = X_test_df.values

        model = _kuquickml_clone(estimator)
        model.fit(X_train_used, y_train)
        y_pred = model.predict(X_test_used)

        if task == "regression":
            mse = float(mean_squared_error(y_test, y_pred))
            fold_scores["r2"].append(float(r2_score(y_test, y_pred)))
            fold_scores["mse"].append(mse)
            fold_scores["rmse"].append(float(np.sqrt(mse)))
            fold_scores["mae"].append(float(_kuquickml_mae(y_test, y_pred)))
        else:
            avg = "binary" if len(np.unique(y)) == 2 else "macro"
            fold_scores["accuracy"].append(float(accuracy_score(y_test, y_pred)))
            fold_scores["precision"].append(float(_kuquickml_precision_score(y_test, y_pred, average=avg, zero_division=0)))
            fold_scores["recall"].append(float(_kuquickml_recall_score(y_test, y_pred, average=avg, zero_division=0)))
            fold_scores["f1"].append(float(f1_score(y_test, y_pred, average=avg, zero_division=0)))
            auc = _kuquickml_safe_auc(y_test, model, X_test_used, task="classification")
            fold_scores["roc_auc"].append(auc)

    return {k: _kuquickml_metric_mean_std(v) for k, v in fold_scores.items()}


def _kuquickml_infer_task(bundle):
    model = bundle.get("model") if isinstance(bundle, dict) else None
    if model is not None:
        try:
            if _kuquickml_is_classifier(model):
                return "classification"
            if _kuquickml_is_regressor(model):
                return "regression"
        except Exception:
            pass
    label_mapping = bundle.get("label_mapping") if isinstance(bundle, dict) else None
    return "classification" if label_mapping is not None else "regression"


def _kuquickml_get_cv_settings_snapshot(self):
    strategy = self.cvSplitStrategyCombo.currentText() if hasattr(self, "cvSplitStrategyCombo") else "StratifiedKFold (classification)"
    group_col = self.cvGroupColumnCombo.currentText() if hasattr(self, "cvGroupColumnCombo") and self.cvGroupColumnCombo.isVisible() else None
    return {
        "strategy": strategy,
        "n_splits": 5,
        "shuffle": False if strategy.startswith("TimeSeriesSplit") or strategy.startswith("GroupKFold") else True,
        "random_state": None if strategy.startswith("TimeSeriesSplit") or strategy.startswith("GroupKFold") else 42,
        "group_column": group_col,
    }


def _kuquickml_enrich_bundle_for_save(self, model_name, bundle):
    enriched = dict(bundle)
    model = enriched.get("model")
    reducer = enriched.get("reducer")
    task = _kuquickml_infer_task(enriched)
    enriched["bundle_format_version"] = "2.0"
    enriched["algorithm_name"] = model_name
    enriched["task"] = task
    enriched["saved_at"] = _dt.datetime.now().isoformat(timespec="seconds") if _dt is not None else ""
    enriched["saved_with_versions"] = {
        "python": platform.python_version() if platform is not None else sys.version,
        "numpy": getattr(np, "__version__", ""),
        "pandas": getattr(pd, "__version__", ""),
        "scikit_learn": getattr(_sklearn_pkg, "__version__", ""),
    }
    enriched["cv_settings"] = _kuquickml_get_cv_settings_snapshot(self)

    # training set metrics (from the actual train split used by the app)
    try:
        X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
        y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
        if hasattr(self, "_drop_sample_and_numeric"):
            X_train_num = self._drop_sample_and_numeric(X_train).fillna(0)
        else:
            X_train_num = X_train.drop(columns=[c for c in ["Sample"] if c in X_train.columns]).apply(pd.to_numeric, errors="coerce").fillna(0)
        X_train_num = X_train_num.reindex(columns=enriched.get("feature_names", list(X_train_num.columns)), fill_value=0)
        X_train_used = reducer.transform(X_train_num.values) if reducer is not None else X_train_num.values
        enriched["training_metrics"] = _kuquickml_training_metrics(task, model, X_train_used, y_train)
    except Exception as e:
        enriched["training_metrics"] = {"error": str(e)}

    # test set metrics (from the actual held-out test split used by the app)
    try:
        X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
        y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()
        if hasattr(self, "_drop_sample_and_numeric"):
            X_test_num = self._drop_sample_and_numeric(X_test).fillna(0)
        else:
            X_test_num = X_test.drop(columns=[c for c in ["Sample"] if c in X_test.columns]).apply(pd.to_numeric, errors="coerce").fillna(0)
        X_test_num = X_test_num.reindex(columns=enriched.get("feature_names", list(X_test_num.columns)), fill_value=0)
        X_test_used = reducer.transform(X_test_num.values) if reducer is not None else X_test_num.values
        enriched["test_metrics"] = _kuquickml_test_metrics(task, model, X_test_used, y_test)
    except Exception as e:
        enriched["test_metrics"] = {"error": str(e)}

    # 5-fold CV metrics (same split strategy currently selected in the app)
    try:
        enriched["cv_metrics"] = _kuquickml_cv_metrics(self, model, reducer, task=task)
    except Exception as e:
        enriched["cv_metrics"] = {"error": str(e)}

    return enriched


_kuquickml_original_saveModel = MyApp.saveModel

def _kuquickml_saveModel_with_metadata(self, model_name, filename):
    bundle = self.models.get(model_name)
    if not isinstance(bundle, dict) or "model" not in bundle:
        QMessageBox.warning(self, "Error", "Selected item is not a valid saved model bundle.")
        return
    try:
        enriched = _kuquickml_enrich_bundle_for_save(self, model_name, bundle)
        joblib.dump(enriched, filename)
        scaler = enriched.get("scaler")
        reducer = enriched.get("reducer")
        task = enriched.get("task", "")
        cv_text = "saved" if isinstance(enriched.get("cv_metrics"), dict) else "not saved"
        QMessageBox.information(
            self,
            "Model Saved",
            f"Model '{model_name}' saved successfully.\n"
            f"Task: {task}\n"
            f"Scaler: {type(scaler).__name__ if scaler else 'None'}\n"
            f"Reducer: {type(reducer).__name__ if reducer else 'None'}\n"
            f"Training/CV metadata: {cv_text}"
        )
    except Exception as e:
        QMessageBox.warning(self, "Error", f"Failed to save model:\n{e}")

MyApp.saveModel = _kuquickml_saveModel_with_metadata


# ===== Compare Models tab/menu patch =====
def _kuquickml_fmt_metric_value(value):
    if value is None:
        return '-'
    if isinstance(value, dict):
        mean = value.get('mean')
        std = value.get('std')
        if mean is None:
            return '-'
        if std is None:
            return f"{float(mean):.4f}"
        return f"{float(mean):.4f} ± {float(std):.4f}"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _kuquickml_compare_columns_for_task(task):
    if task == 'regression':
        return [
            ('File', 'file_name'),
            ('Algorithm', 'algorithm_name'),
            ('Train R2', ('training_metrics', 'r2')),
            ('Train RMSE', ('training_metrics', 'rmse')),
            ('Train MSE', ('training_metrics', 'mse')),
            ('Train MAE', ('training_metrics', 'mae')),
            ('Test R2', ('test_metrics', 'r2')),
            ('Test RMSE', ('test_metrics', 'rmse')),
            ('Test MSE', ('test_metrics', 'mse')),
            ('Test MAE', ('test_metrics', 'mae')),
            ('CV R2', ('cv_metrics', 'r2')),
            ('CV RMSE', ('cv_metrics', 'rmse')),
            ('CV MSE', ('cv_metrics', 'mse')),
            ('CV MAE', ('cv_metrics', 'mae')),
        ]
    return [
        ('File', 'file_name'),
        ('Algorithm', 'algorithm_name'),
        ('Train Accuracy', ('training_metrics', 'accuracy')),
        ('Train F1', ('training_metrics', 'f1')),
        ('Train ROC-AUC', ('training_metrics', 'roc_auc')),
        ('Test Accuracy', ('test_metrics', 'accuracy')),
        ('Test F1', ('test_metrics', 'f1')),
        ('Test ROC-AUC', ('test_metrics', 'roc_auc')),
        ('CV Accuracy', ('cv_metrics', 'accuracy')),
        ('CV F1', ('cv_metrics', 'f1')),
        ('CV ROC-AUC', ('cv_metrics', 'roc_auc')),
    ]


def _kuquickml_extract_compare_row(bundle, path):
    return {
        'file_name': os.path.basename(path),
        'algorithm_name': bundle.get('algorithm_name', type(bundle.get('model')).__name__ if isinstance(bundle, dict) and bundle.get('model') is not None else ''),
        'task': bundle.get('task') or _kuquickml_infer_task(bundle),
        'training_metrics': bundle.get('training_metrics', {}),
        'test_metrics': bundle.get('test_metrics', {}),
        'cv_metrics': bundle.get('cv_metrics', {}),
    }


def _kuquickml_fill_compare_table(table, rows, task):
    columns = _kuquickml_compare_columns_for_task(task)
    table.clear()
    table.setRowCount(len(rows))
    table.setColumnCount(len(columns))
    table.setHorizontalHeaderLabels([title for title, _ in columns])
    for i, row in enumerate(rows):
        for j, (_, key) in enumerate(columns):
            if isinstance(key, tuple):
                parent, child = key
                value = row.get(parent, {}).get(child)
                text = _kuquickml_fmt_metric_value(value)
            else:
                text = str(row.get(key, '-'))
            table.setItem(i, j, QTableWidgetItem(text))
    try:
        table.resizeColumnsToContents()
    except Exception:
        pass


def _kuquickml_setup_compare_tab(self):
    layout = QVBoxLayout()
    note1 = QLabel('저장된 모델들의 Train / Test / CV 성능을 비교합니다. 여러 모델 파일을 불러온 뒤 비교 버튼을 누르세요.')
    note2 = QLabel('분류 모델은 분류끼리, 회귀 모델은 회귀끼리 비교됩니다.')
    note3 = QLabel('다중클래스 ROC-AUC는 One-vs-Rest(OVR) 방식과 macro 평균으로 계산합니다.')
    note1.setWordWrap(True)
    note2.setWordWrap(True)
    note3.setWordWrap(True)
    layout.addWidget(note1)
    layout.addWidget(note2)
    layout.addWidget(note3)

    btn_layout = QHBoxLayout()
    self.compareLoadModelsBtn = QPushButton('모델 파일 불러오기')
    self.compareRunBtn = QPushButton('비교 실행')
    self.compareCopyBtn = QPushButton('결과 전체 복사')
    self.compareSaveBtn = QPushButton('결과 저장 CSV')
    self.compareLoadModelsBtn.clicked.connect(lambda: _kuquickml_load_compare_models(self))
    self.compareRunBtn.clicked.connect(lambda: _kuquickml_run_compare_models(self))
    self.compareCopyBtn.clicked.connect(lambda: _kuquickml_copy_compare_results(self))
    self.compareSaveBtn.clicked.connect(lambda: _kuquickml_save_compare_results(self))
    btn_layout.addWidget(self.compareLoadModelsBtn)
    btn_layout.addWidget(self.compareRunBtn)
    btn_layout.addWidget(self.compareCopyBtn)
    btn_layout.addWidget(self.compareSaveBtn)
    layout.addLayout(btn_layout)

    layout.addWidget(QLabel('선택된 모델 파일'))
    self.compareModelListWidget = QListWidget()
    self.compareModelListWidget.setSelectionMode(QListWidget.ExtendedSelection)
    layout.addWidget(self.compareModelListWidget)

    layout.addWidget(QLabel('Classification Models'))
    self.compareClassificationTable = QTableWidget()
    _kuquickml_enable_copyable_table(self.compareClassificationTable)
    layout.addWidget(self.compareClassificationTable)

    layout.addWidget(QLabel('Regression Models'))
    self.compareRegressionTable = QTableWidget()
    _kuquickml_enable_copyable_table(self.compareRegressionTable)
    layout.addWidget(self.compareRegressionTable)

    self.compareTab.setLayout(layout)
    self.compare_model_paths = []


def _kuquickml_table_to_tsv(table, title=None):
    lines = []
    if title:
        lines.append(title)
    if table.columnCount() > 0:
        headers = []
        for col in range(table.columnCount()):
            item = table.horizontalHeaderItem(col)
            headers.append(item.text() if item else '')
        lines.append('	'.join(headers))
    for row in range(table.rowCount()):
        vals = []
        for col in range(table.columnCount()):
            item = table.item(row, col)
            vals.append(item.text() if item else '')
        lines.append('	'.join(vals))
    return '\n'.join(lines)


def _kuquickml_copy_compare_results(self):
    parts = []
    if hasattr(self, 'compareClassificationTable') and self.compareClassificationTable.rowCount() > 0:
        parts.append(_kuquickml_table_to_tsv(self.compareClassificationTable, 'Classification Models'))
    if hasattr(self, 'compareRegressionTable') and self.compareRegressionTable.rowCount() > 0:
        parts.append(_kuquickml_table_to_tsv(self.compareRegressionTable, 'Regression Models'))
    if not parts:
        QMessageBox.information(self, 'Compare Models', '복사할 비교 결과가 없습니다.')
        return
    QApplication.clipboard().setText('\n\n'.join(parts))


def _kuquickml_save_compare_results(self):
    if (not hasattr(self, 'compareClassificationTable') or self.compareClassificationTable.rowCount() == 0) and (not hasattr(self, 'compareRegressionTable') or self.compareRegressionTable.rowCount() == 0):
        QMessageBox.information(self, 'Compare Models', '저장할 비교 결과가 없습니다.')
        return
    path, _ = QFileDialog.getSaveFileName(self, 'Compare Models Save', '', 'CSV Files (*.csv);;Text Files (*.txt);;All Files (*)')
    if not path:
        return
    content_parts = []
    if self.compareClassificationTable.rowCount() > 0:
        content_parts.append(_kuquickml_table_to_tsv(self.compareClassificationTable, 'Classification Models'))
    if self.compareRegressionTable.rowCount() > 0:
        content_parts.append(_kuquickml_table_to_tsv(self.compareRegressionTable, 'Regression Models'))
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        f.write('\n\n'.join(content_parts))


def _kuquickml_open_compare_tab(self):
    if not hasattr(self, 'compareTab') or self.compareTab is None:
        self.compareTab = QWidget()
        self.tabs.addTab(self.compareTab, 'Compare Models')
        _kuquickml_setup_compare_tab(self)
    self.tabs.setCurrentWidget(self.compareTab)


def _kuquickml_load_compare_models(self):
    paths, _ = QFileDialog.getOpenFileNames(self, 'Compare Models', '', 'Joblib Files (*.joblib);;All Files (*)')
    if not paths:
        return
    self.compare_model_paths = paths
    self.compareModelListWidget.clear()
    for path in paths:
        self.compareModelListWidget.addItem(path)


def _kuquickml_run_compare_models(self):
    paths = getattr(self, 'compare_model_paths', [])
    if not paths:
        QMessageBox.warning(self, 'Compare Models', '먼저 모델 파일을 하나 이상 불러오세요.')
        return

    classification_rows = []
    regression_rows = []
    failed = []

    for path in paths:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bundle = joblib.load(path)
            if not isinstance(bundle, dict) or 'model' not in bundle:
                failed.append(f"{os.path.basename(path)}: invalid model bundle")
                continue
            row = _kuquickml_extract_compare_row(bundle, path)
            if row.get('task') == 'regression':
                regression_rows.append(row)
            else:
                classification_rows.append(row)
        except Exception as e:
            failed.append(f"{os.path.basename(path)}: {e}")

    if not classification_rows and not regression_rows:
        msg = '비교할 모델이 없습니다.'
        if failed:
            msg += '\\n\\n' + '\\n'.join(failed[:10])
        QMessageBox.warning(self, 'Compare Models', msg)
        return

    _kuquickml_fill_compare_table(self.compareClassificationTable, classification_rows, 'classification')
    _kuquickml_fill_compare_table(self.compareRegressionTable, regression_rows, 'regression')

    if failed:
        QMessageBox.warning(self, 'Compare Models', '모델 로드 실패' + '\\n' + '\\n'.join(failed[:10]))


_kuquickml_original_initUI_compare = MyApp.initUI

def _kuquickml_initUI_with_compare_menu(self):
    _kuquickml_original_initUI_compare(self)
    self.compareTab = None
    try:
        existing_titles = [action.text() for action in self.menuBar().actions()]
    except Exception:
        existing_titles = []
    if '6. Compare Models' not in existing_titles:
        compareMenu = self.menuBar().addMenu('★6. Compare Models')
        compareAction = QAction('Compare Models', self)
        compareAction.triggered.connect(lambda: _kuquickml_open_compare_tab(self))
        compareMenu.addAction(compareAction)
        compareMenu.triggered.connect(lambda *_: _kuquickml_open_compare_tab(self))

MyApp.initUI = _kuquickml_initUI_with_compare_menu
# ===== end compare models patch =====



# ===== importance dual-display patch (applied before app launch) =====
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def _kuquickml_to_numeric_df(X):
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)
    for col in X_df.columns:
        if X_df[col].dtype == object:
            X_df[col] = pd.to_numeric(X_df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    return X_df.fillna(0)


def _kuquickml_permutation(estimator, X, y, task='classification', n_repeats=10):
    scoring = 'accuracy' if task == 'classification' else 'r2'
    X_df = _kuquickml_to_numeric_df(X)
    res = permutation_importance(estimator, X_df, np.asarray(y), n_repeats=n_repeats, random_state=42, scoring=scoring)
    return np.asarray(res.importances_mean, dtype=float)


def _kuquickml_mutual_info(X, y, task='classification'):
    X_df = _kuquickml_to_numeric_df(X)
    y_arr = np.asarray(y)
    try:
        if task == 'classification':
            vals = mutual_info_classif(X_df, y_arr, random_state=42)
        else:
            vals = mutual_info_regression(X_df, y_arr, random_state=42)
    except Exception:
        vals = np.zeros(X_df.shape[1], dtype=float)
    return np.asarray(vals, dtype=float)


def _kuquickml_mlp_abs_weight(model):
    try:
        vals = np.mean(np.abs(model.coefs_[0]), axis=1)
        return np.asarray(vals, dtype=float)
    except Exception:
        return np.zeros(0, dtype=float)



def _kuquickml_format_importance_value(val):
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ''
        v = float(val)
        if v == 0:
            return '0.0000'
        if abs(v) < 1e-4:
            return f"{v:.4e}"
        return f"{v:.4f}"
    except Exception:
        return str(val)


def _kuquickml_copy_table_selection(table, include_headers=False, all_cells=False):
    try:
        if all_cells:
            top_row, bottom_row = 0, table.rowCount() - 1
            left_col, right_col = 0, table.columnCount() - 1
            ranges = [(top_row, bottom_row, left_col, right_col)] if table.rowCount() and table.columnCount() else []
        else:
            selected = table.selectedRanges()
            ranges = [(r.topRow(), r.bottomRow(), r.leftColumn(), r.rightColumn()) for r in selected]
        if not ranges:
            return
        parts = []
        for idx, (top, bottom, left, right) in enumerate(ranges):
            if idx > 0:
                parts.append('')
            if include_headers:
                headers = []
                for col in range(left, right + 1):
                    item = table.horizontalHeaderItem(col)
                    headers.append(item.text() if item else '')
                parts.append('	'.join(headers))
            for row in range(top, bottom + 1):
                vals = []
                for col in range(left, right + 1):
                    item = table.item(row, col)
                    vals.append(item.text() if item else '')
                parts.append('	'.join(vals))
        QApplication.clipboard().setText('\n'.join(parts))
    except Exception:
        pass


def _kuquickml_save_table_csv(table, parent=None):
    try:
        path, _ = QFileDialog.getSaveFileName(parent, 'Save Table', '', 'CSV Files (*.csv);;All Files (*)')
        if not path:
            return
        import csv
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            headers = []
            for col in range(table.columnCount()):
                item = table.horizontalHeaderItem(col)
                headers.append(item.text() if item else '')
            writer.writerow(headers)
            for row in range(table.rowCount()):
                writer.writerow([(table.item(row, col).text() if table.item(row, col) else '') for col in range(table.columnCount())])
    except Exception as e:
        QMessageBox.warning(parent, 'Save Table', f'Failed to save table\n{e}')


def _kuquickml_enable_copyable_table(table):
    try:
        if getattr(table, '_kuquickml_copy_enabled', False):
            return
    except Exception:
        pass
    try:
        table.setSelectionMode(QTableWidget.ExtendedSelection)
    except Exception:
        pass
    table.setSelectionBehavior(QTableWidget.SelectItems)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setContextMenuPolicy(Qt.CustomContextMenu)

    def _show_menu(pos, t=table):
        menu = QMenu(t)
        act_copy = menu.addAction('Copy')
        act_copy_all = menu.addAction('Copy All')
        act_copy.triggered.connect(lambda _=False: _kuquickml_copy_table_selection(t, include_headers=True, all_cells=False))
        act_copy_all.triggered.connect(lambda _=False: _kuquickml_copy_table_selection(t, include_headers=True, all_cells=True))
        menu.exec_(t.viewport().mapToGlobal(pos))

    table.customContextMenuRequested.connect(_show_menu)

    copy_shortcut = QAction(table)
    copy_shortcut.setShortcut('Ctrl+C')
    copy_shortcut.triggered.connect(lambda _=False, t=table: _kuquickml_copy_table_selection(t, include_headers=True, all_cells=False))
    table.addAction(copy_shortcut)
    table._kuquickml_copy_enabled = True


def _kuquickml_build_importance_table(headers, rows, parent=None):
    table = QTableWidget(parent)
    table.setRowCount(len(rows))
    table.setColumnCount(len(headers))
    table.setHorizontalHeaderLabels(headers)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            table.setItem(i, j, QTableWidgetItem(str(val)))
    table.resizeColumnsToContents()
    _kuquickml_enable_copyable_table(table)
    return table


def _kuquickml_show_dual_importance_dialog(self, title, feature_names, perm_values, alt_values=None, alt_name=None, intro=''):
    feature_names = list(feature_names)
    perm_values = np.asarray(perm_values, dtype=float)
    alt_values = None if alt_values is None else np.asarray(alt_values, dtype=float)

    perm_order = np.argsort(np.nan_to_num(perm_values, nan=-np.inf))[::-1]
    perm_rows = [
        [str(feature_names[idx]), _kuquickml_format_importance_value(perm_values[idx])]
        for idx in perm_order
    ]

    alt_rows = []
    if alt_values is not None and alt_name:
        safe_alt = np.full(len(feature_names), np.nan, dtype=float)
        safe_alt[:min(len(feature_names), len(alt_values))] = alt_values[:min(len(feature_names), len(alt_values))]
        alt_order = np.argsort(np.nan_to_num(safe_alt, nan=-np.inf))[::-1]
        alt_rows = [
            [str(feature_names[idx]), _kuquickml_format_importance_value(safe_alt[idx])]
            for idx in alt_order
        ]

    dialog = QDialog(self)
    dialog.setWindowTitle(title)
    dialog.resize(980, 760)
    layout = QVBoxLayout(dialog)

    if intro:
        label = QLabel(intro)
        label.setWordWrap(True)
        layout.addWidget(label)

    mi_note = QLabel('Mutual information에는 고정된 유의 기준값이 없습니다. 따라서 절대값보다는 같은 데이터 내 다른 feature들과의 상대적 크기와 순위를 기준으로 해석하는 것이 적절합니다.')
    mi_note.setWordWrap(True)
    layout.addWidget(mi_note)

    perm_label = QLabel('Permutation Importance')
    layout.addWidget(perm_label)
    perm_table = _kuquickml_build_importance_table(['Feature', 'Importance'], perm_rows, dialog)
    layout.addWidget(perm_table)

    if alt_rows:
        alt_label = QLabel(f'Alternative Importance ({alt_name})')
        layout.addWidget(alt_label)
        alt_table = _kuquickml_build_importance_table(['Feature', 'Importance'], alt_rows, dialog)
        layout.addWidget(alt_table)

    guide = QLabel('셀을 드래그해 선택한 뒤 우클릭 복사 또는 Ctrl+C를 사용할 수 있습니다.')
    guide.setWordWrap(True)
    layout.addWidget(guide)

    dialog.setLayout(layout)
    dialog.setWindowModality(Qt.NonModal)
    dialog.show()
    if not hasattr(self, '_kuquickml_open_dialogs'):
        self._kuquickml_open_dialogs = []
    self._kuquickml_open_dialogs.append(dialog)


def _kuquickml_showMLPFeatureImportances_dual(self, feature_importances, alternative_importances=None, alternative_name=None):
    names = [f for f, _ in feature_importances]
    perm_vals = [v for _, v in feature_importances]
    alt_vals = None if alternative_importances is None else [v for _, v in alternative_importances]
    intro = '<b>MLP</b><br>Permutation importance와 대안 지표를 함께 표시합니다.'
    _kuquickml_show_dual_importance_dialog(self, 'Feature Importances', names, perm_vals, alt_vals, alternative_name, intro)


def _kuquickml_createMLPClassificationModel_dual(self):
    if not self.checkDataSplit():
        return
    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()
    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    X_train_used = X_train_numeric.values
    X_test_used = X_test_numeric.values
    hidden_layer_input_text = self.hidden_layer_input.text().strip()
    hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(',')))
    alpha_input_text = self.alpha_input.text().strip()
    alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)
    lr_input_text = self.learning_rate_input.text().strip()
    learning_rate = 0.001 if not lr_input_text else float(lr_input_text)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=int(self.max_iter_input.value()), random_state=int(self.random_state_input.value()), alpha=alpha, solver=self.solver_input.currentText(), activation=self.activation_input.currentText(), learning_rate_init=learning_rate)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
        mlp.fit(X_train_used, y_train)
    if mlp.n_iter_ == mlp.max_iter:
        QMessageBox.warning(self, 'Iteration Warning', 'Maximum iterations reached. Consider increasing max_iter.')
    y_pred_train = mlp.predict(X_train_used)
    y_pred_test = mlp.predict(X_test_used)
    cm = confusion_matrix(y_test, y_pred_test)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
        precision = np.nan_to_num(precision)
    labels = [f'Class {i}' for i in range(cm.shape[0])]
    cm_df = pd.DataFrame(cm, index=[f'Actual {label}' for label in labels], columns=[f'Predicted {label}' for label in labels])
    cm_df['Prediction Accuracy (%)'] = precision
    self.showConfusionMatrix(cm_df)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)
    perm_vals = _kuquickml_permutation(mlp, X_test_used, y_test, task='classification', n_repeats=10)
    alt_vals = _kuquickml_mlp_abs_weight(mlp)
    order = np.argsort(perm_vals)[::-1]
    feature_importances = [(feature_names[idx], float(perm_vals[idx])) for idx in order]
    alternative_importances = [(feature_names[idx], float(alt_vals[idx])) for idx in order]
    self.showMLPFeatureImportances(feature_importances, alternative_importances=alternative_importances, alternative_name='input-layer mean abs weight')
    self.models['MLP Classification'] = {'model': mlp, 'scaler': self._get_bundle_scaler(), 'reducer': None, 'feature_names': feature_names, 'label_mapping': self._get_label_mapping()}


def _kuquickml_createMLPRegressionModel_dual(self):
    if not self.checkDataSplit():
        return
    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()
    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    X_train_used = X_train_numeric.values
    X_test_used = X_test_numeric.values
    hidden_layer_input_text = self.hidden_layer_input.text().strip()
    hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(',')))
    alpha_input_text = self.alpha_input.text().strip()
    alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)
    lr_input_text = self.learning_rate_input.text().strip()
    learning_rate = 0.001 if not lr_input_text else float(lr_input_text)
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=int(self.max_iter_input.value()), random_state=int(self.random_state_input.value()), alpha=alpha, solver=self.solver_input.currentText(), activation=self.activation_input.currentText(), learning_rate_init=learning_rate)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
        mlp.fit(X_train_used, y_train)
    if mlp.n_iter_ == mlp.max_iter:
        QMessageBox.warning(self, 'Iteration Warning', 'Maximum iterations reached. Consider increasing max_iter.')
    y_pred_train = mlp.predict(X_train_used)
    y_pred_test = mlp.predict(X_test_used)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)
    perm_vals = _kuquickml_permutation(mlp, X_test_used, y_test, task='regression', n_repeats=10)
    alt_vals = _kuquickml_mlp_abs_weight(mlp)
    order = np.argsort(perm_vals)[::-1]
    feature_importances = [(feature_names[idx], float(perm_vals[idx])) for idx in order]
    alternative_importances = [(feature_names[idx], float(alt_vals[idx])) for idx in order]
    self.showMLPFeatureImportances(feature_importances, alternative_importances=alternative_importances, alternative_name='input-layer mean abs weight')
    self.models['MLP Regression'] = {'model': mlp, 'scaler': self._get_bundle_scaler(), 'reducer': None, 'feature_names': feature_names, 'label_mapping': None}


def _kuquickml_showKNNPermutationImportance_dual(self, reducer, X_eval, y_eval, model, feature_names, title_prefix='KNN', task='classification'):
    try:
        class _ReducerWrappedEstimator:
            def __init__(self, reducer, model):
                self.reducer = reducer
                self.model = model
            def fit(self, X, y=None):
                return self
            def predict(self, X):
                X_in = X.values if isinstance(X, pd.DataFrame) else X
                if self.reducer is not None:
                    X_in = self.reducer.transform(X_in)
                return self.model.predict(X_in)
            def score(self, X, y):
                y_pred = self.predict(X)
                return accuracy_score(y, y_pred) if task == 'classification' else r2_score(y, y_pred)
        X_use = X_eval.copy()
        y_use = np.asarray(y_eval)
        if len(X_use) > 2000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_use), size=2000, replace=False)
            X_use = X_use.iloc[idx] if isinstance(X_use, pd.DataFrame) else X_use[idx]
            y_use = y_use[idx]
        perm_vals = _kuquickml_permutation(_ReducerWrappedEstimator(reducer, model), X_use, y_use, task=task, n_repeats=10)
        alt_vals = _kuquickml_mutual_info(X_use, y_use, task=task)
        names = list(feature_names) if feature_names and len(feature_names) == len(perm_vals) else [f'X{i}' for i in range(len(perm_vals))]
        intro = f'<b>{title_prefix}</b><br>Permutation importance와 대안 지표를 함께 표시합니다.<br>KNN 대안은 mutual information입니다.'
        _kuquickml_show_dual_importance_dialog(self, 'Feature Importances', names, perm_vals, alt_vals, 'mutual information', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Permutation Importance Error', f'Failed to compute importance:\n{e}')


def _kuquickml_showSVMImportanceUnavailable_dual(self, kernel, reducer, X_test, y_test, model, feature_names, title_prefix='SVM', task='classification'):
    try:
        class _ReducerWrappedEstimator:
            def __init__(self, fitted_model, fitted_reducer=None):
                self._model = fitted_model
                self._reducer = fitted_reducer
            def fit(self, X, y=None):
                return self
            def predict(self, X):
                Xv = X.values if isinstance(X, pd.DataFrame) else X
                if self._reducer is not None:
                    Xv = self._reducer.transform(Xv)
                return self._model.predict(Xv)
            def score(self, X, y):
                y_pred = self.predict(X)
                return accuracy_score(y, y_pred) if task == 'classification' else r2_score(y, y_pred)
        X_use = X_test.copy()
        y_use = np.asarray(y_test)
        if len(X_use) > 2000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_use), size=2000, replace=False)
            X_use = X_use.iloc[idx] if isinstance(X_use, pd.DataFrame) else X_use[idx]
            y_use = y_use[idx]
        perm_vals = _kuquickml_permutation(_ReducerWrappedEstimator(model, reducer), X_use, y_use, task=task, n_repeats=10)
        alt_vals = _kuquickml_mutual_info(X_use, y_use, task=task)
        names = list(feature_names) if feature_names and len(feature_names) == len(perm_vals) else [f'X{i}' for i in range(len(perm_vals))]
        intro = f'<b>{title_prefix}</b><br>Permutation importance와 대안 지표를 함께 표시합니다.<br>SVM 대안은 kernel과 무관하게 mutual information입니다.'
        _kuquickml_show_dual_importance_dialog(self, 'Feature Importances', names, perm_vals, alt_vals, 'mutual information', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Permutation Importance Error', f'Failed to compute importance:\n{e}')


MyApp.showMLPFeatureImportances = _kuquickml_showMLPFeatureImportances_dual
MyApp.createMLPClassificationModel = _kuquickml_createMLPClassificationModel_dual
MyApp.createMLPRegressionModel = _kuquickml_createMLPRegressionModel_dual
MyApp.showKNNPermutationImportance = _kuquickml_showKNNPermutationImportance_dual
MyApp.showSVMImportanceUnavailable = _kuquickml_showSVMImportanceUnavailable_dual
# ===== end importance dual-display patch =====



# ===== loading UI patch for model creation + simplify compare tab buttons =====

def _kuquickml_run_with_loading(self, title, label_text, fn):
    dlg = QProgressDialog(label_text, None, 0, 0, self)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(Qt.WindowModal)
    dlg.setCancelButton(None)
    dlg.setMinimumDuration(0)
    dlg.setAutoClose(True)
    dlg.setAutoReset(True)
    dlg.show()
    QApplication.processEvents()
    try:
        return fn()
    finally:
        dlg.close()
        QApplication.processEvents()


def _kuquickml_make_loading_wrapper(method_name, title, label_text):
    original = getattr(MyApp, method_name, None)
    if original is None:
        return
    def _wrapped(self, *args, **kwargs):
        # Qt signals may pass an extra bool/checked argument. The wrapped model-creation
        # methods in this app do not expect positional signal arguments, so ignore them.
        return _kuquickml_run_with_loading(self, title, label_text, lambda: original(self))
    setattr(MyApp, method_name, _wrapped)


def _kuquickml_setup_compare_tab(self):
    layout = QVBoxLayout()
    note1 = QLabel('저장된 모델들의 Train / Test / CV 성능을 비교합니다. 여러 모델 파일을 불러온 뒤 비교 버튼을 누르세요.')
    note2 = QLabel('분류 모델은 분류끼리, 회귀 모델은 회귀끼리 비교됩니다.')
    note3 = QLabel('다중클래스 ROC-AUC는 One-vs-Rest(OVR) 방식과 macro 평균으로 계산합니다.')
    note1.setWordWrap(True)
    note2.setWordWrap(True)
    note3.setWordWrap(True)
    layout.addWidget(note1)
    layout.addWidget(note2)
    layout.addWidget(note3)

    btn_layout = QHBoxLayout()
    self.compareLoadModelsBtn = QPushButton('모델 파일 불러오기')
    self.compareRunBtn = QPushButton('비교 실행')
    self.compareLoadModelsBtn.clicked.connect(lambda: _kuquickml_load_compare_models(self))
    self.compareRunBtn.clicked.connect(lambda: _kuquickml_run_compare_models(self))
    btn_layout.addWidget(self.compareLoadModelsBtn)
    btn_layout.addWidget(self.compareRunBtn)
    layout.addLayout(btn_layout)

    layout.addWidget(QLabel('선택된 모델 파일'))
    self.compareModelListWidget = QListWidget()
    self.compareModelListWidget.setSelectionMode(QListWidget.ExtendedSelection)
    layout.addWidget(self.compareModelListWidget)

    layout.addWidget(QLabel('Classification Models'))
    self.compareClassificationTable = QTableWidget()
    _kuquickml_enable_copyable_table(self.compareClassificationTable)
    layout.addWidget(self.compareClassificationTable)

    layout.addWidget(QLabel('Regression Models'))
    self.compareRegressionTable = QTableWidget()
    _kuquickml_enable_copyable_table(self.compareRegressionTable)
    layout.addWidget(self.compareRegressionTable)

    self.compareTab.setLayout(layout)


# Wrap model creation methods with loading indicator
_kuquickml_make_loading_wrapper('createClassificationModel', 'Loading', 'KNN 분류 모델을 생성하는 중입니다...')
_kuquickml_make_loading_wrapper('createRegressionModel', 'Loading', 'KNN 회귀 모델을 생성하는 중입니다...')
_kuquickml_make_loading_wrapper('createMLPClassificationModel', 'Loading', 'MLP 분류 모델을 생성하는 중입니다...')
_kuquickml_make_loading_wrapper('createMLPRegressionModel', 'Loading', 'MLP 회귀 모델을 생성하는 중입니다...')
_kuquickml_make_loading_wrapper('createRFClassificationModel', 'Loading', 'RF 분류 모델을 생성하는 중입니다...')
_kuquickml_make_loading_wrapper('createRFRegressionModel', 'Loading', 'RF 회귀 모델을 생성하는 중입니다...')
_kuquickml_make_loading_wrapper('createSVMRegressionModel', 'Loading', 'SVM 회귀 모델을 생성하는 중입니다...')
# createSVMModel already has its own progress dialog in this file, so we leave it as-is.

# ===== end loading UI patch =====



# ===== debug patch for reproducibility / MI inspection =====
import hashlib

def _kuquickml_hash_df(df):
    try:
        if isinstance(df, pd.DataFrame):
            obj = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        else:
            arr = np.asarray(df)
            obj = arr.tobytes()
        return hashlib.md5(obj).hexdigest()
    except Exception as e:
        return f"hash_failed:{e}"


def _kuquickml_debug_print_split(tag, X_train, X_test, y_train, y_test):
    try:
        print(f"\n[DEBUG] ===== {tag} split summary =====")
        print("[DEBUG] X_train shape:", getattr(X_train, 'shape', None))
        print("[DEBUG] X_test shape:", getattr(X_test, 'shape', None))
        print("[DEBUG] y_train shape:", np.shape(y_train))
        print("[DEBUG] y_test shape:", np.shape(y_test))
        if isinstance(X_train, pd.DataFrame):
            print("[DEBUG] X_train first 5 indices:", list(X_train.index[:5]))
            print("[DEBUG] X_test first 5 indices:", list(X_test.index[:5]))
            print("[DEBUG] X_train first 5 columns:", list(X_train.columns[:5]))
        print("[DEBUG] X_train hash:", _kuquickml_hash_df(X_train))
        print("[DEBUG] X_test hash:", _kuquickml_hash_df(X_test))
        ytr = np.asarray(y_train)
        yte = np.asarray(y_test)
        try:
            u1, c1 = np.unique(ytr, return_counts=True)
            print("[DEBUG] y_train unique/counts:", list(zip(u1.tolist(), c1.tolist())))
        except Exception as e:
            print("[DEBUG] y_train unique/counts failed:", e)
        try:
            u2, c2 = np.unique(yte, return_counts=True)
            print("[DEBUG] y_test unique/counts:", list(zip(u2.tolist(), c2.tolist())))
        except Exception as e:
            print("[DEBUG] y_test unique/counts failed:", e)
        print(f"[DEBUG] ===== end {tag} split summary =====\n")
    except Exception as e:
        print("[DEBUG] split summary failed:", e)
        traceback.print_exc()


def _kuquickml_safe_mutual_info_classif(X, y, n_bins=10):
    X_df = _kuquickml_to_numeric_df(X).copy()
    y_arr = np.asarray(y)
    vals = []
    for col in X_df.columns:
        s = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        if s.nunique(dropna=False) <= 1:
            vals.append(0.0)
            continue
        try:
            q = int(min(n_bins, max(2, int(s.nunique()))))
            binned = pd.qcut(s.rank(method='first'), q=q, duplicates='drop').astype(str)
            mi = mutual_info_score(binned, y_arr)
        except Exception:
            try:
                q = int(min(n_bins, max(2, int(s.nunique()))))
                binned = pd.cut(s, bins=q, duplicates='drop').astype(str)
                mi = mutual_info_score(binned, y_arr)
            except Exception:
                mi = 0.0
        vals.append(float(mi))
    return np.asarray(vals, dtype=float)


def _kuquickml_safe_mutual_info_regression(X, y, n_bins=10):
    X_df = _kuquickml_to_numeric_df(X).copy()
    y_arr = pd.to_numeric(pd.Series(np.asarray(y)), errors='coerce').fillna(0)
    vals = []
    try:
        y_q = int(min(n_bins, max(2, int(y_arr.nunique()))))
        y_binned = pd.qcut(y_arr.rank(method='first'), q=y_q, duplicates='drop').astype(str)
    except Exception:
        try:
            y_q = int(min(n_bins, max(2, int(y_arr.nunique()))))
            y_binned = pd.cut(y_arr, bins=y_q, duplicates='drop').astype(str)
        except Exception:
            y_binned = pd.Series(['0'] * len(y_arr))
    for col in X_df.columns:
        s = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
        if s.nunique(dropna=False) <= 1:
            vals.append(0.0)
            continue
        try:
            q = int(min(n_bins, max(2, int(s.nunique()))))
            x_binned = pd.qcut(s.rank(method='first'), q=q, duplicates='drop').astype(str)
            mi = mutual_info_score(x_binned, y_binned)
        except Exception:
            try:
                q = int(min(n_bins, max(2, int(s.nunique()))))
                x_binned = pd.cut(s, bins=q, duplicates='drop').astype(str)
                mi = mutual_info_score(x_binned, y_binned)
            except Exception:
                mi = 0.0
        vals.append(float(mi))
    return np.asarray(vals, dtype=float)


def _kuquickml_mutual_info_debug(X, y, task='classification'):
    X_df = _kuquickml_to_numeric_df(X)
    y_arr = np.asarray(y)
    try:
        print("\n[DEBUG] ===== mutual information debug =====")
        print("[DEBUG] task:", task)
        print("[DEBUG] X shape:", X_df.shape)
        print("[DEBUG] y shape:", y_arr.shape)
        print("[DEBUG] X hash:", _kuquickml_hash_df(X_df))
        print("[DEBUG] X dtypes(first 10):")
        try:
            print(X_df.dtypes[:10])
        except Exception as e:
            print("[DEBUG] dtype print failed:", e)
        print("[DEBUG] X head(3):")
        try:
            print(X_df.head(3))
        except Exception as e:
            print("[DEBUG] head failed:", e)
        try:
            print("[DEBUG] Total NaN in X:", int(pd.isna(X_df).sum().sum()))
        except Exception as e:
            print("[DEBUG] NaN count failed:", e)
        try:
            print("[DEBUG] Total exact zeros in X:", int((X_df == 0).sum().sum()))
        except Exception as e:
            print("[DEBUG] zero count failed:", e)
        try:
            print("[DEBUG] variance describe:")
            print(X_df.var(numeric_only=True).describe())
        except Exception as e:
            print("[DEBUG] variance summary failed:", e)
        try:
            u, c = np.unique(y_arr, return_counts=True)
            print("[DEBUG] y unique/counts:", list(zip(u.tolist(), c.tolist())))
        except Exception as e:
            print("[DEBUG] y unique/counts failed:", e)

        used_fallback = False
        try:
            if task == 'classification':
                vals = mutual_info_classif(X_df, y_arr, random_state=42)
            else:
                vals = mutual_info_regression(X_df, y_arr, random_state=42)
            vals = np.asarray(vals, dtype=float)
            print("[DEBUG] sklearn mutual_info succeeded")
        except Exception as e:
            print("[DEBUG] sklearn mutual_info failed, switching to safe fallback:", e)
            traceback.print_exc()
            used_fallback = True
            if task == 'classification':
                vals = _kuquickml_safe_mutual_info_classif(X_df, y_arr)
            else:
                vals = _kuquickml_safe_mutual_info_regression(X_df, y_arr)
            vals = np.asarray(vals, dtype=float)

        print("[DEBUG] MI first 20:", vals[:20])
        print("[DEBUG] MI min:", float(np.min(vals)) if len(vals) else None)
        print("[DEBUG] MI max:", float(np.max(vals)) if len(vals) else None)
        print("[DEBUG] MI mean:", float(np.mean(vals)) if len(vals) else None)
        print("[DEBUG] MI nonzero count:", int(np.sum(np.abs(vals) > 0)))
        print("[DEBUG] MI backend:", 'safe fallback' if used_fallback else 'sklearn mutual_info')
        print("[DEBUG] ===== end mutual information debug =====\n")
    except Exception as e:
        print("[DEBUG] mutual information calculation failed:", e)
        traceback.print_exc()
        vals = np.zeros(X_df.shape[1], dtype=float)
    return np.asarray(vals, dtype=float)


def _kuquickml_showKNNPermutationImportance_dual_debug(self, reducer, X_eval, y_eval, model, feature_names, title_prefix='KNN', task='classification'):
    try:
        X_use = _kuquickml_to_numeric_df(X_eval)
        print("\n[DEBUG] ===== KNN importance dialog input =====")
        print("[DEBUG] reducer:", type(reducer).__name__ if reducer is not None else None)
        print("[DEBUG] model:", type(model).__name__)
        print("[DEBUG] feature count:", len(feature_names))
        print("[DEBUG] X_eval hash(before reduce):", _kuquickml_hash_df(X_use))
        if reducer is not None:
            X_model = reducer.transform(X_use.values)
            print("[DEBUG] X_model shape(after reduce):", np.shape(X_model))
            print("[DEBUG] X_model hash(after reduce):", _kuquickml_hash_df(pd.DataFrame(X_model)))
        else:
            X_model = X_use
            print("[DEBUG] X_model shape(no reduce):", getattr(X_model, 'shape', None))
        print("[DEBUG] ===== end KNN importance dialog input =====\n")
        perm_vals = _kuquickml_permutation(model, X_model, y_eval, task=task, n_repeats=10)
        alt_vals = _kuquickml_mutual_info_debug(X_use, y_eval, task=task)
        intro = f'<b>{title_prefix}</b><br>Permutation importance와 대안 지표를 함께 표시합니다.<br>KNN 대안은 mutual information입니다.'
        _kuquickml_show_dual_importance_dialog(self, 'Feature Importances', feature_names, perm_vals, alt_vals, 'mutual information', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Feature Importance', f'Failed to compute feature importance:\n{e}')
        traceback.print_exc()


def _kuquickml_createClassificationModel_debug(self):
    if not self.checkDataSplit():
        return
    X_train = pd.read_csv(resource_path("Temp/X_train.csv"))
    X_test = pd.read_csv(resource_path("Temp/X_test.csv"))
    y_train = pd.read_csv(resource_path("Temp/y_train.csv")).values.ravel()
    y_test = pd.read_csv(resource_path("Temp/y_test.csv")).values.ravel()
    _kuquickml_debug_print_split('KNN classification', X_train, X_test, y_train, y_test)

    X_train_numeric = X_train.drop(columns=['Sample'])
    X_test_numeric = X_test.drop(columns=['Sample'])

    n_neighbors = self.n_neighbors_input.value()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    selected_method = self.getSelectedDimReductionMethod()
    if not selected_method:
        QMessageBox.warning(self, "Selection Error", "Please select a dimensionality reduction method.")
        return

    method_name, reducer = selected_method
    print("[DEBUG] selected dim reduction method:", method_name)
    reducer.fit(X_train_numeric.values, y_train)
    X_train_embedded = reducer.transform(X_train_numeric.values)
    X_test_embedded = reducer.transform(X_test_numeric.values)
    print("[DEBUG] X_train_embedded hash:", _kuquickml_hash_df(pd.DataFrame(X_train_embedded)))
    print("[DEBUG] X_test_embedded hash:", _kuquickml_hash_df(pd.DataFrame(X_test_embedded)))

    knn.fit(X_train_embedded, y_train)
    accuracy = knn.score(X_test_embedded, y_test)
    print("[DEBUG] KNN test accuracy:", accuracy)
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
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
        precision = np.nan_to_num(precision)

    cm_df = pd.DataFrame(cm, index=true, columns=pred)
    cm_df['Prediction Accuracy (%)'] = precision
    self.showConfusionMatrix(cm_df)
    feature_names = list(X_train_numeric.columns)
    self.showKNNPermutationImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix="KNN Classification", task="classification")

    self.models["KNN Classification"] = {
        "model": knn,
        "scaler": self._get_bundle_scaler(),
        "reducer": reducer,
        "feature_names": feature_names,
        "label_mapping": self._get_label_mapping()
    }
    if reducer:
        self.model_reducers["KNN Classification"] = reducer

# ===== final stability patch: custom permutation + safe MI + loading =====
def _kuquickml_mutual_info_safe_only(X, y, task='classification'):
    X_df = _kuquickml_to_numeric_df(X)
    y_arr = np.asarray(y)
    try:
        if task == 'classification':
            return _kuquickml_safe_mutual_info_classif(X_df, y_arr)
        return _kuquickml_safe_mutual_info_regression(X_df, y_arr)
    except Exception:
        return np.zeros(X_df.shape[1], dtype=float)

# Use stable safe MI backend consistently
_kuquickml_mutual_info = _kuquickml_mutual_info_safe_only

# Use original-feature custom permutation dialogs
MyApp.showKNNPermutationImportance = _kuquickml_showKNNPermutationImportance_dual
MyApp.showSVMImportanceUnavailable = _kuquickml_showSVMImportanceUnavailable_dual

# Ensure model creation keeps loading dialog
_kuquickml_make_loading_wrapper('createClassificationModel', 'Loading', 'KNN 분류 모델을 생성하는 중입니다...')
# ===== end final stability patch =====



# ===== safe tabular loading patch (CSV/XLSX, commas in numbers, duplicate-like headers) =====
def _kuquickml_read_tabular_any(path):
    lower = str(path).lower()
    if lower.endswith(('.xlsx', '.xls')):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str)

def _kuquickml_normalize_header_only(col):
    return str(col).strip()

def _kuquickml_safe_numeric_series(s):
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _kuquickml_label_numeric_series(s):
    # Label도 숫자 문자열(예: "0", "1", "1,000")이면 숫자로 읽을 수 있게 처리
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
    s = s.str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def _kuquickml_looks_numeric_enough(s, threshold=0.8):
    temp = s.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
    temp = temp.str.replace(",", "", regex=False)
    converted = pd.to_numeric(temp, errors="coerce")
    valid = temp.notna().sum()
    if valid == 0:
        return False
    ok = converted.notna().sum()
    return (ok / valid) >= threshold

def _kuquickml_find_duplicate_like_columns(columns):
    groups = {}
    for c in columns:
        name = str(c)
        base = re.sub(r"\s*\.\d+$", "", name).strip()
        groups.setdefault(base, []).append(name)
    return {k: v for k, v in groups.items() if len(v) > 1}

def _kuquickml_safe_load_csvviewer(self, filename):
    data = _kuquickml_read_tabular_any(filename)
    data.columns = [_kuquickml_normalize_header_only(c) for c in data.columns]

    dup_like = _kuquickml_find_duplicate_like_columns(data.columns)
    if dup_like:
        msg_lines = []
        for base_name, cols in list(dup_like.items())[:10]:
            msg_lines.append(f"{base_name} -> {', '.join(cols)}")
        QMessageBox.warning(
            self,
            "Duplicate-like Columns Detected",
            "중복되었거나 중복처럼 보이는 열 이름이 감지되었습니다.\n"
            "이 열들은 서로 다른 feature일 수도 있고, 중복 헤더일 수도 있습니다.\n\n"
            + "\n".join(msg_lines)
        )

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

    # Label 처리: 숫자형으로 안전하게 변환 가능하면 숫자 사용, 아니면 매핑 다이얼로그 사용
    label_numeric = _kuquickml_label_numeric_series(data[label_column_name])
    label_raw_nonempty = data[label_column_name].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
    if label_raw_nonempty.notna().sum() == label_numeric.notna().sum() and label_raw_nonempty.notna().sum() > 0:
        data[label_column_name] = label_numeric.astype(float)
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

    failed_numeric_cols = []
    converted_cols = []
    for col in feature_columns:
        original = data[col].copy()
        if _kuquickml_looks_numeric_enough(original):
            converted = _kuquickml_safe_numeric_series(original)
            data[col] = converted
            converted_cols.append(col)
            before_nonempty = original.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
            bad_mask = before_nonempty.notna() & converted.isna()
            if bad_mask.any():
                failed_numeric_cols.append(col)
        else:
            # 숫자형처럼 안 보이는 열도 coercion 시도는 하되, 전부 NaN이면 사용자에게 알림
            converted = _kuquickml_safe_numeric_series(original)
            data[col] = converted
            before_nonempty = original.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan})
            if before_nonempty.notna().sum() > 0 and converted.notna().sum() == 0:
                failed_numeric_cols.append(col)

    if failed_numeric_cols:
        preview = ", ".join(map(str, failed_numeric_cols[:10]))
        if len(failed_numeric_cols) > 10:
            preview += f" 외 {len(failed_numeric_cols) - 10}개"
        QMessageBox.warning(
            self,
            "Numeric Conversion Warning",
            "일부 열에서 숫자 변환 중 변환되지 않는 값이 발견되었습니다.\n"
            "해당 값들은 NaN으로 처리되며, 이후 전처리 단계에서 처리되어야 합니다.\n\n"
            f"{preview}"
        )

    if data.shape[0] == 0:
        QMessageBox.warning(self, "Empty Data", "로드된 데이터에 행이 없습니다.")

    # 컬럼명 통일
    data = data.rename(columns={sample_column_name: 'Sample', label_column_name: 'Label'})

    self.original_data = data[['Sample'] + feature_columns].copy()
    self.X = data[feature_columns].to_numpy()
    self.y = y

    self.showCsvData(
        data[['Sample'] + feature_columns + ['Label']].values.tolist(),
        ['Sample'] + feature_columns + ['Label']
    )

def _kuquickml_loadCsv_dialog_safe(self, checked=False):
    options = QFileDialog.Options()
    filename, _ = QFileDialog.getOpenFileName(
        self, "Open CSV/XLSX File", "",
        "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)",
        options=options
    )
    if not filename:
        return
    try:
        self.csvViewer.loadCsv(filename)
        if hasattr(self, "guideWidget"):
            self.guideWidget.hide()
        self.csvViewer.show()

        output_dir = resource_path('Temp')
        os.makedirs(output_dir, exist_ok=True)

        if getattr(self.csvViewer, "original_data", None) is not None:
            self.csvViewer.original_data.to_csv(os.path.join(output_dir, "original_X.csv"), index=False)

        if getattr(self.csvViewer, "y", None) is not None:
            pd.DataFrame(self.csvViewer.y, columns=["Label"]).to_csv(
                os.path.join(output_dir, "scaled_y.csv"), index=False
            )

        self._refresh_cv_group_columns()
        QMessageBox.information(self, "Load Complete", f"Loaded file:\n{os.path.basename(filename)}")
    except Exception as e:
        QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{e}")

def _kuquickml_prepare_unknown_prediction_frame(df, saved_feature_names):
    work = df.copy()
    original_cols = list(work.columns)
    trimmed_map = {}
    for c in original_cols:
        trimmed = str(c).strip()
        # 같은 trim 이름이 여러 번 나오면 첫 번째만 유지하고 나머지는 그대로 둔다.
        if trimmed not in trimmed_map:
            trimmed_map[trimmed] = c
    work = work.rename(columns={orig: trim for trim, orig in trimmed_map.items()})

    saved_feature_names = [str(c).strip() for c in list(saved_feature_names)]
    missing = [f for f in saved_feature_names if f not in work.columns]

    for col in saved_feature_names:
        if col in work.columns:
            work[col] = _kuquickml_safe_numeric_series(work[col])

    X_pred = work.reindex(columns=saved_feature_names)
    if X_pred.isnull().any().any():
        X_pred = X_pred.fillna(0)
    return X_pred, missing

def _kuquickml_loadUnknownSample_safe(self):
    options = QFileDialog.Options()
    filename, _ = QFileDialog.getOpenFileName(
        self, "Open Unknown Sample CSV/XLSX File", "",
        "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)",
        options=options
    )
    if not filename:
        return

    try:
        self.unknown_data = _kuquickml_read_tabular_any(filename)
        unknown_df = self.unknown_data.copy()
        unknown_df.columns = [_kuquickml_normalize_header_only(c) for c in unknown_df.columns]

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

        if feature_names is not None:
            saved_feature_names = [str(c).strip() for c in list(feature_names)]
        elif hasattr(model, "feature_names_in_"):
            saved_feature_names = [str(c).strip() for c in list(model.feature_names_in_)]
        else:
            saved_feature_names = None

        if 'Sample' in unknown_df.columns and (saved_feature_names is None or 'Sample' not in saved_feature_names):
            sample_series = unknown_df['Sample']
            unknown_df = unknown_df.drop(columns=['Sample'])
        else:
            sample_series = pd.Series([f"Sample {i + 1}" for i in range(len(unknown_df))])

        if saved_feature_names is not None:
            data_to_scale, missing_features = _kuquickml_prepare_unknown_prediction_frame(unknown_df, saved_feature_names)
            if missing_features:
                preview = ", ".join(map(str, missing_features[:10]))
                if len(missing_features) > 10:
                    preview += ", ..."
                warning_lines = [
                    "Some required feature columns used to train the model were not found in the loaded file.",
                    "",
                    "Leading/trailing spaces are ignored, but all other name differences are treated as different features.",
                    "For example, 'Butyl angelate' and 'Butyl angelate .1' are treated as different columns.",
                    "",
                    f"Required feature count: {len(saved_feature_names)}",
                    f"Loaded column count: {len(unknown_df.columns)}",
                    "",
                    f"Missing required features ({len(missing_features)}): {preview}",
                    "",
                    "Extra columns are allowed and column order will be aligned automatically."
                ]
                QMessageBox.warning(self, "Feature Mismatch", "\n".join(warning_lines))
        else:
            data_to_scale = unknown_df.copy()
            for col in data_to_scale.columns:
                data_to_scale[col] = _kuquickml_safe_numeric_series(data_to_scale[col])

        if scaler is not None:
            data_scaled = scaler.transform(data_to_scale)
            data_scaled = pd.DataFrame(data_scaled, columns=data_to_scale.columns, index=data_to_scale.index)
        else:
            data_scaled = data_to_scale

        if reducer is not None:
            data_reduced = reducer.transform(data_scaled)
            data_used = pd.DataFrame(
                data_reduced,
                columns=[f"Component {i + 1}" for i in range(data_reduced.shape[1])],
                index=data_scaled.index
            )
        else:
            data_used = data_scaled

        predictions = model.predict(data_used)

        if label_mapping:
            inverse_map = {v: k for k, v in label_mapping.items()}
            predictions = [inverse_map.get(p, p) for p in predictions]

        self.prediction_table.clear()
        self.prediction_table.setColumnCount(2)
        self.prediction_table.setHorizontalHeaderLabels(["Sample", "Prediction"])
        self.prediction_table.setRowCount(len(predictions))
        for i, pred in enumerate(predictions):
            self.prediction_table.setItem(i, 0, QTableWidgetItem(str(sample_series.iloc[i])))
            self.prediction_table.setItem(i, 1, QTableWidgetItem(str(pred)))

        self.tabs.setCurrentWidget(self.predictionTab)
    except Exception as e:
        QMessageBox.critical(self, "Prediction Error", f"Failed to run prediction:\n{e}")

CsvViewer.loadCsv = _kuquickml_safe_load_csvviewer
MyApp.loadCsv = _kuquickml_loadCsv_dialog_safe
MyApp.loadUnknownSample = _kuquickml_loadUnknownSample_safe
# ===== end safe tabular loading patch =====



# ===== v3.79 patches: KNN plot/obs, faster importance, compare layout =====
from matplotlib.lines import Line2D as _V379_Line2D

def _v379_sample_for_importance(X, y, max_rows=200):
    try:
        n = len(X)
    except Exception:
        return X, y
    if n <= max_rows:
        return X, y
    idx = np.linspace(0, n - 1, max_rows, dtype=int)
    if isinstance(X, pd.DataFrame):
        Xs = X.iloc[idx].copy()
    else:
        Xs = np.asarray(X)[idx]
    ys = np.asarray(y)[idx]
    return Xs, ys

# faster generic permutation helper used by later patches
_def_old_perm = _kuquickml_permutation

def _kuquickml_permutation(estimator, X, y, task='classification', n_repeats=5):
    scoring = 'accuracy' if task == 'classification' else 'r2'
    X_df = _kuquickml_to_numeric_df(X)
    X_df, y_arr = _v379_sample_for_importance(X_df, np.asarray(y), max_rows=200)
    res = permutation_importance(estimator, X_df, y_arr, n_repeats=n_repeats, random_state=42, scoring=scoring, n_jobs=-1)
    return np.asarray(res.importances_mean, dtype=float)

# KNN plot: remove misleading contour legend entirely

def _v379_plotResults(self, name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors,
                    score_value=None, score_label=None):
    plt.figure()
    ax = plt.gca()
    unique_labels = np.unique(np.concatenate((y_train, y_test)))
    label_colors = {
        label: "#{:02x}{:02x}{:02x}".format(
            random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for label in unique_labels}
    legend_widget = getattr(self, "legendNameInput", None)
    legend_name = legend_widget.text() if (legend_widget is not None and legend_widget.text()) else "Label"
    for label in unique_labels:
        train_indices = np.where(y_train == label)[0]
        test_indices = np.where(y_test == label)[0]
        color = label_colors[label]
        if len(train_indices) > 0:
            plt.scatter(X_train_embedded[train_indices, 0], X_train_embedded[train_indices, 1], c=color, s=30,
                        marker='o', alpha=0.5, label=f"Train {legend_name} {label}")
        if len(test_indices) > 0:
            plt.scatter(X_test_embedded[test_indices, 0], X_test_embedded[test_indices, 1], c=color, s=30,
                        marker='x', alpha=0.5, label=f"Test {legend_name} {label}")
    legend = ax.legend()
    if legend is not None:
        legend.set_draggable(True)
    title = f"{name} - KNN (k={n_neighbors})"
    if score_value is not None and score_label is not None:
        title += f"\n{score_label} = {score_value:.3f}"
    plt.title(title, fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
    plt.xlabel("Component 1", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
    plt.ylabel("Component 2", fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
    plt.show()

MyApp.plotResults = _v379_plotResults

# KNN classification: also show observed vs predicted graph and use faster importance path
_OLD_createClassificationModel_v379 = MyApp.createClassificationModel

def _v379_createClassificationModel(self):
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
    if reducer is not None:
        reducer.fit(X_train_numeric.values, y_train)
        X_train_embedded = reducer.transform(X_train_numeric.values)
        X_test_embedded = reducer.transform(X_test_numeric.values)
    else:
        X_train_embedded = X_train_numeric.values
        X_test_embedded = X_test_numeric.values
    knn.fit(X_train_embedded, y_train)
    accuracy = knn.score(X_test_embedded, y_test)
    if hasattr(X_train_embedded, "shape") and len(X_train_embedded.shape) == 2 and X_train_embedded.shape[1] == 2:
        self.plotResults(method_name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors,
                         score_value=accuracy, score_label="Test accuracy")
    y_pred_train = knn.predict(X_train_embedded)
    y_pred_test = knn.predict(X_test_embedded)
    try:
        self.plotObservedVsPredicted(
            y_train, y_pred_train, y_test, y_pred_test,
            f"KNN Classification Observed vs Predicted\nTrain Accuracy={accuracy_score(y_train, y_pred_train):.3f}, Test Accuracy={accuracy:.3f}"
        )
    except Exception:
        pass
    cm = confusion_matrix(y_test, y_pred_test)
    unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
    true = [f'true_{label}' for label in unique_labels]
    pred = [f'pred_{label}' for label in unique_labels]
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
        precision = np.nan_to_num(precision)
    cm_df = pd.DataFrame(cm, index=true, columns=pred)
    cm_df['Prediction Accuracy (%)'] = precision
    self.showConfusionMatrix(cm_df)
    feature_names = list(X_train_numeric.columns)
    self.showKNNPermutationImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix="KNN Classification", task="classification")
    self.models["KNN Classification"] = {
        "model": knn,
        "scaler": self._get_bundle_scaler(),
        "reducer": reducer,
        "feature_names": feature_names,
        "label_mapping": self._get_label_mapping()
    }
    if reducer:
        self.model_reducers["KNN Classification"] = reducer

MyApp.createClassificationModel = _v379_createClassificationModel

# faster MLP importance only: parallel + sample and keep existing UI
_OLD_createMLPClassificationModel_v379 = MyApp.createMLPClassificationModel

def _v379_createMLPClassificationModel(self):
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
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=int(self.max_iter_input.value()),
                        random_state=int(self.random_state_input.value()), alpha=alpha,
                        solver=self.solver_input.currentText(), activation=self.activation_input.currentText(),
                        learning_rate_init=learning_rate)
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
    cm_df = pd.DataFrame(cm, index=[f"Actual {label}" for label in labels], columns=[f"Predicted {label}" for label in labels])
    cm_df["Prediction Accuracy (%)"] = precision
    self.showConfusionMatrix(cm_df)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)
    X_imp, y_imp = _v379_sample_for_importance(X_test_used, y_test, max_rows=200)
    perm_importance = permutation_importance(mlp, X_imp, y_imp, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
    feature_importances = [(feature_names[idx], perm_importance.importances_mean[idx]) for idx in sorted_idx]
    self.showMLPFeatureImportances(feature_importances)
    self.models["MLP Classification"] = {"model": mlp, "scaler": self._get_bundle_scaler(), "reducer": None, "feature_names": feature_names, "label_mapping": self._get_label_mapping()}

MyApp.createMLPClassificationModel = _v379_createMLPClassificationModel

_OLD_createMLPRegressionModel_v379 = MyApp.createMLPRegressionModel

def _v379_createMLPRegressionModel(self):
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
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=int(self.max_iter_input.value()),
                       random_state=int(self.random_state_input.value()), alpha=alpha,
                       solver=self.solver_input.currentText(), activation=self.activation_input.currentText(),
                       learning_rate_init=learning_rate)
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
    X_imp, y_imp = _v379_sample_for_importance(X_test_used, y_test, max_rows=200)
    perm_importance = permutation_importance(mlp, X_imp, y_imp, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
    feature_importances = [(feature_names[idx], perm_importance.importances_mean[idx]) for idx in sorted_idx]
    self.showMLPFeatureImportances(feature_importances)
    self.models["MLP Regression"] = {"model": mlp, "scaler": self._get_bundle_scaler(), "reducer": None, "feature_names": feature_names, "label_mapping": None}

MyApp.createMLPRegressionModel = _v379_createMLPRegressionModel

# Faster KNN/SVM importance window using sampled test rows + parallel permutation, and same alt MI for SVM/KNN

def _v379_showKNNPermutationImportance(self, reducer, X_eval, y_eval, model, feature_names, title_prefix="KNN", task="classification"):
    try:
        X_eval = _kuquickml_to_numeric_df(X_eval)
        X_eval, y_small = _v379_sample_for_importance(X_eval, y_eval, max_rows=200)
        wrapped = _ReducerWrappedEstimator(model, reducer)
        scoring = "accuracy" if task == "classification" else "r2"
        result = permutation_importance(wrapped, X_eval, y_small, n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1)
        perm_vals = np.asarray(result.importances_mean, dtype=float)
        alt_vals = _kuquickml_safe_mutual_info(X_eval, y_small, task=task)
        title = f"{title_prefix}"
        _kuquickml_show_dual_importance_dialog(self, title, feature_names, perm_vals, alt_vals, "Alternative (mutual information)")
    except Exception as e:
        QMessageBox.warning(self, "Permutation Importance Error", f"Failed to compute feature importance:\n{e}")

MyApp.showKNNPermutationImportance = _v379_showKNNPermutationImportance

# SVM importance patch if function exists
if hasattr(MyApp, 'showSVMPermutationImportance'):
    def _v379_showSVMPermutationImportance(self, reducer, X_eval, y_eval, model, feature_names, title_prefix="SVM", task="classification"):
        try:
            X_eval = _kuquickml_to_numeric_df(X_eval)
            X_eval, y_small = _v379_sample_for_importance(X_eval, y_eval, max_rows=200)
            wrapped = _ReducerWrappedEstimator(model, reducer)
            scoring = "accuracy" if task == "classification" else "r2"
            result = permutation_importance(wrapped, X_eval, y_small, n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1)
            perm_vals = np.asarray(result.importances_mean, dtype=float)
            alt_vals = _kuquickml_safe_mutual_info(X_eval, y_small, task=task)
            _kuquickml_show_dual_importance_dialog(self, title_prefix, feature_names, perm_vals, alt_vals, "Alternative (mutual information)")
        except Exception as e:
            QMessageBox.warning(self, "Permutation Importance Error", f"Failed to compute feature importance:\n{e}")
    MyApp.showSVMPermutationImportance = _v379_showSVMPermutationImportance

# compare layout narrower with Split rows like requested

def _v379_fill_compare_table(table, rows, task):
    if task == 'regression':
        metric_keys = [('R2','r2'), ('RMSE','rmse'), ('MSE','mse')]
    else:
        metric_keys = [('Accuracy','accuracy'), ('F1','f1'), ('ROC-AUC','roc_auc')]
    cols = ['Model', 'Algorithm', 'Split'] + [m[0] for m in metric_keys]
    table.clear()
    table.setColumnCount(len(cols))
    table.setHorizontalHeaderLabels(cols)
    total_rows = len(rows) * 3
    table.setRowCount(total_rows)
    r = 0
    for row in rows:
        file_name = str(row.get('file_name','-'))
        algo = str(row.get('algorithm_name','-'))
        sections = [('Train', row.get('training_metrics',{})), ('Test', row.get('test_metrics',{})), ('CV', row.get('cv_metrics',{}))]
        table.setSpan(r, 0, 3, 1)
        table.setSpan(r, 1, 3, 1)
        table.setItem(r, 0, QTableWidgetItem(file_name))
        table.setItem(r, 1, QTableWidgetItem(algo))
        for offset, (split_name, metrics) in enumerate(sections):
            rr = r + offset
            table.setItem(rr, 2, QTableWidgetItem(split_name))
            for j, (_, key) in enumerate(metric_keys, start=3):
                value = metrics.get(key) if isinstance(metrics, dict) else None
                text = _kuquickml_fmt_metric_value(value)
                table.setItem(rr, j, QTableWidgetItem(text))
        r += 3
    try:
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
    except Exception:
        pass

_kuquickml_fill_compare_table = _v379_fill_compare_table

# remove compare full copy/save buttons from layout when opening tab

def _v379_setup_compare_tab(self):
    layout = QVBoxLayout()
    note1 = QLabel('저장된 모델들의 Train / Test / CV 성능을 비교합니다. 여러 모델 파일을 불러온 뒤 비교 버튼을 누르세요.')
    note2 = QLabel('분류 모델은 분류끼리, 회귀 모델은 회귀끼리 비교됩니다.')
    note3 = QLabel('다중클래스 ROC-AUC는 One-vs-Rest(OVR) 방식과 macro 평균으로 계산합니다.')
    for n in (note1, note2, note3):
        n.setWordWrap(True)
        layout.addWidget(n)
    btn_layout = QHBoxLayout()
    self.compareLoadModelsBtn = QPushButton('모델 파일 불러오기')
    self.compareRunBtn = QPushButton('비교 실행')
    self.compareLoadModelsBtn.clicked.connect(lambda: _kuquickml_load_compare_models(self))
    self.compareRunBtn.clicked.connect(lambda: _kuquickml_run_compare_models(self))
    btn_layout.addWidget(self.compareLoadModelsBtn)
    btn_layout.addWidget(self.compareRunBtn)
    layout.addLayout(btn_layout)
    layout.addWidget(QLabel('선택된 모델 파일'))
    self.compareModelListWidget = QListWidget()
    self.compareModelListWidget.setSelectionMode(QListWidget.ExtendedSelection)
    layout.addWidget(self.compareModelListWidget)
    layout.addWidget(QLabel('Classification Models'))
    self.compareClassificationTable = QTableWidget()
    _kuquickml_enable_copyable_table(self.compareClassificationTable)
    layout.addWidget(self.compareClassificationTable)
    layout.addWidget(QLabel('Regression Models'))
    self.compareRegressionTable = QTableWidget()
    _kuquickml_enable_copyable_table(self.compareRegressionTable)
    layout.addWidget(self.compareRegressionTable)
    self.compareTab.setLayout(layout)
    self.compare_model_paths = []

_kuquickml_setup_compare_tab = _v379_setup_compare_tab

# star menu names for Prediction and Compare Models
_old_initUI_v379_star = MyApp.initUI

def _v379_initUI_with_star(self):
    _old_initUI_v379_star(self)
    try:
        for action in self.menuBar().actions():
            txt = action.text()
            if txt == '5. Prediction':
                action.setText('★5. Prediction')
            elif txt == '6. Compare Models':
                action.setText('★6. Compare Models')
    except Exception:
        pass

MyApp.initUI = _v379_initUI_with_star
# ===== end v3.79 patches =====




# ===== v3.82b model-specific importance + save-log fix =====
def _v382b_show_single_importance_dialog(self, title, feature_names, values, method_name, intro_html=''):
    dialog = QDialog(self)
    dialog.setWindowTitle(title)
    dialog.resize(900, 700)
    layout = QVBoxLayout(dialog)

    if intro_html:
        lbl = QLabel(intro_html)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

    table = QTableWidget()
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(['Feature', 'Importance'])
    vals = np.asarray(values, dtype=float)
    names = list(feature_names)
    n = min(len(names), len(vals))
    order = np.argsort(np.nan_to_num(vals[:n], nan=-np.inf))[::-1]
    table.setRowCount(n)
    for row, idx in enumerate(order):
        table.setItem(row, 0, QTableWidgetItem(str(names[idx])))
        table.setItem(row, 1, QTableWidgetItem(_kuquickml_format_importance_value(vals[idx])))
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    _kuquickml_enable_copyable_table(table)
    layout.addWidget(table)

    note = QLabel("셀을 드래그해 선택한 뒤 우클릭 복사 또는 Ctrl+C를 사용할 수 있습니다.")
    note.setWordWrap(True)
    layout.addWidget(note)

    btn = QPushButton("Close")
    btn.clicked.connect(dialog.close)
    layout.addWidget(btn)

    dialog.setModal(False)
    dialog.setWindowModality(Qt.NonModal)
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    if not hasattr(self, "_non_modal_dialogs"):
        self._non_modal_dialogs = []
    self._non_modal_dialogs.append(dialog)
    dialog.destroyed.connect(lambda *args: self._non_modal_dialogs.remove(dialog) if dialog in getattr(self, "_non_modal_dialogs", []) else None)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

def _v382b_showKNNImportance(self, reducer, X_eval, y_eval, model, feature_names, title_prefix='KNN', task='classification'):
    try:
        vals = _kuquickml_mutual_info(X_eval, y_eval, task=task)
        intro = (
            f"<b>{title_prefix}</b><br>"
            "이 화면의 중요도는 <b>mutual information</b>으로 계산됩니다.<br>"
            "각 feature가 target과 얼마나 관련되어 있는지를 나타내는 값입니다.<br>"
            "값이 클수록 해당 feature가 target을 구분하는 데 더 많은 정보를 담고 있습니다.<br>"
            "고정된 유의 기준값은 없으며, 같은 데이터 내 다른 feature와의 상대적 크기와 순위로 해석하는 것이 적절합니다."
        )
        _v382b_show_single_importance_dialog(self, 'Feature Importances', feature_names, vals, 'mutual information', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Feature Importance Error', f'Failed to compute feature importance:\n{e}')

def _v382b_showSVMImportance(self, kernel, reducer, X_test, y_test, model, feature_names, title_prefix='SVM', task='classification'):
    try:
        vals = _kuquickml_mutual_info(X_test, y_test, task=task)
        intro = (
            f"<b>{title_prefix}</b><br>"
            "이 화면의 중요도는 <b>mutual information</b>으로 계산됩니다.<br>"
            "각 feature가 target과 얼마나 관련되어 있는지를 나타내는 값입니다.<br>"
            "값이 클수록 해당 feature가 target을 구분하는 데 더 많은 정보를 담고 있습니다.<br>"
            "고정된 유의 기준값은 없으며, 같은 데이터 내 다른 feature와의 상대적 크기와 순위로 해석하는 것이 적절합니다."
        )
        _v382b_show_single_importance_dialog(self, 'Feature Importances', feature_names, vals, 'mutual information', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Feature Importance Error', f'Failed to compute feature importance:\n{e}')

def _v382b_showMLPFeatureImportances(self, feature_importances, alternative_importances=None, alternative_name='input-layer mean abs weight'):
    try:
        if alternative_importances:
            names = [n for n, _ in alternative_importances]
            vals = [v for _, v in alternative_importances]
        else:
            names = [n for n, _ in feature_importances]
            vals = [v for _, v in feature_importances]
        intro = (
            "<b>MLP</b><br>"
            "이 화면의 중요도는 <b>input-layer mean abs weight</b>로 계산됩니다.<br>"
            "각 입력 feature가 첫 번째 은닉층으로 연결될 때의 가중치 절댓값 평균을 사용합니다.<br>"
            "값이 클수록 모델이 그 feature를 더 크게 활용했을 가능성을 의미합니다.<br>"
            "이는 heuristic 해석 지표이며, feature와 target의 직접적인 통계적 관련성보다 "
            "모델 내부 가중치의 크기를 기준으로 해석합니다."
        )
        _v382b_show_single_importance_dialog(self, 'Feature Importances', names, vals, 'input-layer mean abs weight', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Feature Importance Error', f'Failed to compute feature importance:\n{e}')

def _v382b_createMLPClassificationModel(self):
    if not self.checkDataSplit():
        return
    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()
    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    X_train_used = X_train_numeric.values
    X_test_used = X_test_numeric.values
    hidden_layer_input_text = self.hidden_layer_input.text().strip()
    hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(',')))
    alpha_input_text = self.alpha_input.text().strip()
    alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)
    lr_input_text = self.learning_rate_input.text().strip()
    learning_rate = 0.001 if not lr_input_text else float(lr_input_text)
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=int(self.max_iter_input.value()), random_state=int(self.random_state_input.value()), alpha=alpha, solver=self.solver_input.currentText(), activation=self.activation_input.currentText(), learning_rate_init=learning_rate)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
        mlp.fit(X_train_used, y_train)
    if mlp.n_iter_ == mlp.max_iter:
        QMessageBox.warning(self, 'Iteration Warning', 'Maximum iterations reached. Consider increasing max_iter.')
    y_pred_train = mlp.predict(X_train_used)
    y_pred_test = mlp.predict(X_test_used)
    cm = confusion_matrix(y_test, y_pred_test)
    acc = accuracy_score(y_test, y_pred_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    if len(np.unique(y_test)) > 2:
        average = 'macro'
    else:
        average = 'binary'
    f1 = f1_score(y_test, y_pred_test, average=average)
    cm_df = pd.DataFrame(cm)
    self.showConfusionMatrix(cm_df)
    self.plotMLPClassificationResults(X_train_used, y_train, X_test_used, y_test, y_pred_test, train_acc, acc, cm)
    alt_vals = _kuquickml_mlp_abs_weight(mlp)
    alternative_importances = [(feature_names[idx], float(alt_vals[idx])) for idx in np.argsort(alt_vals)[::-1]]
    self.showMLPFeatureImportances([], alternative_importances=alternative_importances, alternative_name='input-layer mean abs weight')
    self.models['MLP Classification'] = {'model': mlp, 'scaler': self._get_bundle_scaler(), 'reducer': None, 'feature_names': feature_names, 'label_mapping': self._get_label_mapping()}

def _v382b_createMLPRegressionModel(self):
    if not self.checkDataSplit():
        return
    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()
    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    X_train_used = X_train_numeric.values
    X_test_used = X_test_numeric.values
    hidden_layer_input_text = self.hidden_layer_input.text().strip()
    hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(',')))
    alpha_input_text = self.alpha_input.text().strip()
    alpha = 0.0001 if not alpha_input_text else float(alpha_input_text)
    lr_input_text = self.learning_rate_input.text().strip()
    learning_rate = 0.001 if not lr_input_text else float(lr_input_text)
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=int(self.max_iter_input.value()), random_state=int(self.random_state_input.value()), alpha=alpha, solver=self.solver_input.currentText(), activation=self.activation_input.currentText(), learning_rate_init=learning_rate)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
        mlp.fit(X_train_used, y_train)
    if mlp.n_iter_ == mlp.max_iter:
        QMessageBox.warning(self, 'Iteration Warning', 'Maximum iterations reached. Consider increasing max_iter.')
    y_pred_train = mlp.predict(X_train_used)
    y_pred_test = mlp.predict(X_test_used)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)
    alt_vals = _kuquickml_mlp_abs_weight(mlp)
    alternative_importances = [(feature_names[idx], float(alt_vals[idx])) for idx in np.argsort(alt_vals)[::-1]]
    self.showMLPFeatureImportances([], alternative_importances=alternative_importances, alternative_name='input-layer mean abs weight')
    self.models['MLP Regression'] = {'model': mlp, 'scaler': self._get_bundle_scaler(), 'reducer': None, 'feature_names': feature_names, 'label_mapping': None}

# override lingering permutation-based handlers
MyApp.showKNNPermutationImportance = _v382b_showKNNImportance
MyApp.showSVMImportanceUnavailable = _v382b_showSVMImportance
MyApp.showMLPFeatureImportances = _v382b_showMLPFeatureImportances
# Reverted MLP monkeypatch to last known working implementation
MyApp.createMLPClassificationModel = _v379_createMLPClassificationModel
MyApp.createMLPRegressionModel = _v379_createMLPRegressionModel
def _v382b_no_permutation(*args, **kwargs):
    raise RuntimeError('Permutation importance is disabled in this version.')
_kuquickml_permutation = _v382b_no_permutation
# ===== end v3.82b patch =====

# ===== Save sidecar TXT log patch =====
import os as _os_for_log
import inspect as _inspect_for_log

def _kuquickml_get_current_feature_context(self, feature_names):
    try:
        X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
        y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()
        if hasattr(self, '_drop_sample_and_numeric'):
            X_num = self._drop_sample_and_numeric(X_test).fillna(0)
        else:
            X_num = X_test.select_dtypes(include=[np.number]).fillna(0)
        # align by feature names if possible
        if feature_names:
            avail = [c for c in feature_names if c in X_num.columns]
            X_num = X_num.reindex(columns=avail)
            feature_names = avail
        return X_num, y_test, feature_names
    except Exception:
        return None, None, feature_names

def _kuquickml_compute_feature_importance_for_log(self, enriched):
    try:
        model = enriched.get('model')
        feature_names = list(enriched.get('feature_names') or [])
        if model is None or not feature_names:
            return {'method': None, 'top_features': []}
        wrapper_name = type(model).__name__
        base_model = model.estimator if hasattr(model, 'estimator') else model
        alg_name = type(base_model).__name__.lower()
        X_ctx, y_ctx, feature_names = _kuquickml_get_current_feature_context(self, feature_names)
        values = None
        method = None
        if hasattr(base_model, 'feature_importances_'):
            values = np.asarray(base_model.feature_importances_, dtype=float)
            method = 'feature_importances_'
        elif 'mlp' in alg_name and hasattr(base_model, 'coefs_') and len(base_model.coefs_) > 0:
            values = np.mean(np.abs(base_model.coefs_[0]), axis=1)
            method = 'input-layer mean abs weight'
        elif ('kneighbors' in alg_name or 'svc' in alg_name or 'svr' in alg_name) and X_ctx is not None and y_ctx is not None:
            task = enriched.get('task', 'classification')
            values = _kuquickml_mutual_info(X_ctx, y_ctx, task=task)
            method = 'mutual information'
        if values is None:
            return {'method': None, 'top_features': []}
        vals = np.asarray(values, dtype=float)
        n = min(len(vals), len(feature_names))
        order = np.argsort(np.nan_to_num(vals[:n], nan=-np.inf))[::-1]
        top = []
        for idx in order[:20]:
            top.append((str(feature_names[idx]), float(vals[idx])))
        return {'method': method, 'top_features': top}
    except Exception as e:
        return {'method': f'failed: {e}', 'top_features': []}


def _kuquickml_unwrap_estimator_for_log(model):
    """Return (wrapper_name, base_estimator) for logging."""
    try:
        cls_name = type(model).__name__
        if cls_name in ("OneVsRestClassifier", "OneVsOneClassifier") and hasattr(model, 'estimator'):
            return cls_name, model.estimator
    except Exception:
        pass
    return None, model


def _kuquickml_default_params_for_estimator(estimator):
    try:
        est_cls = estimator.__class__
        default_est = est_cls()
        return default_est.get_params(deep=False)
    except Exception:
        return {}


def _kuquickml_format_param_value(v):
    try:
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)
    except Exception:
        return repr(v)


def _kuquickml_estimator_family_name(est):
    name = type(est).__name__ if est is not None else ''
    lname = name.lower()
    if 'kneighbors' in lname or 'knn' in lname:
        return 'knn'
    if 'mlp' in lname:
        return 'mlp'
    if 'svc' in lname or 'svr' in lname:
        return 'svm'
    if 'randomforest' in lname:
        return 'rf'
    return 'other'



def _kuquickml_hyperparam_sentence(base_est, max_items=6):
    try:
        if base_est is None or not hasattr(base_est, 'get_params'):
            return 'Estimator hyperparameters were not available.'
        params = base_est.get_params(deep=False)
        defaults = _kuquickml_default_params_for_estimator(base_est)
        changed = []
        for k, v in params.items():
            if k in defaults and defaults.get(k) != v:
                changed.append(f"{k}={_kuquickml_format_param_value(v)}")
        if not changed:
            return 'No estimator hyperparameters differed from sklearn defaults.'
        shown = changed[:max_items]
        tail = '' if len(changed) <= max_items else f"; additional non-default settings: {len(changed) - max_items}"
        return 'Key hyperparameter settings were ' + ', '.join(shown) + tail + '.'
    except Exception:
        return 'Estimator hyperparameters could not be summarized.'

def _kuquickml_algorithm_summary_paragraph(base_est, task, scaler, reducer, cv_settings, train_metrics, test_metrics, cv_metrics):
    algo = _kuquickml_estimator_family_name(base_est)
    algo_label = type(base_est).__name__ if base_est is not None else 'model'
    scaler_name = type(scaler).__name__ if scaler is not None else 'no scaler'
    reducer_name = type(reducer).__name__ if reducer is not None else 'no dimensionality reduction'
    split_label = f"{cv_settings.get('n_splits', 5)}-fold cross-validation" if isinstance(cv_settings, dict) and cv_settings else 'cross-validation'
    hyperparam_sentence = _kuquickml_hyperparam_sentence(base_est)
    train_acc = train_metrics.get('accuracy')
    test_acc = test_metrics.get('accuracy')
    cv_acc = cv_metrics.get('accuracy')
    train_f1 = train_metrics.get('f1')
    test_f1 = test_metrics.get('f1')
    cv_f1 = cv_metrics.get('f1')
    train_auc = train_metrics.get('roc_auc')
    test_auc = test_metrics.get('roc_auc')
    cv_auc = cv_metrics.get('roc_auc')
    train_r2 = train_metrics.get('r2')
    test_r2 = test_metrics.get('r2')
    cv_r2 = cv_metrics.get('r2')
    train_rmse = train_metrics.get('rmse')
    test_rmse = test_metrics.get('rmse')
    cv_rmse = cv_metrics.get('rmse')
    train_mse = train_metrics.get('mse')
    test_mse = test_metrics.get('mse')
    cv_mse = cv_metrics.get('mse')
    train_mae = train_metrics.get('mae')
    test_mae = test_metrics.get('mae')
    cv_mae = cv_metrics.get('mae')

    def fmt(v):
        if isinstance(v, dict):
            mean = v.get('mean')
            std = v.get('std')
            if mean is None:
                return 'not available'
            if std is None:
                return f"{mean:.4f}"
            return f"{mean:.4f} (SD {std:.4f})"
        if v is None:
            return 'not available'
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    if task == 'classification':
        importance_method = {
            'knn': 'mutual information, an external relevance measure that quantifies the statistical association between each feature and the class labels',
            'mlp': 'the mean absolute input-layer weight, a heuristic importance measure derived from the average absolute connection strength between each input feature and the first hidden layer',
            'svm': 'mutual information, used here as an external importance measure for SVM-based models',
            'rf': "the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the forest",
        }.get(algo, 'an algorithm-specific feature relevance summary')
        return (
            f"This saved model is a {algo_label} classification model trained using {scaler_name} preprocessing and {reducer_name}. "
            f"{hyperparam_sentence} "
            f"Feature relevance was summarized with {importance_method}. "
            f"Model performance was summarized with training accuracy {fmt(train_acc)}, test accuracy {fmt(test_acc)}, and {split_label} accuracy {fmt(cv_acc)}; "
            f"training F1-score {fmt(train_f1)}, test F1-score {fmt(test_f1)}, and {split_label} F1-score {fmt(cv_f1)} were also recorded. "
            f"When available, ROC-AUC was recorded as training {fmt(train_auc)}, test {fmt(test_auc)}, and {split_label} {fmt(cv_auc)}; multiclass ROC-AUC uses a One-vs-Rest strategy with macro averaging."
        )
    else:
        importance_method = {
            'knn': 'mutual information, an external relevance measure that quantifies the statistical association between each feature and the response variable',
            'mlp': 'the mean absolute input-layer weight, a heuristic importance measure derived from the average absolute connection strength between each input feature and the first hidden layer',
            'svm': 'mutual information, used here as an external importance measure for SVM-based models',
            'rf': "the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the forest",
        }.get(algo, 'an algorithm-specific feature relevance summary')
        return (
            f"This saved model is a {algo_label} regression model trained using {scaler_name} preprocessing and {reducer_name}. "
            f"{hyperparam_sentence} "
            f"Feature relevance was summarized with {importance_method}. "
            f"Model performance was summarized with training R2 {fmt(train_r2)}, test R2 {fmt(test_r2)}, and {split_label} R2 {fmt(cv_r2)}; "
            f"training RMSE {fmt(train_rmse)}, test RMSE {fmt(test_rmse)}, and {split_label} RMSE {fmt(cv_rmse)} were also recorded. "
            f"Additional regression error summaries included training MSE {fmt(train_mse)}, test MSE {fmt(test_mse)}, {split_label} MSE {fmt(cv_mse)}, and training MAE {fmt(train_mae)}, test MAE {fmt(test_mae)}, and {split_label} MAE {fmt(cv_mae)}."
        )


def _kuquickml_build_model_log_text(self, model_name, enriched, filename):
    model = enriched.get('model')
    scaler = enriched.get('scaler')
    reducer = enriched.get('reducer')
    wrapper_name, base_est = _kuquickml_unwrap_estimator_for_log(model)
    task = enriched.get('task', '')
    feature_names = enriched.get('feature_names') or []
    saved_versions = enriched.get('saved_with_versions', {})
    train_metrics = enriched.get('training_metrics', {}) if isinstance(enriched.get('training_metrics'), dict) else {}
    test_metrics = enriched.get('test_metrics', {}) if isinstance(enriched.get('test_metrics'), dict) else {}
    cv_metrics = enriched.get('cv_metrics', {}) if isinstance(enriched.get('cv_metrics'), dict) else {}
    cv_settings = enriched.get('cv_settings', {}) if isinstance(enriched.get('cv_settings'), dict) else {}
    data_split_settings = enriched.get('data_split_settings', {}) if isinstance(enriched.get('data_split_settings'), dict) else {}
    fi_log = enriched.get('feature_importance_log', {}) if isinstance(enriched.get('feature_importance_log'), dict) else {}

    lines = []
    lines.append('KUickML Model Save Log')
    lines.append('=' * 60)
    lines.append(f"Saved model file: {filename}")
    lines.append(f"Saved at: {enriched.get('saved_at', '')}")
    lines.append(f"Model name in app: {model_name}")
    lines.append(f"Task: {task}")
    lines.append(f"Algorithm: {type(base_est).__name__ if base_est is not None else type(model).__name__}")
    if wrapper_name:
        lines.append(f"Wrapper / multiclass strategy: {wrapper_name}")
    lines.append('')

    lines.append('[Preprocessing]')
    lines.append(f"Scaler used: {type(scaler).__name__ if scaler is not None else 'None'}")
    lines.append(f"Dimensionality reduction used: {type(reducer).__name__ if reducer is not None else 'None'}")
    if scaler is not None and hasattr(self, 'scaler_name'):
        lines.append(f"Scaler display name: {getattr(self, 'scaler_name', '')}")
    if reducer is not None:
        try:
            lines.append(f"Reducer parameters: {reducer.get_params()}")
        except Exception:
            pass
    lines.append(f"Feature count: {len(feature_names)}")
    lines.append('')

    lines.append('[Data split]')
    for k, v in data_split_settings.items():
        lines.append(f"- {k}: {v}")
    if not data_split_settings:
        lines.append('- None')
    lines.append('')

    lines.append('[Hyperparameters]')
    if base_est is not None and hasattr(base_est, 'get_params'):
        params = base_est.get_params(deep=False)
        defaults = _kuquickml_default_params_for_estimator(base_est)
        changed = []
        defaulted = []
        for k, v in params.items():
            if k in defaults and defaults.get(k) != v:
                changed.append(f"{k} = {_kuquickml_format_param_value(v)}")
            else:
                defaulted.append(f"{k} = {_kuquickml_format_param_value(v)}")
        lines.append('Non-default parameters:')
        lines.extend(['  - ' + x for x in changed] if changed else ['  - None'])
        lines.append('Default parameters used as-is:')
        lines.extend(['  - ' + x for x in defaulted[:30]] if defaulted else ['  - None'])
    else:
        lines.append('  - Estimator parameters unavailable')
    lines.append('')

    lines.append('[Metrics]')
    lines.append('Training metrics:')
    for k, v in train_metrics.items():
        lines.append(f"  - {k}: {v}")
    lines.append('Test metrics:')
    for k, v in test_metrics.items():
        lines.append(f"  - {k}: {v}")
    lines.append('CV metrics:')
    for k, v in cv_metrics.items():
        lines.append(f"  - {k}: {v}")
    lines.append('CV settings:')
    for k, v in cv_settings.items():
        lines.append(f"  - {k}: {v}")
    lines.append('')

    lines.append('[Feature importance summary]')
    lines.append(f"Method: {fi_log.get('method')}")
    top_features = fi_log.get('top_features', [])
    if top_features:
        for rank, (fname, val) in enumerate(top_features, start=1):
            try:
                lines.append(f"  {rank}. {fname}: {float(val):.4f}")
            except Exception:
                lines.append(f"  {rank}. {fname}: {val}")
    else:
        lines.append('  - Not available')
    lines.append('')

    lines.append('[Environment]')
    for k, v in saved_versions.items():
        lines.append(f"  - {k}: {v}")
    lines.append('')
    lines.append('Note: multiclass ROC-AUC is computed using One-vs-Rest (OVR) with macro averaging when available.')
    lines.append('')
    lines.append('[Manuscript-ready summary]')
    lines.append(_kuquickml_algorithm_summary_paragraph(base_est, task, scaler, reducer, cv_settings, train_metrics, test_metrics, cv_metrics))



def _kuquickml_saveModel_with_metadata_and_txt(self, model_name, filename):
    bundle = self.models.get(model_name)
    if not isinstance(bundle, dict) or 'model' not in bundle:
        QMessageBox.warning(self, 'Error', 'Selected item is not a valid saved model bundle.')
        return
    try:
        enriched = _kuquickml_enrich_bundle_for_save(self, model_name, bundle)
        enriched['feature_importance_log'] = _kuquickml_compute_feature_importance_for_log(self, enriched)
        joblib.dump(enriched, filename)

        base, _ext = _os_for_log.path.splitext(filename)
        txt_path = base + '_log.txt'
        log_text = _kuquickml_build_model_log_text(self, model_name, enriched, filename)
        with open(txt_path, 'w', encoding='utf-8-sig') as f:
            f.write(log_text)

        scaler = enriched.get('scaler')
        reducer = enriched.get('reducer')
        QMessageBox.information(
            self,
            'Model Saved',
            f"Model '{model_name}' saved successfully.\n"
            f"Scaler: {type(scaler).__name__ if scaler else 'None'}\n"
            f"Reducer: {type(reducer).__name__ if reducer else 'None'}\n"
            f"Log file saved: {txt_path}"
        )
    except Exception as e:
        QMessageBox.warning(self, 'Error', f"Failed to save model:\n{e}")


MyApp.saveModel = _kuquickml_saveModel_with_metadata_and_txt

import os as _os_for_log
import inspect as _inspect_for_log


def _kuquickml_unwrap_estimator_for_log(model):
    """Return (wrapper_name, base_estimator) for logging."""
    try:
        cls_name = type(model).__name__
        if cls_name in ("OneVsRestClassifier", "OneVsOneClassifier") and hasattr(model, 'estimator'):
            return cls_name, model.estimator
    except Exception:
        pass
    return None, model


def _kuquickml_default_params_for_estimator(estimator):
    try:
        est_cls = estimator.__class__
        default_est = est_cls()
        return default_est.get_params(deep=False)
    except Exception:
        return {}


def _kuquickml_format_param_value(v):
    try:
        if isinstance(v, float):
            return f"{v:g}"
        return str(v)
    except Exception:
        return repr(v)


def _kuquickml_estimator_family_name(est):
    name = type(est).__name__ if est is not None else ''
    lname = name.lower()
    if 'kneighbors' in lname or 'knn' in lname:
        return 'knn'
    if 'mlp' in lname:
        return 'mlp'
    if 'svc' in lname or 'svr' in lname:
        return 'svm'
    if 'randomforest' in lname:
        return 'rf'
    return 'other'



def _kuquickml_hyperparam_sentence(base_est, max_items=6):
    try:
        if base_est is None or not hasattr(base_est, 'get_params'):
            return 'Estimator hyperparameters were not available.'
        params = base_est.get_params(deep=False)
        defaults = _kuquickml_default_params_for_estimator(base_est)
        changed = []
        for k, v in params.items():
            if k in defaults and defaults.get(k) != v:
                changed.append(f"{k}={_kuquickml_format_param_value(v)}")
        if not changed:
            return 'No estimator hyperparameters differed from sklearn defaults.'
        shown = changed[:max_items]
        tail = '' if len(changed) <= max_items else f"; additional non-default settings: {len(changed) - max_items}"
        return 'Key hyperparameter settings were ' + ', '.join(shown) + tail + '.'
    except Exception:
        return 'Estimator hyperparameters could not be summarized.'

def _kuquickml_algorithm_summary_paragraph(base_est, task, scaler, reducer, cv_settings, train_metrics, test_metrics, cv_metrics):
    algo = _kuquickml_estimator_family_name(base_est)
    algo_label = type(base_est).__name__ if base_est is not None else 'model'
    scaler_name = type(scaler).__name__ if scaler is not None else 'no scaler'
    reducer_name = type(reducer).__name__ if reducer is not None else 'no dimensionality reduction'
    split_label = f"{cv_settings.get('n_splits', 5)}-fold cross-validation" if isinstance(cv_settings, dict) and cv_settings else 'cross-validation'
    train_acc = train_metrics.get('accuracy')
    test_acc = test_metrics.get('accuracy')
    cv_acc = cv_metrics.get('accuracy')
    train_r2 = train_metrics.get('r2')
    test_r2 = test_metrics.get('r2')
    cv_r2 = cv_metrics.get('r2')

    def fmt(v):
        if isinstance(v, dict):
            mean = v.get('mean')
            std = v.get('std')
            if mean is None:
                return 'not available'
            if std is None:
                return f"{mean:.4f}"
            return f"{mean:.4f} (SD {std:.4f})"
        if v is None:
            return 'not available'
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    if task == 'classification':
        if algo == 'knn':
            return (
                f"This saved model is a {algo_label} classification model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with mutual information, which quantifies how strongly each feature is associated with the class labels in the input data rather than measuring a built-in model coefficient. "
                f"Model performance was summarized with training accuracy {fmt(train_acc)}, test accuracy {fmt(test_acc)}, and {split_label} accuracy {fmt(cv_acc)}. "
                f"When available, additional classification summaries such as F1-score and ROC-AUC were recorded alongside the saved model for later comparison."
            )
        if algo == 'mlp':
            return (
                f"This saved model is an {algo_label} classification model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with the mean absolute input-layer weight, which reflects the average magnitude of each feature's connection to the first hidden layer and should be interpreted as a heuristic importance measure rather than a direct causal effect. "
                f"Model performance was summarized with training accuracy {fmt(train_acc)}, test accuracy {fmt(test_acc)}, and {split_label} accuracy {fmt(cv_acc)}. "
                f"When available, F1-score and ROC-AUC were also stored to support later comparison across saved models."
            )
        if algo == 'svm':
            return (
                f"This saved model is an {algo_label} classification model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with mutual information, which measures the statistical association between each feature and the target labels and is used here as an external importance measure for SVM-based models. "
                f"Model performance was summarized with training accuracy {fmt(train_acc)}, test accuracy {fmt(test_acc)}, and {split_label} accuracy {fmt(cv_acc)}. "
                f"For multiclass problems, ROC-AUC is computed using a One-vs-Rest strategy with macro averaging when that metric is available."
            )
        if algo == 'rf':
            return (
                f"This saved model is a {algo_label} classification model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the forest. "
                f"Model performance was summarized with training accuracy {fmt(train_acc)}, test accuracy {fmt(test_acc)}, and {split_label} accuracy {fmt(cv_acc)}. "
                f"When available, F1-score and ROC-AUC were also stored to support later comparison across saved models."
            )
        return (
            f"This saved model is a classification model trained using {scaler_name} preprocessing and {reducer_name}. "
            f"Training, test, and {split_label} metrics were stored with the model to support later comparison. "
            f"Where available, F1-score and ROC-AUC were also recorded."
        )
    else:
        if algo == 'knn':
            return (
                f"This saved model is a {algo_label} regression model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with mutual information, which estimates how strongly each feature is associated with the response variable in the data rather than measuring a built-in model coefficient. "
                f"Model performance was summarized with training R2 {fmt(train_r2)}, test R2 {fmt(test_r2)}, and {split_label} R2 {fmt(cv_r2)}. "
                f"Additional regression metrics such as RMSE, MSE, and MAE were stored alongside the model for later comparison."
            )
        if algo == 'mlp':
            return (
                f"This saved model is an {algo_label} regression model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with the mean absolute input-layer weight, which reflects the average magnitude of each feature's connection to the first hidden layer and should be interpreted as a heuristic importance measure. "
                f"Model performance was summarized with training R2 {fmt(train_r2)}, test R2 {fmt(test_r2)}, and {split_label} R2 {fmt(cv_r2)}. "
                f"Additional regression metrics such as RMSE, MSE, and MAE were stored alongside the model for later comparison."
            )
        if algo == 'svm':
            return (
                f"This saved model is an {algo_label} regression model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with mutual information, which quantifies the statistical association between each feature and the response variable and is used here as an external importance measure for SVM-based regression models. "
                f"Model performance was summarized with training R2 {fmt(train_r2)}, test R2 {fmt(test_r2)}, and {split_label} R2 {fmt(cv_r2)}. "
                f"Additional regression metrics such as RMSE, MSE, and MAE were stored alongside the model for later comparison."
            )
        if algo == 'rf':
            return (
                f"This saved model is a {algo_label} regression model trained using {scaler_name} preprocessing and {reducer_name}. "
                f"Feature relevance was summarized with the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the forest. "
                f"Model performance was summarized with training R2 {fmt(train_r2)}, test R2 {fmt(test_r2)}, and {split_label} R2 {fmt(cv_r2)}. "
                f"Additional regression metrics such as RMSE, MSE, and MAE were stored alongside the model for later comparison."
            )
        return (
            f"This saved model is a regression model trained using {scaler_name} preprocessing and {reducer_name}. "
            f"Training, test, and {split_label} metrics were stored with the model to support later comparison."
        )


def _kuquickml_build_model_log_text(self, model_name, enriched, filename):
    model = enriched.get('model')
    scaler = enriched.get('scaler')
    reducer = enriched.get('reducer')
    wrapper_name, base_est = _kuquickml_unwrap_estimator_for_log(model)
    task = enriched.get('task', '')
    feature_names = enriched.get('feature_names') or []
    saved_versions = enriched.get('saved_with_versions', {})
    train_metrics = enriched.get('training_metrics', {}) if isinstance(enriched.get('training_metrics'), dict) else {}
    test_metrics = enriched.get('test_metrics', {}) if isinstance(enriched.get('test_metrics'), dict) else {}
    cv_metrics = enriched.get('cv_metrics', {}) if isinstance(enriched.get('cv_metrics'), dict) else {}
    cv_settings = enriched.get('cv_settings', {}) if isinstance(enriched.get('cv_settings'), dict) else {}
    data_split_settings = enriched.get('data_split_settings', {}) if isinstance(enriched.get('data_split_settings'), dict) else {}
    if not data_split_settings:
        try:
            data_split_settings = {
                'input_data_type': 'scaled' if getattr(self, 'last_split_used_scaled', False) else 'raw',
                'test_size': float(getattr(self, 'testSetRatioInput').text()) if hasattr(self, 'testSetRatioInput') else None,
                'random_state': getattr(self, 'random_state', 42) if hasattr(self, 'random_state') else 42,
                'stratify': bool(getattr(self, 'stratifySplitCheckBox').isChecked()) if hasattr(self, 'stratifySplitCheckBox') else None,
            }
        except Exception:
            data_split_settings = {}

    lines = []
    lines.append('KUickML Model Save Log')
    lines.append('=' * 60)
    lines.append(f"Saved model file: {filename}")
    lines.append(f"Saved at: {enriched.get('saved_at', '')}")
    lines.append(f"Model name in app: {model_name}")
    lines.append(f"Task: {task}")
    lines.append(f"Algorithm: {type(base_est).__name__ if base_est is not None else type(model).__name__}")
    if wrapper_name:
        lines.append(f"Wrapper / multiclass strategy: {wrapper_name}")
    lines.append('')

    lines.append('[Preprocessing]')
    lines.append(f"Scaler used: {type(scaler).__name__ if scaler is not None else 'None'}")
    lines.append(f"Dimensionality reduction used: {type(reducer).__name__ if reducer is not None else 'None'}")
    if scaler is not None and hasattr(self, 'scaler_name'):
        lines.append(f"Scaler display name: {getattr(self, 'scaler_name', '')}")
    if reducer is not None:
        try:
            lines.append(f"Reducer parameters: {reducer.get_params()}")
        except Exception:
            pass
    lines.append(f"Feature count: {len(feature_names)}")
    lines.append('')

    lines.append('[Data split]')
    for k, v in data_split_settings.items():
        lines.append(f"- {k}: {v}")
    if not data_split_settings:
        lines.append('- None')

    lines.append('')
    lines.append('[Hyperparameters]')
    if base_est is not None and hasattr(base_est, 'get_params'):
        actual_params = base_est.get_params(deep=False)
        default_params = _kuquickml_default_params_for_estimator(base_est)
        changed = []
        defaults_used = []
        for k in sorted(actual_params.keys()):
            actual = actual_params.get(k)
            if k in default_params:
                default = default_params.get(k)
                if actual != default:
                    changed.append((k, actual, default))
                else:
                    defaults_used.append((k, default))
            else:
                changed.append((k, actual, '[unknown default]'))

        if changed:
            lines.append('Parameters explicitly different from sklearn defaults:')
            for k, actual, default in changed:
                lines.append(f"- {k} = {_kuquickml_format_param_value(actual)}")
        else:
            lines.append('No estimator hyperparameters differed from sklearn defaults.')

        lines.append('')
        lines.append('Parameters using sklearn defaults:')
        if defaults_used:
            for k, default in defaults_used:
                lines.append(f"- {k} = {_kuquickml_format_param_value(default)}")
        else:
            lines.append('- None')
    else:
        lines.append('Estimator parameters could not be inspected.')

    lines.append('')
    lines.append('[Cross-validation settings]')
    for k, v in cv_settings.items():
        lines.append(f"- {k}: {v}")
    if not cv_settings:
        lines.append('- None')

    def _metric_block(title, metrics):
        lines.append('')
        lines.append(f'[{title}]')
        if not metrics:
            lines.append('- None')
            return
        for k, v in metrics.items():
            lines.append(f"- {k}: {_kuquickml_fmt_metric_value(v)}")

    _metric_block('Training metrics', train_metrics)
    _metric_block('Test metrics', test_metrics)
    _metric_block('CV metrics', cv_metrics)

    lines.append('')
    lines.append('[Environment]')
    for k, v in saved_versions.items():
        lines.append(f"- {k}: {v}")
    lines.append('')
    lines.append('Note: Parameters not listed as changed above were saved using sklearn default values at save time.')
    lines.append('Multiclass ROC-AUC, when available, is calculated using One-vs-Rest (OVR) with macro averaging.')
    lines.append('')
    lines.append('[Manuscript-ready summary]')
    lines.append(_kuquickml_algorithm_summary_paragraph(base_est, task, scaler, reducer, cv_settings, train_metrics, test_metrics, cv_metrics))
    return '\n'.join(lines)


def _kuquickml_saveModel_with_metadata_and_txt(self, model_name, filename):
    bundle = self.models.get(model_name)
    if not isinstance(bundle, dict) or 'model' not in bundle:
        QMessageBox.warning(self, 'Error', 'Selected item is not a valid saved model bundle.')
        return
    try:
        enriched = _kuquickml_enrich_bundle_for_save(self, model_name, bundle)
        joblib.dump(enriched, filename)

        base, _ext = _os_for_log.path.splitext(filename)
        txt_path = base + '_log.txt'
        log_text = _kuquickml_build_model_log_text(self, model_name, enriched, filename)
        with open(txt_path, 'w', encoding='utf-8-sig') as f:
            f.write(log_text)

        scaler = enriched.get('scaler')
        reducer = enriched.get('reducer')
        QMessageBox.information(
            self,
            'Model Saved',
            f"Model '{model_name}' saved successfully.\n"
            f"Scaler: {type(scaler).__name__ if scaler else 'None'}\n"
            f"Reducer: {type(reducer).__name__ if reducer else 'None'}\n"
            f"Log file saved: {txt_path}"
        )
    except Exception as e:
        QMessageBox.warning(self, 'Error', f"Failed to save model:\n{e}")


MyApp.saveModel = _kuquickml_saveModel_with_metadata_and_txt


# ===== Final save-log override to ensure split info + manuscript summary always included =====
def _kuquickml_build_model_log_text_final(self, model_name, enriched, filename):
    import os as _os_for_log2
    bundle = enriched if isinstance(enriched, dict) else {}
    base_model = bundle.get('model')
    scaler = bundle.get('scaler')
    reducer = bundle.get('reducer')
    task = bundle.get('task', 'unknown')
    algorithm_name = bundle.get('algorithm_name', type(base_model).__name__ if base_model is not None else 'Unknown')
    feature_names = bundle.get('feature_names', []) or []
    train_metrics = bundle.get('training_metrics', {}) or {}
    test_metrics = bundle.get('test_metrics', {}) or {}
    cv_metrics = bundle.get('cv_metrics', {}) or {}
    cv_settings = bundle.get('cv_settings', {}) or {}
    data_split_settings = bundle.get('data_split_settings', {}) or {}
    saved_versions = bundle.get('saved_with_versions', {}) or {}
    fi_log = bundle.get('feature_importance_log', {}) or {}
    saved_at = bundle.get('saved_at', (_dt.datetime.now().isoformat() if '_dt' in globals() and _dt is not None else ''))

    wrapper_name = None
    base_est = base_model
    try:
        if type(base_model).__name__ in ('OneVsRestClassifier', 'OneVsOneClassifier') and hasattr(base_model, 'estimator'):
            wrapper_name = type(base_model).__name__
            base_est = base_model.estimator
    except Exception:
        pass

    lines = []
    lines.append('KUickML Model Save Log')
    lines.append('=' * 60)
    lines.append(f'Saved model file: {filename}')
    lines.append(f'Saved at: {saved_at}')
    lines.append(f'Model name in app: {model_name}')
    lines.append(f'Task: {task}')
    lines.append(f'Algorithm: {type(base_est).__name__ if base_est is not None else algorithm_name}')
    if wrapper_name:
        lines.append(f'Wrapper / multiclass strategy: {wrapper_name}')
    lines.append('')

    lines.append('[Preprocessing]')
    lines.append(f"Scaler used: {type(scaler).__name__ if scaler is not None else 'None'}")
    lines.append(f"Dimensionality reduction used: {type(reducer).__name__ if reducer is not None else 'None'}")
    lines.append(f'Feature count: {len(feature_names)}')
    lines.append('')

    lines.append('[Data split]')
    if data_split_settings:
        for k, v in data_split_settings.items():
            lines.append(f'- {k}: {v}')
    else:
        lines.append('- None')
    lines.append('')

    lines.append('[Hyperparameters]')
    if base_est is not None and hasattr(base_est, 'get_params'):
        params = base_est.get_params(deep=False)
        defaults = _kuquickml_default_params_for_estimator(base_est)
        changed = []
        defaulted = []
        for k, v in params.items():
            if k in defaults and defaults.get(k) != v:
                changed.append(f"{k} = {_kuquickml_format_param_value(v)}")
            else:
                defaulted.append(f"{k} = {_kuquickml_format_param_value(v)}")
        lines.append('Non-default parameters:')
        lines.extend(['  - ' + x for x in changed] if changed else ['  - None'])
        lines.append('Default parameters used as-is:')
        lines.extend(['  - ' + x for x in defaulted[:30]] if defaulted else ['  - None'])
    else:
        lines.append('Estimator parameters could not be inspected.')
    lines.append('')

    def _metric_block(title, metrics):
        lines.append(f'[{title}]')
        if not metrics:
            lines.append('  - None')
        else:
            for k, v in metrics.items():
                lines.append(f'  - {k}: {v}')
        lines.append('')

    _metric_block('Metrics - Training', train_metrics)
    _metric_block('Metrics - Test', test_metrics)
    _metric_block('Metrics - CV', cv_metrics)

    lines.append('[Cross-validation settings]')
    if cv_settings:
        for k, v in cv_settings.items():
            lines.append(f'  - {k}: {v}')
    else:
        lines.append('  - None')
    lines.append('')

    lines.append('[Feature importance summary]')
    lines.append(f"Method: {fi_log.get('method')}")
    top_features = fi_log.get('top_features', [])
    if top_features:
        for rank, (fname, val) in enumerate(top_features, start=1):
            try:
                lines.append(f'  {rank}. {fname}: {float(val):.4f}')
            except Exception:
                lines.append(f'  {rank}. {fname}: {val}')
    else:
        lines.append('  - Not available')
    lines.append('')

    lines.append('[Environment]')
    for k, v in saved_versions.items():
        lines.append(f'  - {k}: {v}')
    lines.append('')
    lines.append('Note: multiclass ROC-AUC is computed using One-vs-Rest (OVR) with macro averaging when available.')
    lines.append('')
    lines.append('[Manuscript-ready summary]')
    try:
        lines.append(_kuquickml_algorithm_summary_paragraph(base_est, task, scaler, reducer, cv_settings, train_metrics, test_metrics, cv_metrics))
    except Exception as e:
        lines.append(f'Summary generation failed: {e}')
    return '\n'.join(lines)


def _kuquickml_saveModel_with_metadata_and_txt_final(self, model_name, filename):
    bundle = self.models.get(model_name)
    if not isinstance(bundle, dict) or 'model' not in bundle:
        QMessageBox.warning(self, 'Error', 'Selected item is not a valid saved model bundle.')
        return
    try:
        enriched = _kuquickml_enrich_bundle_for_save(self, model_name, bundle)
        enriched['feature_importance_log'] = _kuquickml_compute_feature_importance_for_log(self, enriched)
        # ensure split settings always present
        if 'data_split_settings' not in enriched:
            enriched['data_split_settings'] = {
                'input_data_type': 'scaled' if getattr(self, 'last_split_used_scaled', False) else 'raw',
                'test_size': getattr(self, 'last_test_size', 0.3),
                'random_state': getattr(self, 'last_split_random_state', 42),
                'stratify': getattr(self, 'last_split_stratify', None),
            }
        joblib.dump(enriched, filename)
        base, _ext = _os_for_log.path.splitext(filename)
        txt_path = base + '_log.txt'
        log_text = _kuquickml_build_model_log_text_final(self, model_name, enriched, filename)
        with open(txt_path, 'w', encoding='utf-8-sig') as f:
            f.write(log_text)
        QMessageBox.information(self, 'Model Saved', f"Model '{model_name}' saved successfully.\nLog file saved: {txt_path}")
    except Exception as e:
        QMessageBox.warning(self, 'Error', f'Failed to save model:\n{e}')

MyApp.saveModel = _kuquickml_saveModel_with_metadata_and_txt_final

# --- v3.85: manuscript-ready summary includes hyperparameter information ---
def _kuquickml_nondefault_param_items(est):
    try:
        params = est.get_params(deep=False)
    except Exception:
        return []
    try:
        default_est = type(est)()
        defaults = default_est.get_params(deep=False)
    except Exception:
        defaults = {}
    items = []
    for k, v in params.items():
        if defaults.get(k, object()) != v:
            items.append((k, v))
    return items


def _kuquickml_param_summary_text(est, max_items=6):
    items = _kuquickml_nondefault_param_items(est)
    if not items:
        return 'Default hyperparameter settings were used.'
    parts = []
    for k, v in items[:max_items]:
        parts.append(f"{k}={v}")
    if len(items) > max_items:
        parts.append('etc.')
    return 'Key hyperparameter settings were ' + ', '.join(parts) + '.'



def _kuquickml_hyperparam_sentence(base_est, max_items=6):
    try:
        if base_est is None or not hasattr(base_est, 'get_params'):
            return 'Estimator hyperparameters were not available.'
        params = base_est.get_params(deep=False)
        defaults = _kuquickml_default_params_for_estimator(base_est)
        changed = []
        for k, v in params.items():
            if k in defaults and defaults.get(k) != v:
                changed.append(f"{k}={_kuquickml_format_param_value(v)}")
        if not changed:
            return 'No estimator hyperparameters differed from sklearn defaults.'
        shown = changed[:max_items]
        tail = '' if len(changed) <= max_items else f"; additional non-default settings: {len(changed) - max_items}"
        return 'Key hyperparameter settings were ' + ', '.join(shown) + tail + '.'
    except Exception:
        return 'Estimator hyperparameters could not be summarized.'

def _kuquickml_algorithm_summary_paragraph(base_est, task, scaler, reducer, cv_settings, train_metrics, test_metrics, cv_metrics):
    algo = _kuquickml_estimator_family_name(base_est)
    algo_label = type(base_est).__name__ if base_est is not None else 'model'
    scaler_name = type(scaler).__name__ if scaler is not None else 'no scaler'
    reducer_name = type(reducer).__name__ if reducer is not None else 'no dimensionality reduction'
    split_label = f"{cv_settings.get('n_splits', 5)}-fold cross-validation" if isinstance(cv_settings, dict) and cv_settings else 'cross-validation'
    train_acc = train_metrics.get('accuracy') if isinstance(train_metrics, dict) else None
    test_acc = test_metrics.get('accuracy') if isinstance(test_metrics, dict) else None
    cv_acc = cv_metrics.get('accuracy') if isinstance(cv_metrics, dict) else None
    train_f1 = train_metrics.get('f1') if isinstance(train_metrics, dict) else None
    test_f1 = test_metrics.get('f1') if isinstance(test_metrics, dict) else None
    cv_f1 = cv_metrics.get('f1') if isinstance(cv_metrics, dict) else None
    train_auc = train_metrics.get('roc_auc') if isinstance(train_metrics, dict) else None
    test_auc = test_metrics.get('roc_auc') if isinstance(test_metrics, dict) else None
    cv_auc = cv_metrics.get('roc_auc') if isinstance(cv_metrics, dict) else None
    train_r2 = train_metrics.get('r2') if isinstance(train_metrics, dict) else None
    test_r2 = test_metrics.get('r2') if isinstance(test_metrics, dict) else None
    cv_r2 = cv_metrics.get('r2') if isinstance(cv_metrics, dict) else None
    train_rmse = train_metrics.get('rmse') if isinstance(train_metrics, dict) else None
    test_rmse = test_metrics.get('rmse') if isinstance(test_metrics, dict) else None
    cv_rmse = cv_metrics.get('rmse') if isinstance(cv_metrics, dict) else None

    def fmt(v):
        if isinstance(v, dict):
            mean = v.get('mean')
            std = v.get('std')
            if mean is None:
                return 'not available'
            if std is None:
                return f"{mean:.4f}"
            return f"{mean:.4f} (SD {std:.4f})"
        if v is None:
            return 'not available'
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    hp_text = _kuquickml_param_summary_text(base_est)

    if task == 'classification':
        if algo == 'knn':
            imp = 'Feature relevance was summarized with mutual information, which quantifies how strongly each feature is associated with the class labels in the input data rather than measuring a built-in model coefficient.'
        elif algo == 'mlp':
            imp = "Feature relevance was summarized with the mean absolute input-layer weight, which reflects the average magnitude of each feature's connection to the first hidden layer and should be interpreted as a heuristic importance measure rather than a direct causal effect."
        elif algo == 'svm':
            imp = 'Feature relevance was summarized with mutual information, which measures the statistical association between each feature and the target labels and is used here as an external importance measure for SVM-based models.'
        elif algo == 'rf':
            imp = "Feature relevance was summarized with the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the forest."
        else:
            imp = 'Feature relevance was summarized using the importance metric recorded with the saved model.'
        return (
            f"This saved model is a {algo_label} classification model trained using {scaler_name} preprocessing and {reducer_name}. "
            f"{hp_text} "
            f"{imp} "
            f"Model performance was summarized with training accuracy {fmt(train_acc)}, test accuracy {fmt(test_acc)}, and {split_label} accuracy {fmt(cv_acc)}. "
            f"Additional classification metrics included training F1-score {fmt(train_f1)}, test F1-score {fmt(test_f1)}, and {split_label} F1-score {fmt(cv_f1)}. "
            f"When available, ROC-AUC values were recorded for the training set ({fmt(train_auc)}), test set ({fmt(test_auc)}), and {split_label} procedure ({fmt(cv_auc)}); for multiclass problems, ROC-AUC is computed using a One-vs-Rest strategy with macro averaging."
        )
    else:
        if algo == 'knn':
            imp = 'Feature relevance was summarized with mutual information, which quantifies how strongly each feature is associated with the target values as an external relevance measure.'
        elif algo == 'mlp':
            imp = "Feature relevance was summarized with the mean absolute input-layer weight, which reflects the average magnitude of each feature's connection to the first hidden layer and should be interpreted as a heuristic importance measure."
        elif algo == 'svm':
            imp = 'Feature relevance was summarized with mutual information, which was used as an external importance measure for the regression model.'
        elif algo == 'rf':
            imp = "Feature relevance was summarized with the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the ensemble."
        else:
            imp = 'Feature relevance was summarized using the importance metric recorded with the saved model.'
        return (
            f"This saved model is a {algo_label} regression model trained using {scaler_name} preprocessing and {reducer_name}. "
            f"{hp_text} "
            f"{imp} "
            f"Model performance was summarized with training R² {fmt(train_r2)}, test R² {fmt(test_r2)}, and {split_label} R² {fmt(cv_r2)}. "
            f"Error statistics included training RMSE {fmt(train_rmse)}, test RMSE {fmt(test_rmse)}, and {split_label} RMSE {fmt(cv_rmse)}."
        )

# --- final patch: keep MLP training flow, use abs-weight importance, and ensure manuscript summary includes hyperparameters ---
def _final_format_importance_value(v):
    try:
        v = float(v)
    except Exception:
        return str(v)
    if v == 0:
        return '0.0000'
    if abs(v) < 1e-4:
        return f"{v:.4e}"
    return f"{v:.4f}"


def _final_mlp_abs_weight(model):
    try:
        if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
            w = np.asarray(model.coefs_[0], dtype=float)
            return np.mean(np.abs(w), axis=1)
    except Exception:
        pass
    return np.array([])


def _final_show_mlp_importance(self, feature_importances, alternative_importances=None, alternative_name='input-layer mean abs weight'):
    dialog = QDialog(self)
    dialog.setWindowTitle('Feature Importances')
    dialog.setWindowModality(Qt.NonModal)
    dialog.setAttribute(Qt.WA_DeleteOnClose)
    dialog.resize(900, 760)
    layout = QVBoxLayout(dialog)

    info = QLabel(
        'MLP\n'
        '이 화면의 중요도는 input-layer mean abs weight로 계산됩니다.\n'
        '각 입력 feature가 첫 번째 은닉층으로 연결될 때의 가중치 절댓값 평균을 사용합니다.\n'
        '값이 클수록 모델이 그 feature를 더 크게 활용했을 가능성을 의미합니다.\n'
        '이는 heuristic 해석 지표이며, feature와 target의 직접적인 통계적 관련성보다 모델 내부 가중치의 크기를 기준으로 해석합니다.'
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    if alternative_importances is not None:
        rows = alternative_importances
    else:
        rows = feature_importances

    table = QTableWidget()
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(['Feature', 'Importance'])
    table.setRowCount(len(rows))
    for i, (fname, val) in enumerate(rows):
        table.setItem(i, 0, QTableWidgetItem(str(fname)))
        table.setItem(i, 1, QTableWidgetItem(_final_format_importance_value(val)))
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
    table.setSelectionBehavior(QTableWidget.SelectItems)
    table.setSelectionMode(QTableWidget.ExtendedSelection)
    _kuquickml_enable_copyable_table(table)
    layout.addWidget(table)

    footer = QLabel('셀을 드래그해 선택한 뒤 우클릭 복사 또는 Ctrl+C를 사용할 수 있습니다.')
    layout.addWidget(footer)
    btn = QPushButton('Close')
    btn.clicked.connect(dialog.close)
    layout.addWidget(btn)
    dialog.show()
    self._last_mlp_importance_dialog = dialog


def _final_create_mlp_classification(self):
    if not self.checkDataSplit():
        return

    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()

    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    X_train_used = X_train_numeric.values
    X_test_used = X_test_numeric.values

    hidden_layer_input_text = self.hidden_layer_input.text().strip()
    hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(',')))
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
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
        mlp.fit(X_train_used, y_train)

    if mlp.n_iter_ == mlp.max_iter:
        QMessageBox.warning(self, 'Iteration Warning', 'Maximum iterations reached. Consider increasing max_iter.')

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
    cm_df['Prediction Accuracy (%)'] = precision
    self.showConfusionMatrix(cm_df)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)

    vals = _final_mlp_abs_weight(mlp)
    order = np.argsort(vals)[::-1]
    feature_importances = [(feature_names[idx], float(vals[idx])) for idx in order]
    self.showMLPFeatureImportances([], alternative_importances=feature_importances, alternative_name='input-layer mean abs weight')

    self.models['MLP Classification'] = {
        'model': mlp,
        'scaler': self._get_bundle_scaler(),
        'reducer': None,
        'feature_names': feature_names,
        'label_mapping': self._get_label_mapping()
    }


def _final_create_mlp_regression(self):
    if not self.checkDataSplit():
        return

    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()

    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    X_train_used = X_train_numeric.values
    X_test_used = X_test_numeric.values

    hidden_layer_input_text = self.hidden_layer_input.text().strip()
    hidden_layers = (50, 50) if not hidden_layer_input_text else tuple(map(int, hidden_layer_input_text.split(',')))
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
        warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
        mlp.fit(X_train_used, y_train)

    if mlp.n_iter_ == mlp.max_iter:
        QMessageBox.warning(self, 'Iteration Warning', 'Maximum iterations reached. Consider increasing max_iter.')

    y_pred_train = mlp.predict(X_train_used)
    y_pred_test = mlp.predict(X_test_used)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    self.showMLPResults(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, mse_test, rmse_test)

    vals = _final_mlp_abs_weight(mlp)
    order = np.argsort(vals)[::-1]
    feature_importances = [(feature_names[idx], float(vals[idx])) for idx in order]
    self.showMLPFeatureImportances([], alternative_importances=feature_importances, alternative_name='input-layer mean abs weight')

    self.models['MLP Regression'] = {
        'model': mlp,
        'scaler': self._get_bundle_scaler(),
        'reducer': None,
        'feature_names': feature_names,
        'label_mapping': None
    }


def _final_hyperparam_sentence(base_est, max_items=12):
    try:
        if base_est is None or not hasattr(base_est, 'get_params'):
            return 'Estimator hyperparameters were not available.'
        params = base_est.get_params(deep=False)
        defaults = _kuquickml_default_params_for_estimator(base_est)
        changed = []
        for k, v in params.items():
            if k in defaults and defaults.get(k) != v:
                changed.append(f"{k}={_kuquickml_format_param_value(v)}")
        if changed:
            return 'Hyperparameter settings included ' + ', '.join(changed) + '; remaining hyperparameters used sklearn default values.'
        return 'All estimator hyperparameters used sklearn default values.'
    except Exception:
        return 'Estimator hyperparameters could not be summarized.'


def _final_algorithm_summary_paragraph(base_est, task, scaler, reducer, cv_settings, train_metrics, test_metrics, cv_metrics):
    algo = _kuquickml_estimator_family_name(base_est)
    algo_label = type(base_est).__name__ if base_est is not None else 'model'
    scaler_name = type(scaler).__name__ if scaler is not None else 'no scaler preprocessing'
    reducer_name = type(reducer).__name__ if reducer is not None else 'no dimensionality reduction'
    split_label = f"{cv_settings.get('n_splits', 5)}-fold cross-validation" if isinstance(cv_settings, dict) and cv_settings else 'cross-validation'

    def fmt(v):
        if isinstance(v, dict):
            mean = v.get('mean')
            std = v.get('std')
            if mean is None:
                return 'not available'
            return f"{mean:.4f} (SD {std:.4f})" if std is not None else f"{mean:.4f}"
        if v is None:
            return 'not available'
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    hp_text = _final_hyperparam_sentence(base_est)
    if task == 'classification':
        if algo == 'knn':
            imp = 'Feature relevance was summarized with mutual information, which quantifies the statistical association between each feature and the class labels.'
        elif algo == 'mlp':
            imp = "Feature relevance was summarized with the mean absolute input-layer weight, which reflects the average magnitude of each feature's connection to the first hidden layer and should be interpreted as a heuristic importance measure rather than a direct causal effect."
        elif algo == 'svm':
            imp = 'Feature relevance was summarized with mutual information, which was used as an external importance measure for the SVM classifier.'
        elif algo == 'rf':
            imp = "Feature relevance was summarized with the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the forest."
        else:
            imp = 'Feature relevance was summarized using the importance metric recorded with the saved model.'
        return (
            f"This saved model is a {algo_label} classification model trained using {scaler_name} and {reducer_name}. "
            f"{hp_text} "
            f"{imp} "
            f"Model performance was summarized with training accuracy {fmt(train_metrics.get('accuracy') if isinstance(train_metrics, dict) else None)}, test accuracy {fmt(test_metrics.get('accuracy') if isinstance(test_metrics, dict) else None)}, and {split_label} accuracy {fmt(cv_metrics.get('accuracy') if isinstance(cv_metrics, dict) else None)}. "
            f"Additional classification metrics included training F1-score {fmt(train_metrics.get('f1') if isinstance(train_metrics, dict) else None)}, test F1-score {fmt(test_metrics.get('f1') if isinstance(test_metrics, dict) else None)}, and {split_label} F1-score {fmt(cv_metrics.get('f1') if isinstance(cv_metrics, dict) else None)}. "
            f"When available, ROC-AUC values were recorded for the training set ({fmt(train_metrics.get('roc_auc') if isinstance(train_metrics, dict) else None)}), test set ({fmt(test_metrics.get('roc_auc') if isinstance(test_metrics, dict) else None)}), and {split_label} procedure ({fmt(cv_metrics.get('roc_auc') if isinstance(cv_metrics, dict) else None)}); for multiclass problems, ROC-AUC is computed using a One-vs-Rest strategy with macro averaging."
        )
    else:
        if algo == 'knn':
            imp = 'Feature relevance was summarized with mutual information, which quantifies how strongly each feature is associated with the target values as an external relevance measure.'
        elif algo == 'mlp':
            imp = "Feature relevance was summarized with the mean absolute input-layer weight, which reflects the average magnitude of each feature's connection to the first hidden layer and should be interpreted as a heuristic importance measure."
        elif algo == 'svm':
            imp = 'Feature relevance was summarized with mutual information, which was used as an external importance measure for the regression model.'
        elif algo == 'rf':
            imp = "Feature relevance was summarized with the model's built-in feature importance values, which reflect the average contribution of each feature to impurity reduction across the ensemble."
        else:
            imp = 'Feature relevance was summarized using the importance metric recorded with the saved model.'
        return (
            f"This saved model is a {algo_label} regression model trained using {scaler_name} and {reducer_name}. "
            f"{hp_text} "
            f"{imp} "
            f"Model performance was summarized with training R² {fmt(train_metrics.get('r2') if isinstance(train_metrics, dict) else None)}, test R² {fmt(test_metrics.get('r2') if isinstance(test_metrics, dict) else None)}, and {split_label} R² {fmt(cv_metrics.get('r2') if isinstance(cv_metrics, dict) else None)}. "
            f"Error statistics included training RMSE {fmt(train_metrics.get('rmse') if isinstance(train_metrics, dict) else None)}, test RMSE {fmt(test_metrics.get('rmse') if isinstance(test_metrics, dict) else None)}, and {split_label} RMSE {fmt(cv_metrics.get('rmse') if isinstance(cv_metrics, dict) else None)}."
        )

MyApp.showMLPFeatureImportances = _final_show_mlp_importance
MyApp.createMLPClassificationModel = _final_create_mlp_classification
MyApp.createMLPRegressionModel = _final_create_mlp_regression
_kuquickml_algorithm_summary_paragraph = _final_algorithm_summary_paragraph
MyApp.saveModel = _kuquickml_saveModel_with_metadata_and_txt_final



def _v393_bin_regression_target_for_projection(y, max_bins=5):
    y_s = pd.Series(np.asarray(y).ravel())
    uniq = y_s.nunique(dropna=True)
    n_bins = max(2, min(max_bins, uniq))
    try:
        return pd.qcut(y_s, q=n_bins, labels=False, duplicates='drop').astype(int).to_numpy()
    except Exception:
        try:
            return pd.cut(y_s, bins=n_bins, labels=False, duplicates='drop').astype(int).to_numpy()
        except Exception:
            # fallback single class -> no supervised reducer possible
            return None


def _v393_get_svr_plot_projection(self, X_train_numeric, X_test_numeric, y_train):
    Xtr = np.asarray(X_train_numeric.values if hasattr(X_train_numeric, 'values') else X_train_numeric)
    Xte = np.asarray(X_test_numeric.values if hasattr(X_test_numeric, 'values') else X_test_numeric)
    if getattr(self, 'pcaCheckBox', None) is not None and self.pcaCheckBox.isChecked():
        reducer_plot = PCA(n_components=2, random_state=42 if 'random_state' in PCA().get_params() else None)
        reducer_plot.fit(Xtr)
        return reducer_plot.transform(Xtr), reducer_plot.transform(Xte), reducer_plot
    if getattr(self, 'ldaCheckBox', None) is not None and self.ldaCheckBox.isChecked():
        y_bins = _v393_bin_regression_target_for_projection(y_train)
        if y_bins is not None:
            n_classes = len(np.unique(y_bins))
            max_components = min(Xtr.shape[1], n_classes - 1)
            if max_components >= 2:
                reducer_plot = LDA(n_components=2)
                reducer_plot.fit(Xtr, y_bins)
                return reducer_plot.transform(Xtr), reducer_plot.transform(Xte), reducer_plot
            elif max_components == 1:
                # LDA cannot produce 2D when classes-1 == 1; fall back to PCA for plotting only
                reducer_plot = PCA(n_components=2)
                reducer_plot.fit(Xtr)
                return reducer_plot.transform(Xtr), reducer_plot.transform(Xte), reducer_plot
    if getattr(self, 'ncaCheckBox', None) is not None and self.ncaCheckBox.isChecked():
        y_bins = _v393_bin_regression_target_for_projection(y_train)
        if y_bins is not None and len(np.unique(y_bins)) >= 2:
            reducer_plot = NCA(n_components=2, max_iter=100, tol=1e-5, random_state=42)
            reducer_plot.fit(Xtr, y_bins)
            return reducer_plot.transform(Xtr), reducer_plot.transform(Xte), reducer_plot
    return None, None, None


def _v393_plotScatterWithRegressionSurface(self, X_train_reduced, y_train, X_test_reduced, y_test, model, title):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor('white')

    X_all = np.vstack([np.asarray(X_train_reduced), np.asarray(X_test_reduced)])
    x_min, x_max = X_all[:, 0].min() - 1, X_all[:, 0].max() + 1
    y_min, y_max = X_all[:, 1].min() - 1, X_all[:, 1].max() + 1

    grid_res = 400
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        levels = np.linspace(zmin, zmax, 20)
    else:
        levels = 20
    ctr = plt.contour(xx, yy, Z, levels=levels, colors='gray', linewidths=0.7, linestyles='solid', alpha=0.7)
    try:
        plt.clabel(ctr, inline=True, fontsize=8, fmt='%.2f')
    except Exception:
        pass

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
            bins = pd.qcut(y_all, q=n_bins, duplicates='drop')
        except Exception:
            bins = pd.cut(y_all, bins=n_bins, duplicates='drop')
        bin_str = bins.astype(str)
        y_train_cat = bin_str[:len(y_train_arr)]
        y_test_cat = bin_str[len(y_train_arr):]
        categories = list(pd.unique(bin_str))

    label_colors = {cat: '#{:02x}{:02x}{:02x}'.format(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for cat in categories}

    for cat in categories:
        train_idx = np.where(y_train_cat == cat)[0]
        test_idx = np.where(y_test_cat == cat)[0]
        color = label_colors[cat]
        if len(train_idx) > 0:
            plt.scatter(X_train_reduced[train_idx,0], X_train_reduced[train_idx,1], c=color, s=30, marker='o', alpha=0.55, label=f'Train label {cat}')
        if len(test_idx) > 0:
            plt.scatter(X_test_reduced[test_idx,0], X_test_reduced[test_idx,1], c=color, s=70, marker='x', alpha=0.90, label=f'Test label {cat}')

    legend = plt.legend()
    legend.set_draggable(True)
    plt.title(title, fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
    plt.xlabel('Component 1', fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
    plt.ylabel('Component 2', fontsize=self.fontSizeInput.value(), fontname=self.fontTypeComboBox.currentText())
    plt.show()


def _v393_createSVMRegressionModel(self):
    if not self.checkDataSplit():
        return
    try:
        X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
        X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
        y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
        y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()

        X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
        X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
        feature_names = list(X_train_numeric.columns)

        reducer = None
        if self.pcaCheckBox.isChecked():
            reducer = PCA(n_components=2)
            reducer.fit(X_train_numeric.values)
            X_train_used = reducer.transform(X_train_numeric.values)
            X_test_used = reducer.transform(X_test_numeric.values)
        else:
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

        if kernel == 'linear' and reducer is None and hasattr(svr_model, 'coef_'):
            self.showImportantCoefficients(X_train_numeric, svr_model)
        else:
            self.showSVMImportanceUnavailable(
                kernel=kernel,
                reducer=reducer,
                X_test=X_test_numeric,
                y_test=y_test,
                model=svr_model,
                feature_names=feature_names,
                title_prefix='SVR (Regression)',
                task='regression'
            )

        plot_train_2d, plot_test_2d, plot_reducer = _v393_get_svr_plot_projection(self, X_train_numeric, X_test_numeric, y_train)
        if plot_train_2d is not None and plot_train_2d.shape[1] == 2:
            plot_model = svr_model
            # if actual model was trained on unreduced features, fit a lightweight 2D visual surrogate for plotting only
            if X_train_used.shape[1] != 2:
                plot_model = SVR(kernel=kernel, C=c_value, epsilon=epsilon)
                plot_model.fit(plot_train_2d, y_train)
            self.plotScatterWithRegressionSurface(
                plot_train_2d, y_train, plot_test_2d, y_test, plot_model,
                f"SVR Regression Surface (kernel={kernel})"
            )

        self.plotObservedVsPredicted(
            y_train, y_pred_train, y_test, y_pred_test,
            f"SVR Observed vs Predicted (kernel={kernel})\nTest R2={r2_test:.3f}, RMSE={rmse_test:.3f}"
        )

        self.models['SV Regression'] = {
            'model': svr_model,
            'scaler': self._get_bundle_scaler(),
            'reducer': reducer,
            'feature_names': feature_names,
            'label_mapping': None
        }
        if reducer:
            self.model_reducers['SV Regression'] = reducer
    except Exception as e:
        QMessageBox.warning(self, 'SVR Error', f'Failed to create SVR model:\n{e}')


# v3.93 regression plot patches
MyApp.plotScatterWithRegressionSurface = _v393_plotScatterWithRegressionSurface
MyApp.createSVMRegressionModel = _v393_createSVMRegressionModel

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())


# ===== v5 base cleanup patch: preserve v5 loader, allow None for KNN/SVM, remove permutation-based active handlers =====
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

def _v5clean_to_numeric_df(X):
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)
    for col in X_df.columns:
        if X_df[col].dtype == object:
            X_df[col] = pd.to_numeric(X_df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
    return X_df.fillna(0)


def _v5clean_mutual_info(X, y, task='classification'):
    X_df = _v5clean_to_numeric_df(X)
    y_arr = np.asarray(y)
    if task == 'classification':
        vals = mutual_info_classif(X_df, y_arr, random_state=42)
    else:
        vals = mutual_info_regression(X_df, y_arr, random_state=42)
    return np.asarray(vals, dtype=float)


def _v5clean_show_importance_dialog(self, title, feature_names, values, metric_name, intro_html):
    vals = np.asarray(values, dtype=float)
    names = list(feature_names) if feature_names else [f'Feature {i+1}' for i in range(len(vals))]
    n = min(len(vals), len(names))
    order = np.argsort(np.nan_to_num(vals[:n], nan=-np.inf))[::-1]

    dialog = QDialog(self)
    dialog.setWindowTitle(title)
    dialog.resize(800, 600)
    layout = QVBoxLayout(dialog)

    info = QLabel(intro_html)
    info.setWordWrap(True)
    layout.addWidget(info)

    table = QTableWidget(dialog)
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(['Feature', metric_name])
    table.setRowCount(n)
    for row, idx in enumerate(order):
        table.setItem(row, 0, QTableWidgetItem(str(names[idx])))
        table.setItem(row, 1, QTableWidgetItem(f'{float(vals[idx]):.6f}'))
    table.resizeColumnsToContents()
    try:
        _kuquickml_enable_copyable_table(table)
    except Exception:
        pass
    layout.addWidget(table)

    dialog.setLayout(layout)
    dialog.setWindowModality(Qt.NonModal)
    dialog.show()


def _v5clean_show_knn_importance(self, reducer, X_eval, y_eval, model, feature_names, title_prefix='KNN', task='classification'):
    try:
        vals = _v5clean_mutual_info(X_eval, y_eval, task=task)
        intro = (
            f'<b>{title_prefix}</b><br>'
            '이 화면의 중요도는 <b>mutual information</b>으로 계산됩니다.<br>'
            '각 feature가 target과 얼마나 관련되어 있는지를 나타내는 값입니다.<br>'
            '값이 클수록 해당 feature가 더 많은 정보를 담고 있다고 해석할 수 있습니다.<br>'
            '고정된 절대 기준값보다는 같은 데이터 내 다른 feature와의 상대적 크기와 순위로 해석하는 것이 적절합니다.'
        )
        _v5clean_show_importance_dialog(self, 'Feature Importances', feature_names, vals, 'mutual information', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Feature Importance Error', f'Failed to compute feature importance:\n{e}')


def _v5clean_show_svm_importance(self, kernel, reducer, X_test, y_test, model, feature_names, title_prefix='SVM', task='classification'):
    try:
        vals = _v5clean_mutual_info(X_test, y_test, task=task)
        intro = (
            f'<b>{title_prefix}</b><br>'
            '이 화면의 중요도는 <b>mutual information</b>으로 계산됩니다.<br>'
            '각 feature가 target과 얼마나 관련되어 있는지를 나타내는 값입니다.<br>'
            '값이 클수록 해당 feature가 더 많은 정보를 담고 있다고 해석할 수 있습니다.<br>'
            '고정된 절대 기준값보다는 같은 데이터 내 다른 feature와의 상대적 크기와 순위로 해석하는 것이 적절합니다.'
        )
        _v5clean_show_importance_dialog(self, 'Feature Importances', feature_names, vals, 'mutual information', intro)
    except Exception as e:
        QMessageBox.warning(self, 'Feature Importance Error', f'Failed to compute feature importance:\n{e}')


def _v5clean_apply_current_reducer(self):
    try:
        if not hasattr(self, 'scaled_unknown_data'):
            QMessageBox.warning(self, 'Data Error', 'Please scale the unknown data first.')
            return
        selected_models = [name for name, checkbox in self.modelCheckBoxes.items() if checkbox.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, 'Model Selection Error', 'Please select at least one model.')
            return
        model_name = selected_models[0]
        reducer = self.model_reducers.get(model_name, None)
        unknown_df = pd.DataFrame(self.scaled_unknown_data, columns=self.unknown_data.columns)
        if hasattr(self, 'feature_names'):
            compare_expected = _kuquickml_strip_non_feature_columns(list(self.feature_names))
            compare_loaded = _kuquickml_strip_non_feature_columns(list(unknown_df.columns))
            missing = set(compare_expected) - set(compare_loaded)
            if missing:
                QMessageBox.warning(self, 'Feature Mismatch', f"The following features are missing in unknown data:\n{', '.join(missing)}")
                return
            unknown_df = unknown_df.loc[:, compare_expected]
        if reducer is None:
            reduced = unknown_df.values
            print(f'[Reducer Applied] No dimensionality reduction for {model_name}')
        else:
            reduced = reducer.transform(unknown_df)
            print(f'[Reducer Applied] Using reducer from {model_name}')
        self.reduced_unknown_data = reduced
        self.showUnknownData(pd.DataFrame(reduced))
    except Exception as e:
        QMessageBox.warning(self, 'Reducer Error', f'Failed to apply reducer:\n{e}')


def _v5clean_createClassificationModel(self):
    if not self.checkDataSplit():
        return
    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()

    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    n_neighbors = self.n_neighbors_input.value()

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    selected_method = self.getSelectedDimReductionMethod()
    if not selected_method:
        QMessageBox.warning(self, 'Selection Error', 'Please select a dimensionality reduction method.')
        return

    method_name, reducer = selected_method
    if reducer is not None:
        reducer.fit(X_train_numeric.values, y_train)
        X_train_embedded = reducer.transform(X_train_numeric.values)
        X_test_embedded = reducer.transform(X_test_numeric.values)
    else:
        X_train_embedded = X_train_numeric.values
        X_test_embedded = X_test_numeric.values

    knn.fit(X_train_embedded, y_train)
    accuracy = knn.score(X_test_embedded, y_test)
    if hasattr(X_train_embedded, 'shape') and len(X_train_embedded.shape) == 2 and X_train_embedded.shape[1] == 2:
        self.plotResults(method_name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors, score_value=accuracy, score_label='Test accuracy')

    y_pred_test = knn.predict(X_test_embedded)
    cm = confusion_matrix(y_test, y_pred_test)
    unique_labels = np.unique(np.concatenate((y_test, y_pred_test)))
    true = [f'true_{label}' for label in unique_labels]
    pred = [f'pred_{label}' for label in unique_labels]
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.round(np.diag(cm) / np.sum(cm, axis=0) * 100, 3)
        precision = np.nan_to_num(precision)
    cm_df = pd.DataFrame(cm, index=true, columns=pred)
    cm_df['Prediction Accuracy (%)'] = precision
    self.showConfusionMatrix(cm_df)

    train_accuracy = knn.score(X_train_embedded, y_train)
    result_text = (
        f'Train accuracy: {train_accuracy:.4f}\n'
        f'Test accuracy: {accuracy:.4f}\n'
        f'Neighbors: {n_neighbors}\n'
        f'Dimensionality reduction: {method_name}'
    )
    self.resultsLabel.setText(result_text)
    self.saveModelButton.setVisible(True)
    self.useCurrentModelButton.setVisible(True)
    self.current_model = knn
    self.current_model_type = 'KNN Classification'
    self.showKNNFeatureImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix='KNN (Classification)', task='classification')
    self.plotObservedVsPredicted(y_train, knn.predict(X_train_embedded), y_test, y_pred_test, 'KNN Observed vs Predicted')
    self.models['KNN Classification'] = {'model': knn, 'scaler': self._get_bundle_scaler(), 'reducer': reducer, 'feature_names': feature_names, 'label_mapping': self._get_label_mapping()}
    if reducer is not None:
        self.model_reducers['KNN Classification'] = reducer
    elif 'KNN Classification' in self.model_reducers:
        del self.model_reducers['KNN Classification']


def _v5clean_createRegressionModel(self):
    if not self.checkDataSplit():
        return
    X_train = pd.read_csv(resource_path('Temp/X_train.csv'))
    X_test = pd.read_csv(resource_path('Temp/X_test.csv'))
    y_train = pd.read_csv(resource_path('Temp/y_train.csv')).values.ravel()
    y_test = pd.read_csv(resource_path('Temp/y_test.csv')).values.ravel()

    X_train_numeric = self._drop_sample_and_numeric(X_train).fillna(0)
    X_test_numeric = self._drop_sample_and_numeric(X_test).fillna(0)
    feature_names = list(X_train_numeric.columns)
    n_neighbors = self.n_neighbors_input.value()

    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    selected_method = self.getSelectedDimReductionMethod()
    if not selected_method:
        QMessageBox.warning(self, 'Selection Error', 'Please select a dimensionality reduction method.')
        return

    method_name, reducer = selected_method
    if reducer is not None:
        reducer.fit(X_train_numeric.values, y_train)
        X_train_embedded = reducer.transform(X_train_numeric.values)
        X_test_embedded = reducer.transform(X_test_numeric.values)
    else:
        X_train_embedded = X_train_numeric.values
        X_test_embedded = X_test_numeric.values

    knn.fit(X_train_embedded, y_train)
    y_pred_train = knn.predict(X_train_embedded)
    y_pred_test = knn.predict(X_test_embedded)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    if hasattr(X_train_embedded, 'shape') and len(X_train_embedded.shape) == 2 and X_train_embedded.shape[1] == 2:
        self.plotRegressionResults(method_name, X_train_embedded, y_train, X_test_embedded, y_test, n_neighbors, score_value=r2, score_label='Test R²')
    result_text = (
        f'Test MSE: {mse:.4f}\nTest RMSE: {rmse:.4f}\nTest MAE: {mae:.4f}\nTest R²: {r2:.4f}\nNeighbors: {n_neighbors}\nDimensionality reduction: {method_name}'
    )
    self.resultsLabel.setText(result_text)
    self.saveModelButton.setVisible(True)
    self.useCurrentModelButton.setVisible(True)
    self.current_model = knn
    self.current_model_type = 'KNN Regression'
    self.showKNNFeatureImportance(reducer, X_test_numeric, y_test, knn, feature_names, title_prefix='KNN (Regression)', task='regression')
    self.plotObservedVsPredicted(y_train, y_pred_train, y_test, y_pred_test, 'KNN Regression Observed vs Predicted')
    self.models['KNN Regression'] = {'model': knn, 'scaler': self._get_bundle_scaler(), 'reducer': reducer, 'feature_names': feature_names, 'label_mapping': None}
    if reducer is not None:
        self.model_reducers['KNN Regression'] = reducer
    elif 'KNN Regression' in self.model_reducers:
        del self.model_reducers['KNN Regression']


MyApp.showKNNFeatureImportance = _v5clean_show_knn_importance
MyApp.showKNNPermutationImportance = _v5clean_show_knn_importance
MyApp.showSVMImportanceUnavailable = _v5clean_show_svm_importance
MyApp.applyCurrentReducer = _v5clean_apply_current_reducer
MyApp.createClassificationModel = _v5clean_createClassificationModel
MyApp.createRegressionModel = _v5clean_createRegressionModel

def _v5clean_no_permutation(*args, **kwargs):
    raise RuntimeError('Permutation importance is disabled in this version.')

_kuquickml_permutation = _v5clean_no_permutation
# ===== end v5 base cleanup patch =====

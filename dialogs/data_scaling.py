import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QMessageBox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
def resource_path(relative_path):
    """ PyInstaller로 패키징할 때 리소스 파일 경로를 반환하는 함수 """
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller로 패키징된 환경에서 실행될 때
        base_path = sys._MEIPASS
    else:
        # 개발 환경에서 실행될 때
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
class DataScaler:
    def __init__(self, csv_viewer, parent):
        self.csv_viewer = csv_viewer
        self.parent = parent
        self.current_scaler = None
    def apply_scaling(self, scaler):
        if self.csv_viewer.X is None:
            QMessageBox.warning(self.parent, "Error", "Load data before scaling.")
            return

        self.current_scaler = scaler  # 선택된 스케일러를 저장
        scaled_X = scaler.fit_transform(self.csv_viewer.X)
        self.save_data(self.csv_viewer.original_data, scaled_X, self.csv_viewer.y)
        return self.current_scaler

    def save_data(self, original_X, scaled_X, y):
        output_dir = resource_path('Temp')  # Use resource_path for bundled executable

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        original_X_path = os.path.join(output_dir, 'original_X.csv')
        scaled_X_path = os.path.join(output_dir, 'scaled_X.csv')
        y_path = os.path.join(output_dir, 'scaled_y.csv')

        pd.DataFrame(original_X).to_csv(original_X_path, index=False)

        # 샘플 이름을 scaled_X에 추가
        scaled_X_df = pd.DataFrame(scaled_X, columns=original_X.columns[1:])  # 첫 번째 열은 샘플 이름
        scaled_X_df.insert(0, original_X.columns[0], original_X.iloc[:, 0])  # 첫 번째 열을 샘플 이름으로 추가
        scaled_X_df.to_csv(scaled_X_path, index=False)

        pd.DataFrame(y, columns=['Label']).to_csv(y_path, index=False)

        QMessageBox.information(self.parent, "Information", "Data has been scaled and saved.")
        self.parent.show_scaled_data(scaled_X_df, y, [f'Scaled {col}' for col in original_X.columns[1:]])

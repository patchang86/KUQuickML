from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt

class LabelMappingDialog(QDialog):
    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Label to Number Mapping")
        self.layout = QVBoxLayout(self)
        headerLabel = QLabel("Assign a numeric value to each group/class:")
        headerLabel.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(headerLabel)
        self.labelInputs = {}

        for label in labels:
            rowLayout = QHBoxLayout()
            labelWidget = QLabel(str(label))
            inputWidget = QLineEdit()
            inputWidget.setValidator(QDoubleValidator())
            rowLayout.addWidget(labelWidget)
            rowLayout.addWidget(inputWidget)
            self.layout.addLayout(rowLayout)
            self.labelInputs[label] = inputWidget

        self.acceptButton = QPushButton("OK")
        self.acceptButton.clicked.connect(self.accept)
        self.layout.addWidget(self.acceptButton)

    def getLabelMappings(self):
        labelMappings = {}
        for label, inputWidget in self.labelInputs.items():
            text = inputWidget.text()
            if not text:
                QMessageBox.warning(self, "Input Required", f"Please enter a numeric value for {label}.")
                return None
            try:
                number = float(text)
                labelMappings[label] = number
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Value for {label} is not a valid number. Please enter a valid number.")
                return None
        return labelMappings

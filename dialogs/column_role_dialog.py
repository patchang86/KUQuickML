from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QComboBox, QPushButton, QScrollArea, QWidget

class ColumnRoleDialog(QDialog):
    def __init__(self, headers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Column Roles")
        self.resize(400, 300)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scrollContent = QWidget(scroll)
        scroll.setWidget(scrollContent)
        layout = QVBoxLayout(scrollContent)
        self.selections = {}

        for index, header in enumerate(headers):
            rowLayout = QHBoxLayout()
            label = QLabel(header)
            comboBox = QComboBox()
            comboBox.addItems(["Sample", "Label", "Feature"])

            if index == 0:
                comboBox.setCurrentIndex(comboBox.findText("Sample"))
            elif index == 1:
                comboBox.setCurrentIndex(comboBox.findText("Label"))
            else:
                comboBox.setCurrentIndex(comboBox.findText("Feature"))

            rowLayout.addWidget(label)
            rowLayout.addWidget(comboBox)
            layout.addLayout(rowLayout)
            self.selections[header] = comboBox

        mainLayout = QVBoxLayout(self)
        mainLayout.addWidget(scroll)

        self.acceptButton = QPushButton("OK")
        self.acceptButton.clicked.connect(self.accept)
        mainLayout.addWidget(self.acceptButton)

    def getSelections(self):
        return {header: comboBox.currentText() for header, comboBox in self.selections.items()}

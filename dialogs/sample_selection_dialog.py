from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QDialogButtonBox

class SampleSelectionDialog(QDialog):
    def __init__(self, samples, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Samples")
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.MultiSelection)
        for sample in samples:
            self.list_widget.addItem(str(sample))
        layout.addWidget(self.list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_samples(self):
        return [int(item.text()) for item in self.list_widget.selectedItems()]

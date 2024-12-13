import sys
import os

# Füge den Pfad zu deinem tf_utils-Ordner hinzu
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/tf_utils'))

# Importiere das callbacks-Modul aus tf_utils
import tf_utils.callbacks

import pickle
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from typing import Optional

class ModelManager:
    def __init__(self, model_dir: str = None):
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent / "models"
        else:
            self.model_dir = Path(model_dir)

        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / "malnutrition_model.pkl"
        self.metadata_path = self.model_dir / "model_metadata.pkl"

    def save_model(self, model: DecisionTreeClassifier, metadata: dict = None) -> None:
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {self.model_path}")

        if metadata:
            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            print(f"Metadata saved to {self.metadata_path}")

    def load_model(self) -> Optional[DecisionTreeClassifier]:
        if not self.model_path.exists():
            print(f"Model not found at {self.model_path}")
            return None

        try:
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Model loaded from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_metadata(self) -> Optional[dict]:
        if not self.metadata_path.exists():
            return None

        try:
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
            return metadata
        except Exception as e:
            print(f"✗ Error loading metadata: {e}")
            return None

    def model_exists(self) -> bool:
        return self.model_path.exists()

    def get_model_info(self) -> dict:
        if not self.model_exists():
            return {"exists": False, "message": "No model found"}

        model_size = self.model_path.stat().st_size / 1024  # KB
        metadata = self.load_metadata()

        return {
            "exists": True,
            "path": str(self.model_path),
            "size_kb": round(model_size, 2),
            "metadata": metadata,
        }

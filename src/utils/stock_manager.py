import json
import os
import logging

logger = logging.getLogger(__name__)

class StockManager:
    def __init__(self, file_path="data/inventory.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.inventory = self._load()
        
        # Categories allowed for stock tracking and visualization
        self.allowed_categories = {
            "cell phone", "laptop", "mouse", "keyboard", "remote", "tv",
            "bottle", "cup", "chair", "book", "backpack", "handbag",
            "scissors", "clock", "vase"
        }

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading inventory: {e}")
        return {}

    def _save(self):
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.inventory, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving inventory: {e}")

    def add_item(self, label):
        if label not in self.allowed_categories:
            return False
        
        self.inventory[label] = self.inventory.get(label, 0) + 1
        self._save()
        logger.info(f"STOCK IN: {label} (Total: {self.inventory[label]})")
        return True

    def remove_item(self, label):
        if label not in self.allowed_categories:
            return False
        
        current = self.inventory.get(label, 0)
        if current > 0:
            self.inventory[label] = current - 1
            self._save()
            logger.info(f"STOCK OUT: {label} (Total: {self.inventory[label]})")
            return True
        return False

    def get_stock(self):
        return self.inventory

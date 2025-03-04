import torch
from torch.utils.data import DataLoader

def custom_collate(batch):
    """
    Custom collate function to handle variable-sized segment lists in a batch.
    
    Args:
        batch (list): List of dataset items, each containing multiple segments.

    Returns:
        dict: Collated batch with properly stacked tensors.
    """
    batch_dict = {"audio": [], "tg_audio": [], "file_id": []}

    for sample in batch:
        for segment in sample:  # Разворачиваем список сегментов
            batch_dict["audio"].append(segment["audio"])
            batch_dict["tg_audio"].append(segment["tg_audio"])
            batch_dict["file_id"].append(segment["file_id"])

    # Объединяем тензоры по batch dimension
    batch_dict["audio"] = torch.stack(batch_dict["audio"])
    batch_dict["tg_audio"] = torch.stack(batch_dict["tg_audio"])

    return batch_dict

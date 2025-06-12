import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor

class WildtrackDataset(Dataset):
    def __init__(self, 
                 annotations_dir: str,
                 images_base_dir: str,
                 cameras: dict = {0: 'C1', 5: 'C6'},
                 frame_step: int = 5,
                 max_detections: int = 50):
        
        self.cameras = cameras
        self.frame_step = frame_step
        self.max_detections = max_detections

        # Parsowanie nazw plików JSON: np. 00000000.json, 00000005.json, ...
        self.frame_files = []
        for f in os.listdir(annotations_dir):
            if f.endswith('.json'):
                try:
                    frame_number = int(os.path.splitext(f)[0])
                    self.frame_files.append((f, frame_number))
                except Exception as e:
                    print(f"Ostrzeżenie: Pominięto plik {f} - nieprawidłowy format nazwy")
        
        # Sortuj po numerze klatki
        self.frame_files.sort(key=lambda x: x[1])
        self.frame_files = [f[0] for f in self.frame_files]
        print(self.frame_files)
        # Wczytaj adnotacje
        self.annotations = []
        for frame_file in self.frame_files:
            with open(os.path.join(annotations_dir, frame_file)) as f:
                frame_data = json.load(f)
                self.annotations.append(self._process_frame(frame_data))
        
        # Przygotuj mapowanie ścieżek do obrazów
        self.image_paths = {}
        for view_num, cam_id in self.cameras.items():
            cam_dir = os.path.join(images_base_dir, cam_id)
            self.image_paths[view_num] = {
                int(os.path.splitext(f)[0]): os.path.join(cam_dir, f)
                for f in os.listdir(cam_dir) 
                if f.endswith('.png')
            }

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    def _process_frame(self, frame_data):
        """Przetwarza surowe dane z pliku JSON na format per kamera"""
        frame_anns = {view_num: [] for view_num in self.cameras.keys()}
        
        for person in frame_data:
            for view in person['views']:
                view_num = view['viewNum']
                if view_num in self.cameras and view['xmin'] != -1:
                    bbox = [
                        view['xmin'], 
                        view['ymin'], 
                        view['xmax'] - view['xmin'], 
                        view['ymax'] - view['ymin']
                    ]
                    frame_anns[view_num].append({
                        'bbox': bbox,
                        'track_id': person['personID']
                    })
        
        return frame_anns

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        # Numer klatki wg nazwy pliku (np. 00000000.json → 0, 00000005.json → 5 itd.)
        frame_number = int(os.path.splitext(self.frame_files[idx])[0])
        
        # Wczytaj obrazy dla wszystkich kamer
        images = {}
        for view_num, cam_id in self.cameras.items():
            img_path = self.image_paths[view_num].get(frame_number)
            if not img_path:
                raise FileNotFoundError(f"Brak obrazu {cam_id} dla klatki {frame_number}")
            
            images[view_num] = Image.open(img_path).convert('RGB')
        
        # Pobierz adnotacje dla klatki
        anns = self.annotations[idx]
        
        # Przygotuj dane wyjściowe
        formatted_anns = {}
        for view_num in self.cameras.keys():
            cam_anns = anns.get(view_num, [])
            
            boxes = torch.zeros((self.max_detections, 4))
            track_ids = torch.zeros(self.max_detections, dtype=torch.long)
            
            for i, ann in enumerate(cam_anns[:self.max_detections]):
                boxes[i] = torch.tensor(ann['bbox'])
                track_ids[i] = ann['track_id']
            
            formatted_anns[view_num] = {
                'boxes': boxes,
                'track_ids': track_ids
            }

        return images, formatted_anns


def create_collate_fn(processor):

    def custom_collate_fn(batch):
        images_batch, anns_batch = zip(*batch)
        
        # Przetwórz obrazy przez procesor DETR
        processed_images = {}
        for view_num in batch[0][0].keys():
            view_images = [img[view_num] for img in images_batch]
            processed = processor(images=view_images, return_tensors="pt")
            processed_images[view_num] = {
                'pixel_values': processed['pixel_values'],
                'pixel_mask': processed['pixel_mask']
            }
        
        # Przetwórz adnotacje
        formatted_anns = {
            'detection_labels': {},
            'assignments': {}
        }
        
        for view_num in anns_batch[0].keys():
            all_boxes = torch.stack([anns[view_num]['boxes'] for anns in anns_batch])
            all_track_ids = torch.stack([anns[view_num]['track_ids'] for anns in anns_batch])
            
            formatted_anns['detection_labels'][view_num] = all_boxes
            formatted_anns['assignments'][view_num] = all_track_ids
        
        return processed_images, formatted_anns
    return custom_collate_fn
# Przykład użycia
if __name__ == '__main__':
    path = os.getcwd()
    dataset = WildtrackDataset(
        annotations_dir=os.path.join(path, 'Dataset/annotations_positions'),
        images_base_dir=os.path.join(path, 'Dataset/image_subsets'),
        cameras={0: 'C1', 5: 'C6'},
        frame_step=5
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=create_collate_fn(dataset.processor),
        shuffle=False,
        num_workers=0
    )
    
    print(dataset[0])
    for images, anns in dataloader:
        print("Obrazy:", images)
        print("Adnotacje:", anns)
        
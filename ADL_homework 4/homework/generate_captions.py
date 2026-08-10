from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    karts = extract_kart_objects(info_path, view_index)
    track = extract_track_info(info_path)
    
    captions = []
    
    captions.append(f"There are {len(karts)} karts in the scenario.")
    captions.append(f"The track is {track}.")
    ego = next((kart for kart in karts if kart["is_center_kart"]), None)

    if ego is not None:
        ego_x, ego_y = ego["center"]
        
        captions.append(f"{ego['kart_name']} is the ego car.")
        for kart in karts:
            if kart["is_center_kart"]:
                continue
            kx, ky = kart["center"]
            name = kart["kart_name"]
            captions.append(f"{name} is {'left' if kx < ego_x else 'right'} of the ego car.")
            captions.append(f"{name} is {'in front of' if ky < ego_y else 'behind'} the ego car.")
    return captions


def generate_all(split: str = "train", img_width: int = 150, img_height: int = 100):
    """
    Generate captions for all views in the dataset.
    """
    import json
    from .data import DATA_DIR
    split_dir = DATA_DIR / split
    info_files = sorted(split_dir.glob("*_info.json"))
    all_captions = []
    for info_file in info_files:
        with open(info_file) as f:
            info = json.load(f)
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(len(info["detections"])):
            image_file = f"{split}/{base_name}_{view_index:02d}_im.jpg"
            for caption in generate_caption(str(info_file), view_index, img_width, img_height):
                all_captions.append({
                    "image_file": image_file,
                    "caption": caption,
                })
    output_file = split_dir / f"{split}_captions.json"
    with open(output_file, "w") as f:
        json.dump(all_captions, f)
    print(f"Saved {len(all_captions)} captions to {output_file}")

def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_all})


if __name__ == "__main__":
    main()

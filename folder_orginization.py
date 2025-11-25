import os
import shutil
def organize_dataset(base_path):
    for split in ["train", "test1"]:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            continue

        # Create class folders
        cat_folder = os.path.join(split_path, "cats")
        dog_folder = os.path.join(split_path, "dogs")
        os.makedirs(cat_folder, exist_ok=True)
        os.makedirs(dog_folder, exist_ok=True)

        # Move files
        for filename in os.listdir(split_path):
            filepath = os.path.join(split_path, filename)

            # skip already-sorted folders
            if os.path.isdir(filepath):
                continue

            if filename.startswith("cat"):
                shutil.move(filepath, os.path.join(cat_folder, filename))
            elif filename.startswith("dog"):
                shutil.move(filepath, os.path.join(dog_folder, filename))

    print("Dataset organized successfully!")

organize_dataset("dogs-vs-cats")

train_path = "dogs-vs-cats/test1"

# for filename in os.listdir(train_path):
#     filepath = os.path.join(train_path, filename)

#     # Skip folders
#     if os.path.isdir(filepath):
#         continue

#     # Delete loose .jpg files
#     if filename.endswith(".jpg"):
#         os.remove(filepath)

# print("Loose JPG files deleted.")


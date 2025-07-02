import os
import pandas as pd

def generate_test_csv(test_dir="../data/test", output_path="../data/test.csv"):
    rows = []

    for folder in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, folder)
        if os.path.isdir(folder_path):
            try:
                with open(os.path.join(folder_path, "file_1.txt"), encoding="utf-8") as f1, \
                     open(os.path.join(folder_path, "file_2.txt"), encoding="utf-8") as f2:
                    text_1 = f1.read().strip()
                    text_2 = f2.read().strip()
                    article_id = int(folder.split("_")[-1])
                    rows.append({
                        "id": article_id,
                        "text_1": text_1,
                        "text_2": text_2
                    })
                    print(f"✅ Loaded {folder}")
            except Exception as e:
                print(f"❌ Failed to load {folder}: {e}")

    df = pd.DataFrame(rows)
    df.sort_values("id", inplace=True)
    df.to_csv(output_path, index=False)
    print(f"\n📁 Saved test CSV to: {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_test_csv()

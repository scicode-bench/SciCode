import os
import filecmp

def compare_common_files(dir1, dir2):
    # Collect files (ignoring subdirectories) from both directories
    files1 = {f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f))}
    files2 = {f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f))}

    # Compare only files that appear in both directories
    common_files = files1 & files2
    counter = 0
    for filename in common_files:
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)
        # Compare file contents (shallow=False ensures thorough comparison)
        if not filecmp.cmp(path1, path2, shallow=False):
            return False
        else:
            counter += 1
    print(f"Total common files: {counter}")

    return True

dir1 = "/eagle/tpc/zilinghan/SciCode/eval_results/prompt/dummy/without_background"
dir2 = "/eagle/tpc/zilinghan/SciCode/integration/inspection_ai/tmp/prompt/without_background"

print(compare_common_files(dir1, dir2))

dir1 = "/eagle/tpc/zilinghan/SciCode/eval_results/generated_code/dummy/without_background"
dir2 = "/eagle/tpc/zilinghan/SciCode/integration/inspection_ai/tmp/generated_code/without_background"

print(compare_common_files(dir1, dir2))
# Videos Directory

This directory contains video files used for testing the vehicle re-identification system.

## Note
Video files are excluded from Git repository due to GitHub's file size limitations (100MB limit).

## Setup
To use the system:

1. Add your video files to this directory
2. Update the default paths in the main scripts if needed:
   - `main.py`: Update `DEFAULT_TGT_VID_B`
   - `main_reid.py`: Update `DEFAULT_REF_VID_A` and `DEFAULT_TGT_VID_B`
   - `main_live_reid.py`: Update `DEFAULT_REF_VID_A` and `DEFAULT_TGT_VID_B`

## Video Format Support
- MP4 (recommended)
- AVI
- MOV
- Any format supported by OpenCV

## Example Usage
```bash
# Using videos from this directory
python main_reid.py -r videos/reference.mp4 -t videos/target.mp4
```

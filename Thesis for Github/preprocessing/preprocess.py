# preprocessing script for mostgan-v 
# including centre,cropping, border removal, frame and segment extraction, audio removal and standardisation of resolution
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4"}

# Haar cascade for face detection ships with opencv-python itself, no separate download needed.
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# preset face detection

# run helps run command lines in our shell and automatically extracts the output of whatever command 
# capture_output = capture the output into python variables 
# return the output as a readable string stored as result
# unpacks any other extra options passed into run()

# in case of fail raise runtimeerror: exit code + full command line of error 
# return the output until the error and then error after 
def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result

# from the function title path is a hint that it should be a path object to a file for example 
# and it will return a float for duration 
# ffprobe reads information about the video file
# -v verbosity level set to error otherwise it will output lots of clutter
# format = duration - just pick out the duration 
# -of is output formatting and the following default etc is what you want the output to be ie no wrapper and no key so you just get the number 
# video path is the file you want to inspect 
# strip will get rid of the trailing /n that will be part of the output and is converted into a float rather than string
def get_duration_seconds(video_path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ])
    return float(result.stdout.strip())

# same idea as get_duration_seconds but asks ffprobe for width/height instead
# -select_streams v:0 means "just look at the first video stream" (ignores audio streams etc)
# -of csv=s=x:p=0 formats the output as just "WIDTHxHEIGHT" e.g. "640x480", no extra labels
# split on the "x" to get the two numbers back out as separate ints
def get_video_dimensions(video_path: Path):
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", str(video_path),
    ])
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)

# Uses ffmpeg's cropdetect filter on the first few seconds to find black borders (it only prints suggested crop boxes to stderr, doesn't crop itself parse the last/most stable line).
# Compares the detected size against the original and skips if the difference is under 2% avoiding unnecessary work on videos with no real borders.


def detect_border_crop(video_path: Path, sample_seconds: float = 2.0):
    orig_w, orig_h = get_video_dimensions(video_path)

    result = subprocess.run(
        ["ffmpeg", "-t", str(sample_seconds), "-i", str(video_path),
         "-vf", "cropdetect=24:16:0", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # cropdetect prints its findings to stderr as it analyses frames
    crop_lines = [line for line in result.stderr.splitlines() if "crop=" in line]
    if not crop_lines:
        return None

    last_line = crop_lines[-1]
    crop_str = last_line.split("crop=")[-1].split()[0]  # e.g. "400:480:120:0"
    w, h, x, y = (int(v) for v in crop_str.split(":"))

    # only bother re-encoding if the detected content region is meaningfully smaller
    shrink_w = (orig_w - w) / orig_w
    shrink_h = (orig_h - h) / orig_h
    if shrink_w < 0.02 and shrink_h < 0.02:
        return None

    return w, h, x, y

#performs the border removal, producing a separate temp file rather than
# modifying the original, since extract_frames() can then just treat this temp file exactly like a normal video for everything later without needing to juggle coordinate math across multiple chained crops
# tmp_dir - a hidden folder next to the output, just for these temporary files
# -c:v libx264 -preset fast -crf 18 - re-encode reasonably fast, good quality, since this is
# a throwaway intermediate file that only needs to survive until frames are extracted from it
def strip_borders_to_temp(video_path: Path, crop_box, tmp_dir: Path) -> Path:
    """Produce a border-free copy of video_path in tmp_dir, using the given (w, h, x, y) crop box."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    w, h, x, y = crop_box
    tmp_path = tmp_dir / f"{video_path.stem}_borderfree{video_path.suffix}"
    run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an",
        str(tmp_path),
    ])
    return tmp_path

# again input_dir has to be a path to a file - input_dir means a folder 
# iterdir goes through each file in the folder 1 by 1 
# if p is a file not a folder and the extension is one of in the video extensions return the sorted alphabetical list 
def find_videos(input_dir: Path):
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

# five inputs - where to start looking (start), how long to look for (duration), how much space around the detected face to give (margin) with a default value
# how many points across the clip to check (num_samples) with a default value 
def detect_face_crop_box(
    video_path: Path,
    start: float,
    duration: float,
    margin: float = 0.6,
    num_samples: int = 5,
):

    # cap opens a video file 
    # fps asks the video the fps and defaulted to 25 if none 
    # frame_w asks the video what the width is 
    # frame_h asks the video what the height is 
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# time stamps across the clips (5 by default) 
# i + 0.5 is used to sample evenly the 0.5 means that for the clip duration we landing at the middle point of the clip
    sample_times = [start + duration * (i + 0.5) / num_samples for i in range(num_samples)]
    # empty list before detection loops 
    boxes = []  # each entry: (center_x, center_y, size) in pixels

# sample times is the mid point of each clip 
# frame_idx is the frame number from the fps and the sample_time that has been chosen 
# cap.set - cap opens the video object and set changes the property of the video in question with the current frame and changes it to 
# frame number - frame_idx 
# cap.read() produces two output a boolean if succesfull and frame which is the image data 
    for t in sample_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
# convert color to grayscale as haar cascade works with greyscale images 
# face_cascade returns a list of detected faces, scaleFactor smaller = more thorough checking but slower, minNieghbours - higher = fewer false positives but 
# chance to miss some real faces, minSize ignores anything smaller than 40x40 pixels 
# if nothing detected then continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            continue

        # if multiple faces detected, use the largest (most likely the main subject)
        # given one face tuple return width times height (face rectangle with the biggest area)
        # cx cy is a mid point 
        # this gets added to the empty boxes list
        # recap this
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cx, cy = x + w / 2, y + h / 2
        size = max(w, h)
        boxes.append((cx, cy, size))
# release the video file 
    cap.release()
# what do if no faces were found - return none as the list is empty
    if not boxes:
        return None

    # average (smooth) across all valid samples -> one stable box for the whole clip
    # this is the average position of the face detected - x and y along with the size of the box 
    avg_cx = sum(b[0] for b in boxes) / len(boxes)
    avg_cy = sum(b[1] for b in boxes) / len(boxes)
    avg_size = sum(b[2] for b in boxes) / len(boxes)

    # expand by margin to include shoulders/gestures, then make it square
    # expands the face size by the margin 
    square_size = avg_size * (1 + margin)

    # clamp so the crop box stays within the frame bounds
    # choose the smallest of the three values to make sure the crop box isnt bigger than the frame size 
    # start from the centre and move left by half the box to find the outer edge, if the calculation is negative then make it 0 
    # the furthest right part ensures that right edge at max is on the frames edge 
    square_size = min(square_size, frame_w, frame_h)
    crop_x = min(max(avg_cx - square_size / 2, 0), frame_w - square_size)
    crop_y = min(max(avg_cy - square_size / 2, 0), frame_h - square_size)
# returns top left x, top left y and the size of the square 
    return int(crop_x), int(crop_y), int(square_size)

# ten inputs - which video to read, where to save, target resolution, strip borders bool, video_override path
# speech dataset: originally, border detection+stripping ran fresh for EVERY segment of a long
# recording, re-encoding the ENTIRE source video each time just to pull out one 3-second piece -
# for a 58-segment clip that's up to 58 full re-encodes of the whole clip. video_override lets
# process_long_dataset resolve borders ONCE per source video and pass that already-clean path
# in directly, skipping border detection entirely for every individual segment call.
def extract_frames(
    video_path: Path,
    out_dir: Path,
    resolution: int,
    fps: float,
    start: float = None,
    duration: float = None,
    crop_mode: str = "center",
    face_margin: float = 0.6,
    strip_borders: bool = True,
    video_override: Path = None,
):
    """Extract frames from (a segment of) a video: crop to square, resize, save as PNGs."""
    # create the output folder if it doesnt exist 
    out_dir.mkdir(parents=True, exist_ok=True)

# NEW: effective_path starts off as just the original video - but if border stripping is on
# and detect_border_crop() finds a meaningful border, we create a border-free temp copy and
# point effective_path at THAT instead, so everything below (face detection, final ffmpeg
# extraction) automatically works on the clean version without needing separate logic
    effective_path = video_path
    tmp_borderfree_path = None
    if video_override is not None:
        # NEW: caller already resolved (and will clean up) the border-free version -
        # use it directly, skip border detection/stripping entirely for this call
        effective_path = video_override
    elif strip_borders:
        border_crop_box = detect_border_crop(video_path)
        if border_crop_box is not None:
            tmp_dir = out_dir.parent / ".tmp_borderfree"
            tmp_borderfree_path = strip_borders_to_temp(video_path, border_crop_box, tmp_dir)
            effective_path = tmp_borderfree_path
            print(f"    Detected and stripped black borders: crop={border_crop_box}")

# start by assuming no face based crop box yet 
# checks crop_mode is face
# use start if given 
# use given duration if given 
# calls function walked through earlier 
# NOTE: these now all use effective_path instead of video_path, so face detection runs on
# the border-free version if one was created above, not the original bordered footage
    crop_box = None
    if crop_mode == "face":
        clip_start = start if start is not None else 0.0
        clip_duration = duration if duration is not None else get_duration_seconds(effective_path)
        crop_box = detect_face_crop_box(effective_path, clip_start, clip_duration, margin=face_margin)
# if crop_box has a value unpack the tuple and builds the ffmpege crop filter 
    if crop_box is not None:
        x, y, size = crop_box
        crop_expr = f"crop={size}:{size}:{x}:{y}"
# run this when cropbox is none - either in centre mode or crop_box has no face detection 
    else:
        if crop_mode == "face":
            print(f"    WARNING: no face detected, falling back to center-crop for {video_path.name}")
        # center-crop: square using the shorter side - the expression to be used in ffmpeg - crop square of side size being the shorter of either the height/width
        crop_expr = "crop='min(in_w,in_h)':'min(in_w,in_h)'"

    vf = f"{crop_expr},scale={resolution}:{resolution},fps={fps}"
# program name + -y to automate confirmation otherwise it would ask for confirmation 
    cmd = ["ffmpeg", "-y"]
# if start value given -ss flag to start seeking and the start position
# adds -i which is input file into cmd regardless and followed by the str(effective_path) -
# NOTE: this was video_path before, now effective_path so we actually read from the
# border-free temp file when one exists
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(effective_path)]
# add -t how longg fffmpeg should read the file along with the duration
# -vf vdideo filter, vf the filter that was built, -an discard the audio and finally the output output directory + the numbering 4 numbers e.g. 0001
    if duration is not None:
        cmd += ["-t", str(duration)]
# output from PNG to JPEG for memory savings
    cmd += ["-vf", vf, "-an", "-q:v", "2", str(out_dir / "%04d.jpg")]
# run this command into run that we defined previously 
    run(cmd)

# NEW: clean up the temp border-free file now that frames have been extracted from it -
# missing_ok=True just means "don't error if it's somehow already gone"
    if tmp_borderfree_path is not None:
        tmp_borderfree_path.unlink(missing_ok=True)

# count number of png files in output folder 
# glob is a genersator but list turns this into a readable list 
# if nothing has been output delete the empty folder
# process short + long dataset both uses n_frames to determine whether to count the video clip 
    n_frames = len(list(out_dir.glob("*.jpg")))
    if n_frames == 0:
        shutil.rmtree(out_dir, ignore_errors=True)
    return n_frames

# five inputs - input folder, where the output should be saved, target resolutionand fps and which crop mode centre or face 
# call the find videos function defined earlier, if no videos found end function early with 0
def process_short_dataset(input_dir: Path, output_dir: Path, resolution: int, fps: float, crop_mode: str):
    videos = find_videos(input_dir)
    if not videos:
        print(f"  WARNING: no video files found in {input_dir}")
        return 0

# wrapped in try/except - a corrupted or unreadable file (e.g. incomplete upload)
# would otherwise crash the WHOLE batch and lose progress on every remaining clip.
# Now it just prints an error and moves on to the next video instead.
    total_clips = 0
    for video_path in videos:
        clip_id = video_path.stem
        out_dir = output_dir / clip_id
        try:
            n_frames = extract_frames(video_path, out_dir, resolution, fps, crop_mode=crop_mode)
        except RuntimeError as e:
            print(f"  {video_path.name} -> SKIPPED (error reading file: {e})")
            continue
        status = f"{n_frames} frames" if n_frames > 0 else "SKIPPED (no frames)"
        print(f"  {video_path.name} -> {clip_id}/ ({status})")
        if n_frames > 0:
            total_clips += 1
    return total_clips

# same inputs with an additional segment_seconds e.g. how the clip should be split 

def process_long_dataset(
    input_dir: Path, output_dir: Path, resolution: int, fps: float, segment_seconds: float, crop_mode: str
):
    videos = find_videos(input_dir)
    if not videos:
        print(f"  WARNING: no video files found in {input_dir}")
        return 0

# we get the duration of each video and divide it by the segments to determine how many segments make up one clip 
# wrapped in try/except - same reasoning as process_short_dataset, so one corrupted
# recording doesn't crash the whole batch and lose progress on every remaining video
    total_clips = 0
    for video_path in videos:
        try:
            duration = get_duration_seconds(video_path)
        except RuntimeError as e:
            print(f"  {video_path.name} -> SKIPPED (error reading file: {e})")
            continue
        n_segments = int(duration // segment_seconds)
        print(f"  {video_path.name}: {duration:.1f}s -> {n_segments} segments of {segment_seconds}s")

# border-stripping ONCE per source video here, before looping over its segments
        effective_video_path = video_path
        tmp_borderfree_path = None
        if crop_mode in ("face", "center"):  
            border_crop_box = detect_border_crop(video_path)
            if border_crop_box is not None:
                tmp_dir = output_dir / ".tmp_borderfree"
                tmp_borderfree_path = strip_borders_to_temp(video_path, border_crop_box, tmp_dir)
                effective_video_path = tmp_borderfree_path
                print(f"    Detected and stripped black borders (once for whole clip): crop={border_crop_box}")

# determine  the starting point of each segment 
# create a unique name for each segment without the extension through stem
# determine the output path 
# call the extract frames function however we want to extract the frames per segment which is why we have the start and duration in this function
        for seg_idx in range(n_segments):
            start = seg_idx * segment_seconds
            clip_id = f"{video_path.stem}_seg{seg_idx:04d}"
            out_dir = output_dir / clip_id
            try:
                n_frames = extract_frames(
                    video_path, out_dir, resolution, fps,
                    start=start, duration=segment_seconds, crop_mode=crop_mode,
                    video_override=effective_video_path,
                )
            except RuntimeError as e:
                print(f"    segment {seg_idx:04d} -> SKIPPED (error: {e})")
                continue
            if n_frames > 0:
                total_clips += 1

# clean up per-video temp border-free file now that every segment of  video is done (not per-segment - this file is shared across all of this video's segments)
        if tmp_borderfree_path is not None:
            tmp_borderfree_path.unlink(missing_ok=True)
    return total_clips

# takes 8 inputs: 
# name of the dataset, mode - short or long processing, the input folder, output folder, target resolution and fps, length of segments and the crop mode
def process_dataset(name, mode, input_dir, output_dir, resolution, fps, segment_seconds, crop_mode):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
# checks the input folder exists 
    if not input_dir.exists():
        print(f"SKIPPING '{name}': input dir not found: {input_dir}")
        return
# print what is processing and its details 
# create the output folder 
    print(f"\n=== Processing '{name}' (mode={mode}, crop_mode={crop_mode}) ===")
    output_dir.mkdir(parents=True, exist_ok=True)

# determine which mode is to be used
    if mode == "short":
        total = process_short_dataset(input_dir, output_dir, resolution, fps, crop_mode)
    elif mode == "long":
        total = process_long_dataset(input_dir, output_dir, resolution, fps, segment_seconds, crop_mode)
    else:
        raise ValueError(f"Unknown mode: {mode} (expected 'short' or 'long')")

    print(f"=== '{name}' done: {total} clips written to {output_dir} ===")


# list of dictionaries for speeches and actions
DEFAULT_DATASETS = [
    {"name": "speech", "mode": "long", "crop_mode": "face",
     "input_dir": "data/raw_clips/speech", "output_dir": "data/processed/speech"},
    {"name": "actions", "mode": "short", "crop_mode": "center",
     "input_dir": "data/raw_clips/actions", "output_dir": "data/processed/actions"},
]


def main():
# argeparse to show what command line commands are accepted by the script along with explanations of each argument 
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, default=256, help="Output frame resolution (square)")
    parser.add_argument("--fps", type=float, default=25.0, help="Output frame rate")
    parser.add_argument("--segment_seconds", type=float, default=3.0, help="Segment length for 'long' mode datasets")
    parser.add_argument("--datasets_json", type=str, default=None,
                        help="Optional path to a JSON file overriding the default dataset list")

    parser.add_argument("--single", type=str, default=None, help="Process only this dataset name")
    parser.add_argument("--mode", type=str, choices=["short", "long"], help="Required with --single")
    parser.add_argument("--crop_mode", type=str, choices=["center", "face"], default="center",
                        help="Crop strategy: 'center' or 'face' (only used with --single)")
    parser.add_argument("--input_dir", type=str, help="Required with --single")
    parser.add_argument("--output_dir", type=str, help="Required with --single")

# args ensures the command line used is in line with the arguments defined above
    args = parser.parse_args()

# check that the tools used are installed and available
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            print(f"ERROR: '{tool}' not found on PATH. Install ffmpeg first (e.g. `conda install -c conda-forge ffmpeg`).")
            sys.exit(1)

# if the single flag is used but the required flags are not included include an error with the flags required
# return after the single dataset has been processed
    if args.single:
        if not (args.mode and args.input_dir and args.output_dir):
            parser.error("--single requires --mode, --input_dir, and --output_dir")
        process_dataset(args.single, args.mode, args.input_dir, args.output_dir,
                         args.resolution, args.fps, args.segment_seconds, args.crop_mode)
        return

# open the file, if its a json file it is opened and converted into a python list structure through json.load()
# json can be used to test different structures for our model without changing the pre processing model entirely 
    datasets = DEFAULT_DATASETS
    if args.datasets_json:
        with open(args.datasets_json) as f:
            datasets = json.load(f)

# we pass the values from our dataset and the shared command line settings
    for ds in datasets:
        process_dataset(
            ds["name"], ds["mode"], ds["input_dir"], ds["output_dir"],
            args.resolution, args.fps, args.segment_seconds, ds.get("crop_mode", "center"),
        )


if __name__ == "__main__":
    main()
